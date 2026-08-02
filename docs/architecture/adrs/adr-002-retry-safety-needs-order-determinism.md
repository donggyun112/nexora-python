# ADR-002: 재시도 안전성은 멱등성만으로 부족하다 — 순서 결정성이 필요하다

**상태**: Accepted (2026-08-03)
**범위**: `nexora.tools`, `contracts/types.py`, `AGENTS.md`, [ADR-001](adr-001-plain-loop-over-graph-engine.md)

---

## 논거 (단계별)

1. **재시도는 일어난다.** 크래시 후 재개, 워커 교체, 그리고 실제 관측 사례로
   [langgraph#7417](https://github.com/langchain-ai/langgraph/issues/7417) — 3분 넘는 도구 호출이
   하트비트 타임아웃으로 **원본이 아직 실행 중인데** 체크포인트에서 재발행된다. 둘 다 완료된다.

2. **재시도는 배치 단위다.** 완료 기록이 없는 호출들이 다시 실행된다. 하나가 아니라 여럿이다.

3. **호출별 멱등성은 배치를 안전하게 만들지 못한다.** 같은 자원을 건드리는 두 호출은 각자 멱등이어도
   순서가 바뀌면 최종 상태가 다르다.

   ```
   [write(f, "A"), write(f, "B")]
     1차:   A → B   ⇒  f = "B"
     재시도: B → A   ⇒  f = "A"
   ```

   각 `write`는 몇 번을 해도 같은 결과다. 그런데 배치 결과가 다르다.

4. **병렬 실행은 완료 순서를 보장하지 않는다.** `asyncio.gather`는 시작 순서만 정하고 완료는
   스케줄링에 달렸다. 같은 입력으로 두 번 돌려도 인터리빙이 다를 수 있다.

5. **∴ 재시도가 안전하려면 순서가 결정적이어야 하고, 그건 순차 실행을 뜻한다** — 그 배치 안에서
   서로 간섭하지 않는다고 선언된 호출만 예외다.

6. **TypeScript Nexora가 이미 그렇게 한다.** `tool.ts`의 기본값은 *"fail-closed"*로 명시돼 있다:

   ```
   isReadOnly=false, isConcurrencySafe=false, isExclusive=false, isDestructive=false
   ```

   `isConcurrencySafe`: *"True if safe to run concurrently with other tools. **Default: false
   (sequential).**"* 병렬은 도구 저자가 명시적으로 옵트인하는 것이지 기본이 아니다.

## 결정

**배치는 기본적으로 순차 실행한다.** 병렬은 도구가 `is_concurrency_safe`를 선언한 경우에만,
그리고 같은 배치의 다른 호출들도 모두 선언한 경우에만 적용한다.

`AGENTS.md`의 *"External side effects must be idempotent"*는 유지하되 그것만으로 충분하다는
함의를 지운다. 두 가지가 다 필요하다:

- **호출별 멱등성** — 같은 `tool_call.id`로 두 번 실행돼도 한 번의 효과
- **배치 순서 결정성** — 재시도가 같은 순서로 실행

## ADR-001에 주는 영향

이것이 `engines/plain`을 남기는 첫 번째 **정확성** 근거다. 그전까지의 근거는 가독성뿐이었다.

LangGraph의 `ToolNode`는 배치를 **항상** `asyncio.gather`로 돌리고
([tool_node.py:858](https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/tool_node.py))
순차 옵션이 없다. Python 저장소에 해당 이슈조차 없고, JS 쪽에만
[#303](https://github.com/langchain-ai/langgraphjs/issues/303) ·
[#861](https://github.com/langchain-ai/langgraphjs/issues/861)이 열려 있다.

`wrap_tool_call`에서 공유 `asyncio.Lock`을 잡으면 순차가 되는 것은 실측으로 확인했다
(`start a / end a / SUSPEND b / BLOCKED c`). 하지만:

- 기본값이 순차라면 **거의 모든 호출이 락을 잡는다.** `ToolNode`의 병렬성은 실질적으로 안 쓰인다.
- 그러면서 `ToolNode`가 `gather`를 쓰고 `wrap_tool_call`이 그 안에서 불린다는 **문서화되지 않은
  구현 세부**에 의존하게 된다.

즉 프레임워크의 기본값을 매 호출 뒤집으면서, 그 대가로 아무것도 얻지 않는다.

## 대가

- 순차가 기본이면 **느리다.** 읽기 전용 도구 여러 개를 동시에 돌리는 흔한 이득을 놓친다.
  → 완화: `is_concurrency_safe`를 선언한 도구들끼리는 묶어서 병렬로.
- `is_concurrency_safe`가 아직 구현돼 있지 않다. 현재 `tools.execute_calls`는 `execute_batch`가
  없으면 무조건 순차이며, 이 ADR은 그 기본값을 정당화할 뿐 병렬 경로를 아직 만들지 않았다.

## 뒤집힐 조건

- `ToolNode`에 순차/동시성 정책 옵션이 생기면, LangGraph 채택의 이 걸림돌이 사라진다.
- 배치 안에서 자원 충돌을 정적으로 판정할 수 있게 되면(도구가 접근 범위를 선언한다든지), 순서
  결정성 없이도 안전한 병렬이 가능해진다.
