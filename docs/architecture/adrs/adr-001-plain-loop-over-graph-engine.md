# ADR-001: 실행 엔진 — 그래프 대신 평범한 `async while`

**상태**: Accepted (구현 완료 — 2026-08-03)
**날짜**: 2026-08-03
**범위**: `nexora.loop`, `nexora.tools`, `nexora.types`, `AGENTS.md` 프로젝트 계약

---

## 한 줄 요약

에이전트 루프를 LangGraph `StateGraph`로 세웠다가 전부 걷어내고 `async while`로 다시 썼다.

## 컨텍스트

포팅 시작 시점의 `AGENTS.md`는 *"LangGraph is the only execution engine"*을 계약으로 갖고 있었다.
체크포인트와 중단/재개를 직접 만들지 않아도 된다는 계산이었다. 실제로 얹어보고 제거한 기록이다.

---

## 왜 걷어냈나 — 같은 규칙을 두 가지로 써보면

포팅해야 할 규칙 하나를 예로 든다. `react.ts`의 **"도구 호출이 없으면 종료한다. 단, 종료 직전에
사용자 메시지(steer)가 도착했으면 종료하지 않고 한 번 더 돈다"** (L150-154).

### `while`로 쓰면

```python
if not tool_calls:
    if steers := drain_steers():
        messages += steers
        continue          # 종료 취소, 다음 라운드
    return done()
```

읽으면 규칙 그대로다.

### 그래프로 쓰면

먼저 지역변수였던 것들이 전부 상태 필드가 된다. 그래프 노드는 서로 다른 함수 호출이라
지역변수를 공유할 수 없기 때문이다.

```python
class AgentState(TypedDict, total=False):
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    pending_steers: list[str]
    iteration: int
    max_iterations: int
    stop_requested_by_policy: bool
    suspended_call_id: str
    aborted: bool
```

그리고 종료 판정이 라우터 함수로 빠지는데, 여기서 **"지금 루프 어디쯤인가"를 상태로 되추론**해야
한다. `while`이라면 코드 위치가 곧 답인 것을:

```python
def next_step(state: AgentState) -> Literal["model", "tools", "end"]:
    # 도구가 방금 돌았나? 직접 알 방법이 없어서 tool_results가 비었는지로 추측한다.
    # 그래서 "model 노드는 시작할 때 tool_results를 비워야 한다"는 암묵 계약이 생긴다.
    if state.get("tool_results"):
        ...
    if not state.get("tool_calls"):
        if state.get("pending_steers"):
            return "model"
        return "end"
    return "tools"

builder.add_conditional_edges("model", next_step, {...})
```

규칙 하나가 세 곳(상태 정의 / 라우터 / 엣지 배선)에 흩어지고, 코드에 없는 규칙("비워야 한다")이
사람 머릿속에 남는다. `react.ts`의 종료 조건은 **7개**다 — abort, LLM 예외, 도구 호출 없음,
suspend, 도구 주도 종료, 정책 훅, 반복 상한. 전부 이 대접을 받는다.

---

## 그래프를 쓰면 생기는 일 넷

### 1. 재개하면 노드를 처음부터 다시 실행한다

`interrupt()`로 멈췄다가 재개하면 그 노드가 **재진입**한다. 중간부터가 아니라 처음부터다.

그래서 노드 안에서 뭔가 발행했다면 두 번 발행된다. 승인 요청 이벤트, 감사 로그, 트랜스크립트
기록 전부. 막으려면 모든 이벤트에 "재시도해도 같게 계산되는 id"가 필요해지고, 외부 전송은 그
id로 중복 제거해야 한다. **이벤트 시스템 설계 전체가 이 요구 하나에 묶인다.**

`while`에는 재진입이 없다. 이벤트는 한 번 나간다.

### 2. 저장량이 스텝 수에 비례해 불어난다

체크포인트는 매 스텝 **상태 전체**를 다시 쓴다. 실측:

```text
step  1 | {"thread_id":"t1","input":"hello","phase":"completed","output":"HELLO"}
step  0 | {"thread_id":"t1","input":"hello","phase":"ready","branch:to:model":null}
step -1 | {"__start__":{"thread_id":"t1","input":"hello","phase":"ready"}}
```

`"hello"` 하나가 세 번 저장돼 있다. 대화 이력을 상태에 넣으면 `스텝 수 × 이력 크기`가 된다.

### 3. 반복 상한이 안전망이 못 된다

`recursion_limit` 실측 기본값이 **10007**이고, 도달하면 `GraphRecursionError` **예외**로 터진다.
정상 종료가 아니라 그래프가 죽는 것이라 종료 사유가 안 남는다.

### 4. 파라미터 이름까지 계약이다

`add_node`는 노드 함수의 파라미터 **이름**을 검사한다. 그래서 이게 통과 안 된다:

```text
error: No overload variant of "add_node" matches argument types
       "str", "Callable[[AgentState], Awaitable[AgentStateUpdate]]"
```

`state`라는 이름을 강제하는 Protocol을 따로 선언해야 했다.

---

## 그래도 그래프가 주는 것 — 하나

**라운드 경계 상태를 Postgres에 저장해주는 것.** `langgraph-checkpoint-postgres`가 스키마·
마이그레이션·동시성을 갖고 있다.

덤으로 `pending_writes`가 있다. 병렬 도구 2개 중 하나가 완료하고 다른 하나가 멈추면 완료분이
보존돼 재개 시 재실행되지 않는다. 실측으로 확인했고, TS는 같은 걸 손으로 만드느라 별도
커밋(18파일, `fix: make suspended tool batches atomic`)을 썼다.

## 그런데 이건 안 준다

- **취소 전파** — 분산 취소 기능이 없다. 직접 만들어야 한다.
- **중복 재개 방어** — TS `SuspendedTurnStore.claim()`이 하던 것. LangGraph의 `channel_versions`는
  저장 시점에 충돌을 감지하는데, 그때는 이미 두 워커가 LLM을 부르고 도구를 실행한 뒤다. 중복
  **저장**은 막아도 중복 **작업**은 못 막는다. 대화 단위 락이 어느 쪽이든 따로 필요하다.
- **이벤트 발행** — 없다.

## 결론

테이블 하나와 `save`/`load`/`claim` 세 메서드를 안 짜려고, 이벤트 시스템 전체를 멱등 설계로 바꾸고
지역변수를 전부 상태 필드로 펼쳤다. **남는 장사가 아니다.** 게다가 `claim()`은 LangGraph에 없어서
어차피 직접 짜야 한다.

### 그리고 에이전트는 애초에 그래프가 아니다

다음에 뭘 할지 LLM이 런타임에 정한다. 그려봐야 `model ⇄ tools` 사이클 하나이고, 그건 그래프가
아니라 `while`이다.

### 재개는 그래프 없이도 된다

그래프의 마지막 실용적 이점은 "멈춘 위치를 노드 id로 표현할 수 있다"인데, Temporal이나 Palantir
Orchestrator 같은 durable execution은 그걸 **저널**로 대체한다. 함수를 처음부터 재실행하되 이미
끝난 스텝은 저널에서 값만 꺼낸다. 멈춘 위치를 저장할 필요 자체가 사라진다.

그리고 우리 히스토리에 이미 저널이 있다 — `tool_use.id`가 스텝 이름이고 `tool_result`가 그 값이다.
`tool_use`를 **실행 전에** 기록하면 성립한다.

---

## 대가 — 우리가 떠안은 것

- **라운드 경계 영속화를 직접 만들어야 한다.** 아직 안 만들었다. transcript를 write-ahead log로
  쓰는 방향이고, 별도 스냅샷 테이블 없이 간다는 것이 현재 가정이다.
- **재개 경로가 없다.** `resumeContext`(`react.ts` L67-83) 미포팅.
- **`pending_writes` 등가물이 없다.** 병렬 배치 중 일부만 완료한 상태에서 프로세스가 죽으면 그
  라운드가 통째로 유실된다. 도구 결과를 완료 즉시 기록하면 등가가 되지만 명시적 작업이다.

## 뒤집힐 조건

- 오케스트레이터(스텝 그래프, 팬아웃, 부분 실패 재개)를 만들 때. 다만 거기서도 저널이 더 맞을 수
  있다 — Palantir는 오케스트레이션도 straight-line 코드로 쓴다.
- 사람이 GUI로 흐름을 편집해야 할 때, 또는 "어떤 경로가 어떤 데이터에 닿는가"의 정적 분석이
  컴플라이언스 요구가 될 때. 그래프가 진짜로 값을 하는 두 자리다.

---

## 부록: 실측 재현

숫자는 langgraph 1.2.10 / langgraph-checkpoint-postgres 3.1 기준이고 해당 의존성은 제거됐다.
재확인하려면 임시 설치 후 `InMemorySaver`로 체크포인트를 `alist()`하면 된다.

## 참고

- `packages/architectures/src/react.ts` — 이식 대상 루프 (293줄)
- `packages/contracts/src/suspended-turn.ts` — TS의 수제 체크포인트
- `packages/orchestrator/src/workflow-state-store.ts` — `WorkflowCheckpoint`
