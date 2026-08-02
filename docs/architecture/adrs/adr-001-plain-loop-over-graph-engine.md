# ADR-001: 실행 엔진 — 프레임워크 루프 대신 평범한 `async while`

**상태**: Accepted (구현 완료 — 2026-08-03)
**날짜**: 2026-08-03
**범위**: `nexora.loop`, `nexora.tools`, `nexora.types`, `AGENTS.md` 프로젝트 계약

> **정정 (2026-08-03):** 초판은 *직접 `StateGraph`를 조립하는 것*과 `while`을 비교했다. 그건
> 허수아비였다 — LangGraph 위에서 에이전트를 만드는 실제 방식은 `create_agent(model, tools,
> middleware=[...])`이고, 거기서는 초판이 나열한 비용(상태 필드 펼치기, 라우터 함수, 엣지 배선,
> 파라미터 이름 결합)을 **내지 않는다.** langchain-ai/deepagents를 읽고 확인했다. 아래는 올바른
> 비교로 다시 쓴 것이며, 결론은 유지하되 근거가 바뀌었다.

---

## 한 줄 요약

에이전트 루프를 LangGraph 위에 세웠다가 걷어내고 `async while`로 다시 썼다. 이유는 "그래프가
무겁다"가 아니라 **훅 집합이 우리 것이 아니어서**다.

## 컨텍스트

포팅 시작 시점의 `AGENTS.md`는 *"LangGraph is the only execution engine. Build with the low-level
`StateGraph` API"*를 계약으로 갖고 있었다. 그 지시대로 직접 조립했고, 그게 첫 번째 오판이었다.

---

## 1단계: 직접 `StateGraph`를 짠 것 — 이건 그냥 하면 안 되는 일이었다

포팅해야 할 규칙 하나를 예로 든다. `react.ts` L150-154의 **"도구 호출이 없으면 종료. 단 종료 직전에
사용자 메시지(steer)가 도착했으면 종료하지 않고 한 번 더 돈다."**

`while`로 쓰면 규칙 그대로다:

```python
if not tool_calls:
    if steers := drain_steers():
        messages += steers
        continue          # 종료 취소
    return done()
```

직접 그래프로 옮기면 지역변수가 전부 상태 필드가 되고(노드끼리 지역변수를 공유할 수 없으니),
"지금 루프 어디쯤인가"를 상태에서 되추론해야 한다:

```python
def next_step(state: AgentState) -> Literal["model", "tools", "end"]:
    # 도구가 방금 돌았나? 직접 알 방법이 없어 tool_results가 비었는지로 추측한다.
    # → "model 노드는 시작할 때 tool_results를 비워야 한다"는, 코드에 없는 규칙이 생긴다.
    if state.get("tool_results"):
        ...
    if not state.get("tool_calls"):
        return "model" if state.get("pending_steers") else "end"
    return "tools"
```

규칙 하나가 상태 정의 / 라우터 / 엣지 배선 세 곳으로 흩어진다. `react.ts`의 종료 조건은 7개다.

**하지만 이건 LangGraph를 반대할 근거가 못 된다.** 아무도 이렇게 안 쓴다. `AGENTS.md`가 그렇게
쓰라고 해서 그렇게 쓴 것뿐이고, 그 지시가 틀렸다.

---

## 2단계: 실제 대안 — `create_agent` + 미들웨어

LangChain의 `create_agent`가 루프와 그래프를 만들어주고, 사용자는 미들웨어만 조립한다.
deepagents의 `graph.py`는 944줄인데 그래프를 만드는 건 맨 끝 다섯 줄이다:

```python
return create_agent(model, tools, middleware=deepagent_middleware, ...)
```

훅은 여섯 개다 — `before_agent` / `before_model` / `after_model` / `after_agent` /
`wrap_model_call` / `wrap_tool_call`. 훅은 `{"jump_to": "tools" | "model" | "end"}`로 흐름을 바꾼다.
상태 필드도, 라우터도, 엣지 배선도 직접 쓰지 않는다.

### 그래서 react.ts의 규칙들을 여기 얹어보면

| # | 규칙 | 얹을 자리 |
|---|---|---|
| 1 | abort | `before_model` → `jump_to: end` |
| 2 | LLM 예외 → error 이벤트 | `wrap_model_call`이 잡아 대체 반환 |
| 3 | 도구 호출 없음 → 종료 | `create_agent` 내장 |
| 4 | suspend (사람 대기) | `after_model`의 `interrupt()` |
| 5 | exclusive 단독 실행 | `after_model`이 `tool_calls`를 하나로 트림 |
| 6 | steer 흡수 (라운드 진입) | `before_model`이 메시지 주입 |
| 7 | steer 흡수 (종료 직전) | `after_model` → `jump_to: model` |
| 8 | **도구 주도 종료 (`terminatesLoop`)** | **자리 없음** |
| 9 | **정책 훅 (`shouldStopAfterTurn`)** | **자리 없음** |

**`after_tools` 훅이 없다.** 8과 9는 둘 다 "도구가 방금 돌고 난 직후"에 판정해야 하는데 그 시점에
걸 훅이 없다. 차선은 다음 라운드의 `before_model`에서 하는 것이고, 그러려면 **"방금 도구가
돌았나"를 상태에서 되추론**해야 한다 — 1단계에서 봤던 그 문제가 프레임워크를 제대로 써도
그대로 나온다.

(`wrap_tool_call`이 `ToolMessage | Command`를 반환할 수 있어 8은 우회할 여지가 있다. 확인하지
않았다.)

---

## 결정과 근거

**루프는 평범한 `async while`이다.** 근거는 셋이다.

**1. 훅 집합이 우리 것이 아니다.** 9개 중 2개가 갈 곳이 없다. 그리고 이건 일회성 비용이 아니라
앞으로 포팅할 규칙마다 "이게 여섯 훅 중 어디에 들어가지?"를 묻는 일이 된다. 안 맞으면 상태
되추론으로 우회하고, 우회할 때마다 코드에 없는 규칙이 하나씩 늘어난다. `while`에서는 그 규칙이
있어야 할 줄에 그냥 쓴다.

**2. `AGENTS.md`가 Nexora 소유라고 못박은 것과 겹친다** — tool ordering, permissions, streaming
contract. `create_agent`가 루프를 소유하면 이 셋이 미들웨어 표현력의 제약 아래 놓인다. 소유권
경계를 프레임워크가 긋게 된다.

**3. 의존성과 스트림 소유.** `langchain` + `langgraph` + `langchain-core`가 코어 의존성이 되고,
방출하는 이벤트가 LangChain의 것이 된다. 우리는 이벤트 계약을 직접 정의하기로 했다.

### 그리고 에이전트는 애초에 그래프가 아니다

다음에 뭘 할지 LLM이 런타임에 정한다. 그려봐야 `model ⇄ tools` 사이클 하나이고, 그건 그래프가
아니라 `while`이다. `create_agent`가 그래프를 감춰주는 것도 결국 이 사실의 반증이 아니라 확인이다.

### 재개는 그래프 없이도 된다

그래프의 마지막 실용적 이점은 "멈춘 위치를 노드 id로 표현할 수 있다"인데, Temporal이나 Palantir
Orchestrator 같은 durable execution은 그걸 **저널**로 대체한다. 함수를 처음부터 재실행하되 이미 끝난
스텝은 저널에서 값만 꺼낸다 — 멈춘 위치를 저장할 필요 자체가 사라진다.

그리고 우리 히스토리에 이미 저널이 있다. `tool_use.id`가 스텝 이름이고 `tool_result`가 그 값이다.
`tool_use`를 **실행 전에** 기록하면 성립한다.

---

## deepagents에서 인정할 것

같은 문제를 먼저 풀었고, **승인 게이트는 우리와 같은 결론**에 도달해 있다.

```python
decisions = interrupt(hitl_request)["decisions"]   # langchain HumanInTheLoopMiddleware:315
```

동기 블로킹이 아니라 `interrupt` — 체크포인트를 남기고 워커를 반납한다. TS Nexora의
`transport.request(timeout=5분)` 방식이 아니라서 승인이 며칠 걸려도 된다. 우리도 같은 결론이지만,
**그쪽은 배관을 프레임워크에서 받고 우리는 직접 만들어야 한다.** 아직 안 만들었다.

배울 점 둘:

- **배치 단위로 한 번 묻는다.** 승인 필요한 호출을 모아 한 번의 `interrupt`로 올리고 결정을 배열로
  받는다. 우리는 콜 단위로 게이트를 돌리고 첫 suspend에서 끊는다. 사람 입장에선 프롬프트가 한 번인
  쪽이 낫다.
- **`edit` 결정이 있다.** 사람이 인자를 고쳐서 통과시키는 경로. 우리 게이트는 allow/deny/ask 셋뿐이라
  이게 없다. 있어야 한다.

`_fs_interrupt.py` 183줄은 전부 "이 경로가 승인 대상인가" 판정이다 — `glob`의 `pattern`이 `path`를
우회하는 케이스, `path="."` 우회 방지, pathless `grep`은 무조건 발동. **그래프와 무관한 정책
코드고, 우리도 똑같이 필요하다.**

---

## 대가 — 우리가 떠안은 것

- **라운드 경계 영속화를 직접 만들어야 한다.** 아직 없다. transcript를 write-ahead log로 쓰는
  방향이고, 별도 스냅샷 테이블 없이 간다는 것이 현재 가정이다.
- **suspend/resume 배관이 없다.** 게이트가 `suspend` 결과를 내고 `on_suspend`로 넘기는 데까지만
  구현됐다. 저장·조회·재개는 미구현이고 `resumeContext`(`react.ts` L67-83)도 미포팅이다.
- **`pending_writes` 등가물이 없다.** 병렬 배치 중 일부만 완료한 상태에서 프로세스가 죽으면 그
  라운드가 통째로 유실된다. 도구 결과를 완료 즉시 기록하면 등가가 되지만 명시적 작업이다.

## 뒤집힐 조건

- **`after_tools` 훅이 생기면** 근거 1이 약해진다. 9개 중 2개가 안 맞는 것이 0개가 되고, 그러면
  근거는 2·3만 남는다.
- 오케스트레이터(스텝 그래프, 팬아웃, 부분 실패 재개)를 만들 때. 다만 거기서도 저널이 더 맞을 수
  있다 — Palantir는 오케스트레이션도 straight-line 코드로 쓴다.
- 사람이 GUI로 흐름을 편집해야 하거나, "어떤 경로가 어떤 데이터에 닿는가"의 정적 분석이 컴플라이언스
  요구가 될 때. 그래프가 진짜로 값을 하는 두 자리다.

---

## 부록: 직접 조립했을 때 관측한 것

1단계에서 실제로 본 것들. `create_agent`를 쓰면 해당 없지만, 왜 직접 조립을 멈췄는지의 기록이다.
langgraph 1.2.10 / langgraph-checkpoint-postgres 3.1 기준이고 해당 의존성은 제거됐다.

- **체크포인트는 매 스텝 상태 전체를 다시 쓴다.** `"hello"` 하나가 세 스텝에 세 번 저장돼 있었다.
- **`recursion_limit` 기본값이 10007**이고, 도달하면 `GraphRecursionError` 예외로 터진다. 정상 종료가
  아니라서 종료 사유가 남지 않는다.
- **`add_node`가 파라미터 이름을 검사한다.** `Callable[[AgentState], ...]`는 positional-only로
  취급돼 매칭에 실패하고, `state`라는 이름을 강제하는 Protocol을 따로 선언해야 했다.
- **`pending_writes`는 실제로 값을 한다.** 병렬 도구 2개 중 하나가 완료하고 다른 하나가 interrupt하면
  완료분이 보존돼 재개 시 재실행되지 않는다. TS는 같은 걸 손으로 만드느라 별도 커밋(18파일,
  `fix: make suspended tool batches atomic`)을 썼다.
- **`channel_versions`는 중복 재개를 막지 못한다.** 저장 시점에 충돌을 감지하는데 그때는 이미 두
  워커가 LLM을 부르고 도구를 실행한 뒤다. 중복 저장은 막아도 중복 작업은 못 막는다. 대화 단위 락이
  어느 쪽이든 따로 필요하다.

## 참고

- `packages/architectures/src/react.ts` — 이식 대상 루프 (293줄)
- `langchain-ai/deepagents` `libs/deepagents/deepagents/graph.py` — `create_agent` 조립
- `langchain/agents/middleware/types.py` — 훅 여섯 개와 `jump_to`
- `langchain/agents/middleware/human_in_the_loop.py` — `interrupt` 기반 승인 게이트
- `packages/contracts/src/suspended-turn.ts` — TS의 수제 체크포인트
