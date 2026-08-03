# ADR-001: 기본 엔진은 평범한 `async while`

**상태**: Accepted — 근거는 3차 전면 교체 (2026-08-03)
**범위**: `nexora.engines`, `nexora.contracts`, `AGENTS.md`

> **이력.** 이 ADR은 세 번 다시 쓰였고, 그때마다 근거가 틀린 것으로 드러났다.
>
> 1. **초판**: 직접 `StateGraph`를 조립하는 비용을 근거로 들었다 → 허수아비였다. 아무도 그렇게 안
>    쓴다. 실제 대안은 `create_agent` + 미들웨어고, 거기선 그 비용을 안 낸다.
> 2. **2판**: "LangChain의 훅 여섯 개로는 react.ts 시맨틱을 다 표현할 수 없다" →
>    [실험](../../experiments/2026-08-03-langgraph-conformance.md)에서 **반증**됐다. 12/12 통과.
> 3. **현재**: 남았던 "계약 소유"와 "의존성"도 스스로 뒤집었다 — LangChain의 메시지·프로바이더
>    계층을 채택했고 `langchain-core`는 코어 의존성이다.
>
> 남은 근거는 하나뿐이고, 사실이 아니라 **가독성 선호**다. 그렇게 적는다.

---

## 결정

기본 엔진은 `nexora.engines.plain` — 평범한 `async while`. `nexora.engines.langgraph`는
`create_agent` 위에 같은 동작을 구현해 나란히 유지한다. 둘은 같은 입력에서 같은 이벤트를 낸다.

## 남은 유일한 근거

**제어 흐름이 그 자리에 쓰여 있는가.**

react.ts의 규칙 하나 — *"도구 호출이 없으면 종료. 단 종료 직전에 steer가 도착했으면 한 번 더"*
(L150-154):

```python
if not requested:
    messages.append(AIMessage(turn_text))
    if drain_steers and (steers := drain_steers()):
        messages += steers
        continue
    yield await _done(emit, turn_text, calls_made, "completed", spent)
    return
```

LangGraph 엔진에서 같은 규칙은 `_Steering.after_model`이 `jump_to: "model"`을 반환하는 형태가
되고, 종료 판정은 아예 다른 클래스(`_RoundEnd`)로 간다.

측정된 차이는 **41줄**이다. LangChain에 after-tools 훅이 없어서, `before_model`에서 메시지 꼬리의
`ToolMessage`를 보고 "방금 도구가 돌았다"를 되추론해야 한다 (`_trailing_tool_messages`,
`_RoundEnd.abefore_model`, `_as_call`). 여기에 `jump_to`가 종료 이유를 못 날라서 사이드 채널
dataclass(`_Outcome`)가 붙는다.

**동작은 같다. 읽는 방식이 다르다.** 그게 전부고, 그걸로 기본값을 정한다.

## 두 엔진을 다 유지하는 이유

- **적합성 스위트가 근거를 계속 검증한다.** `tests/test_engine_conformance.py`가 두 엔진을
  parametrize로 돌린다. 갈리면 실패다. 선호를 주장이 아니라 측정 가능한 상태로 둔다.
- **durable suspend/resume이 아직 없다.** LangGraph는 체크포인터와 `interrupt`를 이미 준다.
  그게 필요해질 때 기본 엔진을 바꿀 수 있다.

## 채택한 것 — LangChain 메시지·프로바이더 계층

`BaseMessage` / `ToolCall` / `BaseChatModel`을 쓴다. 우리 타입을 갖고 있던 값이:

- 양방향 번역 계층
- `ChatOpenAI`를 다시 구현한 프로바이더 어댑터 ~120줄
- 스트리밍 도구 인자를 조립하는 `ModelTurn` 66줄

세 개 다 삭제했다. `AIMessageChunk` 덧셈이 조각난 인자를 조립하고, `bind_tools`가 스키마를
넘긴다. 타입 자체는 우리가 가치를 더하던 부분이 아니었다.

**우리 것으로 남는 것**: 훅 여섯 개(`aborted`/`before_tool_call`/`emit`/`drain_steers`/
`should_stop_after_turn`/`on_suspend`), 이벤트 어휘, react.ts 시맨틱.

## 대가

- **`langchain-core`가 코어 의존성**이다. 이전 근거 3을 스스로 뒤집었다.
- **라운드 경계 영속화가 없다.** transcript를 write-ahead log로 쓰는 방향이나 미구현.
- **`resumeContext` 미포팅** (`react.ts` L67-83).
- **`on_suspend`가 LangGraph 엔진에 없다.** 이벤트는 나가지만 스냅샷 인계가 없다.

## 뒤집힐 조건

- **durable suspend/resume이 요구사항이 될 때.** LangGraph가 이미 갖고 있고 우리는 없다. 이게
  가장 그럴듯한 경로다.
- **LangChain에 after-tools 훅이 생길 때.** 41줄이 0이 되고 마지막 차이가 사라진다.
- **두 엔진 유지 비용이 적합성 스위트의 값을 넘을 때.**

## 참고

- [실험: LangGraph 적합성](../../experiments/2026-08-03-langgraph-conformance.md) — 12/12, 예측 6개 오답
- `packages/architectures/src/react.ts` — 이식 대상 (293줄)
- `langchain/agents/middleware/types.py` — 훅 여섯 개와 `jump_to`
