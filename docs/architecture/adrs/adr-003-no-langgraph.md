# ADR-003: LangGraph를 쓰지 않는다. `langchain-core`는 쓴다

**상태**: Accepted (2026-08-03) — [ADR-001](adr-001-plain-loop-over-graph-engine.md)을 대체
**범위**: `nexora.engines`, `pyproject.toml`

---

## 결정

| | |
|---|---|
| **유지** | `langchain-core` — `BaseMessage`, `ToolCall`, `BaseChatModel` |
| **폐기** | `langgraph`, `langchain`(agents) — `create_agent`, `ToolNode`, 체크포인터 |

"LangChain을 버린다"가 아니다. **모델·메시지 계층은 남기고 실행 계층만 버린다.**

## 근거

### 1. 메시지·모델 계층은 값을 한다 — 이건 유지한다

채택하면서 삭제한 것: 프로바이더 어댑터 120줄, 스트리밍 도구 인자를 조립하던 `ModelTurn` 66줄.
`AIMessageChunk` 덧셈이 조각난 인자를 합치고 `bind_tools`가 스키마를 넘긴다. 프로바이더 커버리지는
덤이다. 우리가 가치를 더하던 부분이 아니었다.

### 2. `create_agent`의 값은 셋으로 좁혀진다

루프, `ToolNode`, 체크포인터. 그래프 배선은 이 셋의 부산물이다.

### 3. 루프 — 차이는 취향이다

[적합성 실험](../../experiments/2026-08-03-langgraph-conformance.md): 12/12 동등. 차이는
after-tools 훅이 없어 생기는 되추론 41줄뿐이다. **사실이 아니라 가독성 선호**이므로 결정 근거로
쓰지 않는다.

### 4. `ToolNode` — 우리 계약과 기본값이 반대다

`ToolNode`는 배치를 **항상** `asyncio.gather`로 돌리고 옵트아웃이 없다
([tool_node.py:858](https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/tool_node.py)).
우리 계약은 fail-closed 순차다([ADR-002](adr-002-retry-safety-needs-order-determinism.md)).

우회는 가능하다. 도구를 dict 스키마로 넘기면 `ToolNode`가 아예 생기지 않고
([factory.py:1052](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py)),
`after_model`에서 우리가 실행하면 된다. 실측으로 확인했고 **코드가 60줄 줄어든다**.

**그런데 우회하면 `ToolNode`가 주는 것이 0이 된다.** 남는 건 3과 5뿐이다.

### 5. 체크포인터 — transcript와 겹친다

체크포인터가 저장하는 것과 우리가 이미 가질 것:

| 저장 항목 | 우리 쪽 |
|---|---|
| `channel_values` (메시지) | transcript가 가짐 |
| 다음 노드 | 마지막 메시지에서 파생 |
| `turn` / `calls_made` / usage | 파생 가능 |
| `pending_writes` | 도구 결과를 완료 즉시 기록하면 등가 |

`interrupt()`는 **노드 중간 재개**를 위해 체크포인터를 요구한다. 우리는 노드 중간에서 재개하지
않는다 — react.ts의 suspend는 run을 끝내고, 답이 오면 tool_result를 주입해 새 run을 시작한다.
배치 경계에서만 멈춘다.

그리고 체크포인터는 우리가 **실제로 필요한 둘을 안 준다**: 중복 재개를 막는 lease, 도구 실행 순서.

대신 없으면 안 생길 실패 모드를 하나 추가한다 —
[langgraph#7417](https://github.com/langchain-ai/langgraph/issues/7417): 하트비트 타임아웃으로
체크포인트에서 재발행되어, 원본이 아직 실행 중인데 도구가 다시 돈다. at-least-once는 문서화된
동작이지만, **재발행 경로 자체가 체크포인터에서 나온다.**

### 6. 결론

4와 5를 빼면 `create_agent`는 `while` 하나짜리 루프를 위한 그래프 배선만 남긴다.

## 적응 비용의 증거

`ToolNode`에 우리 실행 계약을 끼워넣는 과정에서 한 세션에 버그 4개가 나왔다. 넷 다 이벤트 타입
개수를 세는 테스트로는 안 보였다.

| 버그 | 결과 |
|---|---|
| 게이트 payload에 `turn` 누락 | 두 엔진 이벤트 불일치 |
| 도구 이름을 call id로 전달 | 멱등키 소실 — 재시도를 새 호출로, 다른 호출을 같은 것으로 |
| 렌더링을 결과로 발행 | **실패한 도구 호출이 감사 로그에 성공으로 기록** |
| `can_jump_to` 미선언 | `jump_to` 무시 — terminating tool 이후에도 모델 계속 호출 |

셋은 우리 어댑팅 코드 버그지 LangChain 버그가 아니다. 그래도 **적응 표면이 실재한다**는 증거다.
우회(4)하면 앞의 셋은 애초에 생기지 않는다.

## 반증 조건

- **노드 중간 재개가 필요해질 때.** 도구 하나가 몇 시간짜리고 그 안에서 멈췄다 이어가야 하면
  체크포인터가 유일한 답이다.
- **transcript를 만들지 않기로 할 때.** 그러면 체크포인터가 유일한 durability다.
- **LangGraph Cloud에 배포할 때.** 운영 표면 전체가 체크포인터 전제다.
- **`ToolNode`에 동시성 정책 옵션이 생길 때.** 4가 사라진다.

## 이 문서의 신뢰도에 대하여

이 세션에서 제기된 LangGraph 반대 논거 중 **여덟 개가 틀렸다** — 그래프가 상태 필드를 강제한다,
훅 여섯 개로 표현 불가, 코드가 더 적다, ReAct 방식이 다르다, 병렬을 잃는다, 순차가 불가능하다,
계약을 소유해야 한다, 의존성이 무겁다. 실험과 실측으로 전부 반증되거나 스스로 뒤집혔다.

살아남은 근거(4의 기본값 충돌, 5의 중복)는 **검토 과정에서 반대편이 제기한 것**이다. 그 이력을
남겨두는 이유는, 이 문서를 나중에 읽는 사람이 근거의 출처를 보고 무게를 다시 재기 위해서다.

전체 이력: [ADR-001](adr-001-plain-loop-over-graph-engine.md)의 3차 재작성 기록과
[실험 문서](../../experiments/2026-08-03-langgraph-conformance.md)의 예측 대조표.

## 후속

- `engines/langgraph`를 삭제하지 않고 `docs/experiments/`의 참조물로 강등한다. 반증 조건이
  실현되면 여기서 다시 시작한다.
- `engines/plain`이 유일한 엔진이 되면 "plain"이라는 이름은 의미를 잃는다. `nexora.loop`로 되돌린다.
- 적합성 스위트는 비교 대상이 사라지면 단일 엔진 테스트가 된다. 발견한 버그 4개에 대한 payload
  단언은 유지한다 — 우리 엔진에도 같은 종류의 실수가 가능하다.
