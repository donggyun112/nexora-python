# ADR-004: LangGraph는 내부 Effect 오케스트레이터다

**상태:** **Superseded by [ADR-005](adr-005-withdraw-langgraph.md)** (2026-08-05)

## 맥락

Nexora의 제품 경계는 범용 workflow가 아니라 agent runtime이다. 기본 agent engine의 plain
`async while`은 읽기 쉽고 빠르지만, 권한 gate와 외부 Effect 사이의 durable program counter를
직접 구현하면 복구 상태 머신이 분산된다.

## 결정

공개 API는 `AgentRuntime`으로 유지한다. 내부 tool Effect 실행은 직접 조립한 LangGraph
`StateGraph`가 담당한다.

```text
plain while 또는 LangGraph agent engine
                ↓ tool Effect
LangGraph effect orchestrator
  persist_pending → gate* → execute_effect* → commit_round
                ↓
StepLog + 실제 tool executor
```

Agent engine과 orchestration engine은 별개의 선택이다.

- Agent engine 기본값: plain while
- Effect orchestration engine: LangGraph StateGraph
- LangGraph agent engine: 선택 가능하지만 공개 Runtime API는 동일

## 두 종류의 내구성

LangGraph checkpointer와 Effect ledger는 서로 대체하지 않는다.

- **Graph checkpointer:** 완료된 gate/effect node와 다음 program counter를 복구한다.
- **StepLog:** `pending/running/done`, call-id 멱등성, execute/graph-checkpoint 사이의 장애 창을
  처리한다.

외부 시스템이 성공한 뒤 graph node checkpoint 전에 서버가 죽으면 checkpointer만으로 실행 여부를
알 수 없다. 재개된 node가 동일 call id로 `StepLog`를 읽어 완료 결과를 재사용해야 한다.

## 실행 의미

- gate는 호출 순서대로 checkpoint된다.
- Effect는 기본 순차 실행이다.
- 배치의 모든 도구가 `is_concurrency_safe`를 선언한 경우만 한 node 안에서 병렬 실행한다.
- gate 장애는 실패한 gate node에서 재개한다.
- Effect 완료 후 journal 장애는 node를 재시도하되 실제 tool은 StepLog 결과를 재사용한다.
- `POST_TOOL_BATCH`는 agent engine이 한 번만 발행한다.

## 비용

로컬 zero-I/O 100라운드, best-of-three 측정:

| 경로 | 라운드당 |
|---|---:|
| direct while | 260µs |
| LangGraph orchestration, checkpointer 없음 | 1.58ms |
| `InMemorySaver` | 1.76ms |

실제 model/tool I/O에서는 작은 비율이지만, 순수 제어 루프에는 약 1.3–1.5ms가 추가된다.

## 남은 경계

현재 그래프가 소유하는 것은 tool Effect round다. Model invocation과 transcript store는 아직 agent
engine 경계에 있으므로 모든 model/control 지점의 자동 crash resume까지 제공한다고 주장하지 않는다.
다음 확장은 model invocation을 Effect 계약으로 만들고, growing history 전체를 매번 복사하지 않는
append-only transcript store를 붙이는 것이다.
