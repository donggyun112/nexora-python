# ADR-005: LangGraph 실행 경로를 전면 철회한다

**상태:** Accepted (2026-08-05)

**대체:** ADR-001, ADR-003, ADR-004의 실행 엔진 결론을 이 문서로 통합한다.

**보강:** 뒤집힘 조건 2·3·5는 [ADR-007](adr-007-everything-nondeterministic-is-a-step.md)이
검토 후 닫았다. 남는 재검토 방아쇠는 Platform 운영 구매라는 사업 결정뿐이다.

## 맥락

Nexora에는 같은 durable tool round를 수행하는 두 구현이 존재했다.

1. `Orchestrator.execute_round`의 직접 경로
2. 주입된 LangGraph effect `StateGraph`

직접 경로는 이미 다음 계약을 소유한다.

- pending call order 기록
- `StepLog`의 pending/running/done과 call-id 멱등성
- lease와 fencing
- 기본 순차 실행 및 명시적 concurrency-safe 병렬 실행
- suspension snapshot 저장과 worker 없는 resume
- `recover_pending`의 완료 결과 재사용과 미실행 effect 복구

따라서 LangGraph 도입 여부는 새 기능을 구축하는 선택이 아니라, 동일 의미를 가진 두 실행 경로 중
어느 하나를 계속 소유할지의 선택이 되었다.

## 측정

로컬 zero-I/O 측정:

| 축 | 직접 경로 | LangGraph 경로 |
|---|---:|---:|
| 라운드 지연 | 약 260µs | 1.58ms, InMemorySaver 1.76ms |
| RSS | 기준 약 26.5MB | 약 40MB 추가 |
| 최초 로드 | 약 0.02s | compile+invoke 약 0.24s |
| 의존성 | `langchain-core` | 현재 lock 기준 12개 패키지 추가 |

실제 model/tool I/O에서는 지연 차이가 작다. 결정 근거는 의존성 크기보다 중복된 정확성 추론과
업그레이드 때마다 필요한 conformance 비용이다.

## 결정

- agent planner는 `engines/plain`의 `async while` 하나만 둔다.
- 모든 tool effect는 Python `Orchestrator.execute_round`를 통과한다.
- `StepLog`를 effect 복구의 단일 source of truth로 둔다.
- LangGraph agent engine, effect graph, checkpointer 주입 API와 `langgraph` 의존성을 제거한다.
- gate가 재평가되어도 effect 결과와 외부 부작용은 call-id StepLog로 재사용한다.
- suspension은 `persist_suspension`을 커밋한 뒤 `AgentSuspended`로 attempt를 종료한다.

```text
plain agent planner
        ↓ effect intent
Orchestrator: record_pending → gate → Stepped effect → result/suspend
        ↓
StepLog + typed tool executor
```

## 왜 graph checkpointer가 한계 효용이 작은가

외부 effect 성공과 graph checkpoint 사이의 장애 창은 checkpointer가 증명할 수 없다. LangGraph
경로에서도 StepLog, fencing, call-id 멱등성과 `recover_pending`은 제거할 수 없었다. Graph가 추가로
보존한 것은 gate/effect node의 program counter였지만, 직접 경로는 완료 effect를 StepLog에서
재사용하고 미완료 gate를 안전하게 재평가해 같은 외부 결과를 만든다.

철회로 받아들이는 차이는 checkpoint되지 않은 gate의 재평가다. Gate는 외부 effect를 실행하지 않는
결정 함수여야 하며, 규칙 변경을 resume에 반영하는 기존 정책과도 일치한다.

## 결과

좋은 결과:

- 실행·복구 의미를 한 코드 경로에서만 유지한다.
- 배포 크기, import 비용과 LangGraph superstep 결합을 제거한다.
- planner는 얇고 교체 가능한 의도 결정 계층으로 남는다.
- multi-agent와 subagent도 같은 mediated effect 경계를 재사용할 수 있다.

감수하는 결과:

- gate 장애 후 해당 durable round의 gate를 다시 평가할 수 있다.
- 모든 model/control 지점의 자동 program-counter 복구를 제공하지 않는다.
- transcript는 별도 저장소가 구현될 때까지 복구 호출자가 명시적으로 제공한다.

## 뒤집힘 조건

다음 조건이 실제 측정으로 충족되면 새 ADR에서 재검토한다.

1. 라운드당 gate 수와 gate 재평가 비용이 운영상 유의미해진다.
2. model invocation을 durable Effect로 만들면서 관리할 program counter가 크게 늘어난다.
3. nested subagent, 병렬 branch/join, selective retry 또는 tool gate 밖의 interrupt가 실제
   제품 요구가 되어 부모·자식 control state를 별도로 복구해야 한다.
4. StepLog 의미를 보존하면서 실질적인 복구 코드 삭제가 가능하다.
5. transcript checkpoint 쓰기 증폭을 해결한 저장 구조가 먼저 존재한다.

재검토는 기능 개수를 세어 자동으로 채택하지 않는다. 같은 장애 주입 시나리오를 현재 직접 경로,
LangGraph Functional API, 필요하면 직접 조립한 `StateGraph`에 적용해 다음을 함께 측정한다.

- 완료된 model/task와 병렬 branch의 재실행 횟수
- transcript/checkpoint 쓰기 증폭
- 삭제되는 Nexora 복구 코드와 새로 생기는 적응 코드
- `StepLog`의 intent, call-id, lease, fencing 의미 보존 여부
- checkpointer와 `StepLog` 사이에 중복된 replay source of truth가 생기는지 여부

LangGraph는 이 비교에서 control-state durability를 맡아 기존 코드를 실제로 줄이고, 외부 effect의
복구 권한은 `StepLog`에 남길 수 있을 때만 채택한다.
