# ADR-001: 실행 엔진 — 그래프 대신 평범한 `async while`

**상태**: Accepted (구현 완료 — 2026-08-03)
**날짜**: 2026-08-03
**범위**: `nexora.loop`, `nexora.tools`, `nexora.types`, `AGENTS.md` 프로젝트 계약

> 포팅 시작 시점의 `AGENTS.md`는 *"LangGraph is the only execution engine. Build with the
> low-level `StateGraph` API"*를 계약으로 못박고 있었다. 실제로 얹어보고 걷어낸 기록이다.

---

## 컨텍스트

TypeScript Nexora의 `react.ts`는 293줄짜리 `for` 루프다. 이걸 Python으로 옮기면서 LangGraph
`StateGraph`를 실행 엔진으로 쓰기로 되어 있었다 — 체크포인트·interrupt/resume·pending writes를
직접 만들지 않아도 된다는 계산이었다.

`StateGraph`로 최소 그래프를 세우고, 상태 계약을 `TypedDict`로 펼치고, 종료 판정을 조건부 엣지로
옮기는 데까지 갔다가 전부 제거했다.

## 결정

**루프는 평범한 `async while`이다.** LangGraph를 포함해 어떤 그래프 엔진도 쓰지 않는다.
영속성은 루프 **안**이 아니라 **주위**에 얹는다.

오케스트레이터(다단계 워크플로우)는 이 ADR의 범위 밖이다. 다만 아래 근거 5는 거기에도 그래프가
필요하지 않을 수 있음을 시사한다.

## 근거

### 1. LangGraph가 실제로 파는 건 하나다

**라운드 경계 상태를 Postgres에 쓰는 것.** `langgraph-checkpoint-postgres`가 스키마·마이그레이션·
동시성을 갖고 있고, 그걸 직접 만들면 TS에서 `SuspendedTurnStore`(contract + json/pg/in-memory 3구현)
와 `WorkflowStateStore`를 쓴 만큼의 일이다.

`pending_writes`도 실측으로 확인했다 — 병렬 도구 2개 중 하나가 완료하고 다른 하나가 interrupt하면,
완료분이 체크포인트에 보존되고 재개 시 재실행되지 않는다. TS는 이걸 위해 별도 커밋(18파일,
`fix: make suspended tool batches atomic`)이 필요했다.

### 2. 치르는 값은 넷이다

**노드 재진입.** `interrupt()` 후 재개하면 노드를 처음부터 재실행한다. 그래서 노드 안의 모든
부수효과와 이벤트 발행이 멱등해야 하고, `event_id`가 재시도마다 동일하게 계산돼야 한다. 이벤트
시스템 전체가 이 요구에 물린다.

**상태 직렬화.** 지역변수가 될 것들이 전부 체크포인트 필드가 된다. `has_more`, `iteration`,
`pending`이 `AgentState` 항목이 되고, "model 노드가 `tool_results`를 비워야 한다" 같은 암묵 계약이
생긴다. 스코프가 사라지면 사람이 규칙을 지켜야 한다.

**스텝마다 전체 스냅샷.** 실측:

```text
step  1 | {"thread_id":"t1","input":"hello","phase":"completed","output":"HELLO"}
step  0 | {"thread_id":"t1","input":"hello","phase":"ready","branch:to:model":null}
step -1 | {"__start__":{"thread_id":"t1","input":"hello","phase":"ready"}}
```

`"hello"`가 세 번 저장돼 있다. 대화 이력을 상태에 넣으면 스텝 수 × 이력 크기로 불어난다.

**파라미터 이름 결합.** `add_node`의 `_Node` Protocol이 `def __call__(self, state: T)`라 파라미터
**이름**까지 계약이다. `Callable[[AgentState], ...]`는 positional-only로 취급돼 매칭에 실패하고,
`state`라는 이름을 강제하는 Protocol을 따로 선언해야 했다.

### 3. 안 사주는 것도 셋이다

- **취소 전파** — 분산 취소 기능이 없다. 협조적 체크·transport 취소 메시지·재개 시 플래그, 전부 직접.
- **중복 재개 방어** — TS `SuspendedTurnStore.claim()`이 `awaiting → resumed`를 원자적으로 전이시켜
  막던 것. LangGraph의 `channel_versions`는 **write 시점** 충돌 감지라 이미 늦다. 낙관적 동시성은
  덮어쓰기를 막지 중복 작업을 막지 못한다. 대화 단위 lease가 어느 쪽이든 따로 필요하다.
- **이벤트 발행** — 없다.

**`recursion_limit`도 안전망이 못 된다.** 실측 기본값이 10007이고, 도달하면 `GraphRecursionError`
예외로 터진다. 정상 종료가 아니라 그래프가 터지는 것이라 종료 사유가 남지 않는다.

### 4. 에이전트는 애초에 그래프가 아니다

엣지를 LLM이 런타임에 정한다. 그려봐야 `model ⇄ tools` 사이클 하나인데, 그건 그래프가 아니라
`while`이다. 구조가 없는 것을 구조 문법으로 표현하는 셈이다.

반대로 언어에 이미 있는 것을 데이터로 다시 만들어야 한다:

| 언어 | 그래프 |
|---|---|
| 지역변수 (스코프 있음) | 공유 상태 딕셔너리 (전역) |
| `if` | 조건부 엣지 + 라우터 함수 |
| `while` | 사이클 + 종료 판정 상태 필드 |
| 클로저 / `try·finally` | 대응물 없음 |

그리고 `react.ts`의 종료 조건은 7개다(abort / LLM 예외 / tool_call 없음 / suspend / 도구 주도 /
정책 훅 / 반복 상한). steering이 그중 하나를 취소할 수 있고, 이 전부가 조건부 엣지의 반환값
도메인으로 올라가야 한다. `while` 안에서는 그냥 `return`이다.

### 5. 재개는 그래프 없이도 된다

그래프가 파는 마지막 실용적 이점은 "실행 위치를 노드 id로 표현할 수 있다"인데, durable execution
계열(Temporal, Restate, Palantir Orchestrator)은 그걸 **저널**로 대체한다. 함수를 처음부터
재실행하고 이름 붙은 스텝은 저널에서 값만 꺼낸다 — "어디서 멈췄나"를 저장할 필요가 사라진다.

그리고 우리 히스토리에 이미 저널이 있다. `tool_use.id`가 스텝 이름이고 `tool_result`가 저널 값이다.
`tool_use`를 **실행 전에** 기록하면 그 저널이 성립한다.

## 대가

떠안은 것을 분명히 해둔다.

- **라운드 경계 영속화를 직접 만들어야 한다.** 아직 안 만들었다. transcript를 write-ahead log로 쓰는
  방향이고, 별도 스냅샷 테이블 없이 간다는 것이 현재 가정이다.
- **재개 경로가 없다.** `resumeContext`(`react.ts` L67-83)는 미포팅이다.
- **`pending_writes` 등가물이 없다.** 병렬 배치 중 일부 완료 상태에서 프로세스가 죽으면 그 라운드가
  통째로 유실된다. 도구 결과를 완료 즉시 기록하면 등가가 되지만 명시적 작업이다.

## 뒤집힐 조건

- 오케스트레이터(스텝 그래프, 팬아웃, 부분 실패 재개)를 만들 때. 다만 근거 5대로 거기서도 저널이
  더 맞을 수 있다.
- 사람이 GUI로 흐름을 편집해야 할 때, 또는 "어떤 경로가 어떤 데이터에 닿는가"의 정적 분석이
  컴플라이언스 요구가 될 때. 그래프가 진짜로 값을 하는 두 자리다.

## 실측 재현

이 ADR의 숫자는 langgraph 1.2.10 / langgraph-checkpoint-postgres 3.1 기준이며, 해당 의존성은
제거됐다. 재확인이 필요하면 임시로 설치해 `InMemorySaver`로 체크포인트를 `alist()` 하면 된다.

## 참고

- `packages/architectures/src/react.ts` — 이식 대상 루프
- `packages/contracts/src/suspended-turn.ts` — TS의 수제 체크포인트
- `packages/orchestrator/src/workflow-state-store.ts` — `WorkflowCheckpoint`
