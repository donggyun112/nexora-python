# ADR-003: LangGraph를 쓰지 않는다. `langchain-core`는 쓴다

**상태**: **Superseded by [ADR-005](adr-005-withdraw-langgraph.md)** (2026-08-05)
**범위**: `nexora.engines`, `pyproject.toml`

---

## 결정

| | |
|---|---|
| **유지** | `langchain-core` — `BaseMessage`, `ToolCall`, `BaseChatModel` |
| **폐기** | `create_agent`, `ToolNode`, 체크포인터. `langgraph` extra는 참조 엔진용으로 유지 |

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

### 4. ~~배치 단위 정책을 표현할 자리가 없다~~ → `create_agent`에서만 없다

> **두 번 정정됐다.**
>
> 1. (2026-08-03) 처음 *"`ToolNode`가 항상 `gather`이고 옵트아웃이 없다"*로 적혀 있었다 → 우리가
>    쓰는 경로에서는 틀렸다. 실제 원인은 팬아웃 위치였다.
> 2. (2026-08-04) **그 팬아웃도 `create_agent`의 선택이었다.** 직접 `StateGraph`를 조립하면
>    `Send`를 쓰지 않고 노드가 라운드 전체를 받으므로 `execute_calls`를 `while` 엔진과 똑같이 부른다
>    — 배치 정책이 그대로 표현된다. **이 근거는 더 이상 langgraph 전체에 대한 근거가 아니다.**
>
> 아래 본문은 `create_agent` 경로에 한정해 읽어야 한다. 전문은 [§정정 기록](#정정-기록-2026-08-03).

우리 계약은 fail-closed 순차이고([ADR-002](adr-002-retry-safety-needs-order-determinism.md)),
병렬은 실행기가 **배치 단위로** 옵트인한다(`Tools.execute_batch`).

**순서는 문제가 아니었다.** `create_agent`는 한 라운드를 호출당 하나의
`Send("tools", [call])`로 팬아웃하고([factory.py:1881](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py)),
그래프 실행기가 그 태스크들을 세마포어로 게이팅한다([_executor.py:135](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_executor.py)).
`config={"max_concurrency": 1}` 한 줄로 호출 순서대로 순차가 된다 — 실측. 엔진에 적용했고
`test_a_batch_runs_one_call_at_a_time`이 두 엔진에 대해 고정한다.

**실제 문제는 팬아웃 위치다.** 배치가 노드에 닿기 전에 엣지에서 해체된다. 실측: 호출 3개짜리
라운드에서 `ToolNode._afunc`가 **1개씩 3번** 불린다. 그래서:

| 표현하려는 것 | 왜 안 되는가 |
|---|---|
| `execute_batch` 위임 (실행기가 그룹 결정) | 노드도 훅도 배치를 못 봄 |
| 쌍 단위 충돌 (`write` 둘, 각자 safe) | 단항 플래그로 표현 불가 — 관계 속성이다 |
| 게이트 전원 통과 후 실행([tools.py:155](../../../src/nexora/tools.py)) | 게이트가 호출별로 흩어짐 |

닫히지 않는 우회들 — 전부 실측:

- **공유 락 (`awrap_tool_call`)**: 순서는 복구되나 `execute_batch`가 안 불려 옵트인 병렬 상실
  (22ms → 79ms, 도구 N개면 N배). 게이트-먼저 불변식도 복구 안 됨.
- **플래그 조건부 락**: 직렬화 자체가 안 된다. safe 도구가 락을 안 잡으므로 unsafe가 경합 없이
  획득해 그냥 병렬로 돈다.
- **커스텀 `ToolNode`**: `create_agent(tools=...)`는 `Sequence[BaseTool | Callable | dict]`만 받고
  `ToolNode`를 자기가 만든다 — `TypeError: 'ToolNode' object is not iterable`. 넣을 자리가 없고,
  넣어도 호출을 1개씩 받는다.
- **`after_model`에서 배치를 그룹으로 쪼개 다음 라운드로 미루기**: 그래프가 `model → tools → model`
  이라 **그룹마다 모델 호출이 한 번 더 든다.** 토큰을 쓰는 우회다.

`max_concurrency`는 `astream` 시점의 run 단위 상수라 *"항상 순차"* 또는 *"항상 병렬"*만 말할 수
있다. 모델이 무엇을 부를지 모르는 시점이므로 배치별 선택이 아니다.

**그리고 순서는 재시도 안전성만의 문제가 아니다.** 루프가 오케스트레이터의 스텝이 되면(§5),
리플레이는 **스텝 시퀀스가 동일해야** 성립한다. 배치 순서가 실행마다 달라지면 `run()` 호출 순서가
달라지고 리플레이가 깨진다. 즉 순서 결정성은 [ADR-002](adr-002-retry-safety-needs-order-determinism.md)의
재시도 요건이면서 **durable replay의 전제조건**이다. 그 프레임에서 무조건 `gather`는 불편이 아니라
실격이다.

**upstream 요구는 `ToolNode`가 아니라 엣지에 있다** — `_make_model_to_tools_edge`가 만드는 `Send`
묶음을 정할 수 있어야 한다:

```python
create_agent(..., tool_batch_policy=lambda calls: [[a, b], [c]])
```

### 5. 체크포인터 — 리플레이 권한은 하나여야 한다

> **재작성 (2026-08-03).** 이 항목은 *"체크포인터가 transcript와 겹친다"*였다. 논증은 맞았지만
> **미구현 기능(transcript)에 기대고 있었고**, 그래서 약했다. 오케스트레이터를 위에 두는 아키텍처가
> 정해지자 같은 결론이 미구현에 안 기대는 형태로 정리된다. 아래가 그것이다.

루프는 durable orchestrator의 **스텝 하나**다. durability는 루프 안이 아니라 그 밖에 있다:

```
supervisor = durable orchestrator
  run("draft", react_loop)        ← 루프 통째로 한 스텝, 안에 durability 없음
  run(call_id, tools.execute)     ← 외부 효과 있는 도구만 자기 스텝. 스텝 이름 = 멱등키
  signal(f"approve:{call_id}")    ← HITL은 그래프 상태가 아니라 이름으로 기다리는 외부 이벤트
```

durable execution 패턴이다(Temporal / Azure Durable Functions / DBOS 계열). `run(name, fn)`이
메모이즈되고, 리플레이 때 완료된 스텝은 기록된 값을 돌려주고 재실행하지 않는다.

**그러면 체크포인터는 우리 루프와 경합하는 게 아니라 오케스트레이터와 경합한다.** 스텝 경계에서
리플레이를 소유하는 주체가 있는데 그 스텝 **안에** 두 번째 리플레이 권한을 두면, 중복이 아니라
충돌이다. 어느 쪽이 진실인지 정하는 규칙이 없다.

그리고 `interrupt()`가 요구하는 **노드 중간 재개는 스텝으로 얻는다** — 몇 시간짜리 도구는 그 도구를
자기 스텝으로 만들면 되고, 그래프를 체크포인트할 필요가 없다.

### 정확한 경계 (2026-08-04, 검토에서 정정)

앞선 판이 두 군데를 과장했다. 정확한 문장은 이것이다:

> **LangGraph checkpointer는 워크플로 상태와 완료 task를 복원하지만, OSS checkpointer 단독으로는
> effect intent와 실행 lease를 제공하지 않는다.** Nexora의 비멱등 외부 효과에는 별도의 intent
> journal, lease/heartbeat/fencing, 그리고 reconciliation 또는 receiver-side idempotency가
> 필요하다.

| 보장 | OSS checkpointer | `StepLog` |
|---|---|---|
| 워크플로 상태 영속 | ✅ | 별도 구현 |
| HITL resume | ✅ `interrupt()` | 별도 구현 |
| 완료된 task 재사용 | ✅ 태스크 단위 `pending_writes` | ✅ |
| **시작했지만 완료 불명** | 재실행 가능 | `running → Indeterminate` |
| **동일 run 실행 잠금** | ❌ (Agent Server에는 있다) | ✅ lease + fencing |
| 임의 외부 효과 exactly-once | ❌ | ❌ |

두 가지를 고쳤다:

1. *"체크포인터에 lease가 없다"* → **OSS checkpointer 단독**에 없다. LangGraph **Agent Server**는
   thread별 run lease를 갖는다. [#7417](https://github.com/langchain-ai/langgraph/issues/7417)도
   "lease가 전혀 없다"의 증거가 아니라 heartbeat 회수 중 살아 있는 작업이 재발행된 사례다.
2. *"`StepLog`이 exactly-once를 준다"* → **아니다.** intent 저장 → 약국 전송 성공 → `finish` 전에
   크래시하면 전송 여부를 모른다. 다시 보내지 않고 `Indeterminate`로 멈출 뿐이다. 정확한 이름은
   **durable intent + ambiguity detection + exclusive execution**이고, 자동 시도 기준으로는
   at-most-once다. 진짜 exactly-once는 수신 시스템의 멱등키·transactional outbox·조정 프로토콜이
   있어야 한다.

**그래서 둘을 같이 써도 된다.** LangGraph가 graph state를, `StepLog`이 외부 effect만 소유하면 두
번째 권한이 아니라 계층 분리다. 지금 붙이지 않은 이유는 그 분리가 필요한 요구가 아직 없다는 것뿐이다.

남는 사실 두 개:

- 체크포인터는 우리가 **실제로 필요한 둘을 안 준다**: effect intent, 그리고 OSS 단독일 때의 lease.
- 없으면 안 생길 실패 모드를 하나 추가한다 —
  [langgraph#7417](https://github.com/langchain-ai/langgraph/issues/7417): 하트비트 타임아웃으로
  체크포인트에서 재발행되어, 원본이 아직 실행 중인데 도구가 다시 돈다. at-least-once는 문서화된
  동작이지만, **재발행 경로 자체가 체크포인터에서 나온다.**

이 코드베이스는 이미 밖에 리플레이 권한이 있다고 전제하고 있었다. `EventEnvelope.event_id`가
랜덤이 아니라 좌표에서 파생되는 이유가 그것이다 —
*"A run that crashes and resumes re-emits the events of rounds it had already finished; a
derived id lets an outbox drop the duplicates."* ([events.py](../../../src/nexora/contracts/events.py))

### 6. 결정 규칙 — 제품이 보장을 주장하는 축의 계약은 소유한다

4와 5를 빼면 `create_agent`는 `while` 하나짜리 루프를 위한 그래프 배선만 남긴다. 그런데 그
계산만으로는 [§LangGraph 쪽 편익](#langgraph-쪽-편익--이-문서가-빠뜨렸던-것)을 못 센다. 둘을
같이 놓으면 규칙은 하나로 정리된다.

**Nexora가 보장을 주장하는 축의 계약은 소유하고, 나머지는 채택한다.**

Nexora는 durable multi-agent runtime이다. 제품의 주장은 [ADR-002](adr-002-retry-safety-needs-order-determinism.md)에
있다 — 재시도가 안전하려면 배치 순서가 결정적이어야 한다. 그리고 그 축은 **LangGraph가 보장하지
않는다고 이 문서 §5가 이미 지적한 바로 그 축**이다: 중복 재개를 막는 lease도, 도구 실행 순서도
체크포인터가 주지 않고, at-least-once 재발행은 오히려 실패 모드를 추가한다
([langgraph#7417](https://github.com/langchain-ai/langgraph/issues/7417)).

보장하지 않는 축의 계약을 남에게 맡기면, 계약이 지켜지지 않을 때 할 수 있는 게 upstream을 기다리는
것뿐이다. 우리가 올린 [langgraph#8517](https://github.com/langchain-ai/langgraph/issues/8517)이
그 사례다 — 문서화된 `RunnableConfig.max_concurrency`가 한 실행 경로에서 조용히 무시된다.

거꾸로 모델·메시지·도구 스키마는 우리 제품의 주장이 아니다. 그래서 채택했고, 그 결정은 유효하다.
프로바이더 어댑터 120줄과 `ModelTurn` 66줄이 삭제됐다.

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

- **durable orchestrator를 두지 않기로 할 때.** 그러면 리플레이 권한이 비고, 체크포인터가 유일한
  durability가 된다. §5 전체가 이 전제에 걸려 있으므로 **가장 중요한 반증 조건이다.**
- ~~노드 중간 재개가 필요해질 때~~ — 스텝으로 얻는다(§5). 반증 조건이 아니다.
- ~~transcript를 만들지 않기로 할 때~~ — 오케스트레이터의 스텝 로그가 그 자리다.
- **LangGraph Cloud에 배포할 때.** 운영 표면 전체가 체크포인터 전제다.
- ~~`create_agent`의 model→tools 엣지가 배치 그룹을 받을 때~~ — **기다릴 필요가 없었다**
  (2026-08-04). 직접 조립하면 `Send`를 안 쓰고 노드가 배치를 받는다. 4는 `create_agent`에 한정된
  근거였고, 그 엔진은 삭제됐다.
- **`while`이 200줄 함수 하나로 안 읽히게 될 때.** 이제 남은 차이가 "지역변수 대 state 스키마"
  하나뿐이므로, 루프가 커져서 그 이점이 사라지면 그래프가 이긴다.

## LangGraph 쪽 편익 — 이 문서가 빠뜨렸던 것

> 검토에서 반대편이 제기했다. 이 문서는 §2~5에서 `create_agent`가 주는 것을 "루프·`ToolNode`·
> 체크포인터" 셋으로 좁혔는데, **그 셋 다 기능이고, 실제 편익은 기능이 아니라 계약이다.**
> 빠뜨린 채로 결정하면 공정하지 않으므로 여기 적는다.

LangChain/LangGraph를 채택하는 실질 가치는 그래프가 아니라 **이미 정해진 계약 위에서 조립한다**는
것이다.

| 계층 | 계약 | 우리 상태 |
|---|---|---|
| 프로바이더 | `BaseChatModel` | **채택** |
| 메시지·도구 호출 | `BaseMessage` / `ToolCall` | **채택** |
| 도구 | `BaseTool` | 부분 — 우리 `Tools`가 별도로 있고 `_as_langchain_tools`가 번역 |
| 실행 개입 | 미들웨어 훅 6개 | **미채택** — 우리 훅 6개 |
| 실행 설정 | `RunnableConfig` (동시성·콜백·태그·재귀 한계) | **미채택** — 인자로 흩어져 있음 |
| 중단·재개 | 체크포인터 / `interrupt` | **미채택** — `on_suspend` + 스냅샷 |

여기에 따라오는 것: 계약을 남이 정의·검증·문서화·버전관리하고, 진화도 남이 한다. LangSmith 트레이싱,
Studio, deepagents류 미들웨어 재사용, 그리고 **아는 사람이 많다**는 것까지.

**그래서 선택은 "계약 기반 vs 비계약 기반"이 아니다.** LangGraph의 실행 계약을 채택할 것인가,
Nexora가 자기 실행 계약을 **명시적으로** 만들 것인가다. 후자를 고르면 그 비용은 우리 것이다 —
`execute_batch`가 `Tools` Protocol에 없이 `getattr`로만 발견되던 것이 정확히 그 예였다. 동작은
있었지만 계약이 아니었고, 아무것도 그걸 타입 체크하지 않았다.

### 소유하기로 한 대가는 지불한다

- `BatchTools` Protocol을 정식화했다 (`contracts/types.py`) — `runtime_checkable`, 약속하는 규칙
  네 개(게이트-먼저·배치 단위 옵트인·결과 순서·suspend 중단)를 문서에 적었다.
- `execute_calls`가 `getattr` 대신 `isinstance(tools, BatchTools)`로 판별한다.
- 게이트-먼저 불변식에 계약 테스트를 붙였다 (`test_a_batch_arrives_gated`) — 어디에도 고정돼 있지
  않았고, 그래서 langgraph 엔진에서 깨진 것을 스위트가 못 봤다.

남은 것: ~~`is_concurrency_safe` 미구현~~(2026-08-04 `tools.Concurrent`로 구현),
도구 결과 태그 유니온이 `dict[str, Any]`인 것,
`RunnableConfig`에 해당하는 실행 설정 계약이 없는 것. **소유 비용의 잔액이고, ADR의 while 쪽
비용으로 센다.**

## 정정 기록 (2026-08-04) — 직접 조립은 허수아비가 아니었다

`create_agent` 대신 `StateGraph`를 직접 조립해 적합성 스위트를 돌렸다. **42/42 통과, 첫 시도에
40개.** 그리고 `create_agent` 버전은 삭제했다.

```
plain/loop.py                 300줄
langgraph/engine.py 직접 조립  328줄   ← 적응 코드 0
langgraph/engine.py 구버전     481줄   ← 삭제됨
```

| `create_agent`에서 필요했던 것 | 직접 조립 |
|---|---|
| `_translate` 36줄 — 메시지 델타에서 이벤트 역산 | 없음. `get_stream_writer()`로 노드가 직접 쓴다 |
| dual `stream_mode` + dedup 18줄 | 없음 |
| `_Outcome` 사이드 채널 29줄 | 없음. `state["stop_reason"]` |
| `_RoundEnd` 되추론 36줄 | 없음. `act` 노드가 곧 after-tools |
| `_as_langchain_tools` + `_CALL_ID` 39줄 | 없음. `ToolNode`를 안 쓴다 |
| `_Round` 공유 카운터 | 없음. `state["turn"]` |
| **`Send` 팬아웃 → 배치 해체** | **없음.** 노드가 라운드 전체를 받는다 |
| 갈림 3개 (`exclusive`·도구 `suspend`·`on_suspend`) | 없음. 노드가 라운드를 소유하므로 추론할 게 없다 |

**§4가 langgraph 전체에 대한 근거가 아니게 됐다.** 팬아웃은 그래프의 성질이 아니라 `create_agent`의
선택이었고, 그게 §4의 하중을 받던 부분이었다.

### 그래서 남은 차이는 하나뿐이다

```python
class State(TypedDict):          # while 엔진의 지역변수 5개가 이렇게 된다
    messages: Annotated[list[BaseMessage], add_messages]
    turn: int
    spent: Annotated[Counter[str], _merge]   # 리듀서 직접 작성
    ...
```

**채널에 있는 상태는 누군가 갱신을 잊는 상태다.** 이 세션의 langgraph 버그 셋(`turn` 상수 0,
batch-summary 라운드 스킵, c3 결과 유실)이 전부 그 계열이었다. 그게 `while`이 기본으로 남는
이유이고, 이제 그것이 **유일한** 이유다 — 나머지는 측정으로 지워졌다.

그리고 체크포인터는 여전히 안 붙였다. 그것만이 `while`이 못 하는 것이고, 아직 필요 없다(§5).

## 정정 기록 (2026-08-03)

엔진 선택을 다시 검토하면서 이 문서와 [실험 문서](../../experiments/2026-08-03-langgraph-conformance.md)의
주장을 재측정했다. **결정은 안 바뀌었다. 근거 네 개가 바뀌었다.**

| # | 원래 주장 | 어디 | 측정 결과 |
|---|---|---|---|
| 1 | `ToolNode`는 항상 `gather`이고 **옵트아웃이 없다** | 이 문서 §4 | **틀림.** `create_agent` 경로는 `max_concurrency=1`을 지킨다. 우리가 config를 안 넘기고 있었을 뿐 — 1줄 |
| 2 | 우회 = dict 스키마로 `ToolNode`를 안 만들기, **−60줄** | 이 문서 §4 | **전제가 틀림.** 문제는 `gather`가 아니라 팬아웃이 엣지에서 일어나는 것. 우회해도 배치는 안 온다 |
| 3 | `wrap_tool_call`의 **공유 락으로 되돌릴 수 있다** | [ADR-001](adr-001-plain-loop-over-graph-engine.md) 근거 1 | **틀림.** 락은 순서만 산다. `execute_batch` 미호출로 옵트인 병렬 상실. 조건부 락은 직렬화 자체가 안 됨 |
| 4 | 적합성 **12/12 동등** | [실험](../../experiments/2026-08-03-langgraph-conformance.md) | **과대.** 스위트가 안 물어본 갈림 4개가 있었다 (아래) |

### 갈림 4개 (스위트가 안 물어봤던 것)

| 갈림 | 상태 |
|---|---|
| 배치 실행 순서 | **닫힘** — `max_concurrency=1` + `test_a_batch_runs_one_call_at_a_time` |
| `exclusive` 미트림 | **열림, 우리 버그.** `select_for_execution`이 이벤트 발행에만 적용돼 `ToolNode`는 전부 실행한다 — 두 번째 도구가 **이벤트 기록 없이** 돈다 |
| 도구가 반환한 `suspend` 무시 | **열림, 우리 버그.** artifact로 흘러 모델에게 가고 run은 `completed`로 끝난다. 승인 대기가 사라진다 |
| `on_suspend` 미호출 | **열림** — 이미 docstring에 적혀 있던 것 |

앞의 것 하나는 config 1줄로 닫았고 테스트로 고정했다. 나머지 셋은 `engines/langgraph`가 참조물로
강등되므로 열어둔다. **적합성 스위트가 "동등"을 주장하려면 셋에 대한 테스트가 있어야 한다** —
없는 동등성을 주장하는 것이 이 문서가 네 번 틀린 방식이었다.

### 유지되는 근거

배치 단위 정책이 미들웨어 훅으로 표현 불가(§4)와 체크포인터 중복(§5). 그리고 §4의 이유가
"`ToolNode` 기본값"에서 "**팬아웃이 엣지에서 일어나 노드가 호출 하나씩 받음**"으로 정정됐다.

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
