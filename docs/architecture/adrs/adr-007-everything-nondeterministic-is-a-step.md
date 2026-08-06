# ADR-007: 비결정적인 것은 전부 step이다 — 복구 권한은 ledger 하나

**상태:** Accepted (2026-08-05)

**관계:** ADR-005를 유지하고 보강한다. ADR-005가 열어둔 뒤집힘 조건 2·3·5를 검토한 결과,
세 축 모두 graph checkpointing이 아니라 기존 effect ledger의 확장으로 닫힌다.

## 맥락

ADR-005는 LangGraph 철회를 tool effect 범위에서 결정하면서 세 축을 재검토 대상으로 남겼다:
durable model invocation(조건 2), nested subagent와 병렬 branch/join(조건 3), transcript
checkpoint 쓰기 증폭(조건 5). 이 문서는 그 세 축에 대한 논쟁의 결론을 기록한다. 논쟁의 결과
LangGraph 재도입 논거는 하나씩 소거되었고, 남은 것은 기술 논거가 아니라 Platform 운영 구매
결정뿐이다.

## 논증 1: 모델 호출도 effect step이다

토큰 중복은 어떤 프레임워크도 제거하지 못한다. LangGraph checkpointer도 모델 호출 **도중**
크래시하면 재호출한다 — checkpoint는 superstep 경계에서만 커밋되므로 스트리밍 중간의 부분
응답은 없던 일이 된다. checkpointer가 닫는 창은 "응답 완료 → 커밋 완료" 사이의 쓰기 경합뿐이다.

model invocation을 `StepLog` step으로 만들면 **정확히 같은 크기의 창**이 남는다: 응답 완료 →
`finish` 커밋. 따라서 이 축에서 graph checkpointer의 한계효용은 0이고, ADR-004가 예고한
"model invocation을 Effect 계약으로"가 맞는 경로다. 모델 호출은 비멱등 외부 effect의 한
종류일 뿐이며, 그 복구는 이미 소유한 ledger의 문제다.

## 논증 2: subagent 호출도 effect step이다 — 그래프로 표현하지 않는다

subagent 호출은 비결정적 호출이고, tool call과 같은 개념이다. supervisor 관계를 정적 그래프로
표현하는 것은 채택 사유가 아니라 반대 사유다.

존재 증명 둘:

- **Temporal child workflow.** 부모 히스토리에 자식 호출이 step으로 기록되고, 자식은 자기
  히스토리를 가진 독립 실행이다. 그래프 없이 검증된 모델.
- **레퍼런스 구현 자체.** Claude Code의 Task tool — subagent는 tool call이고 결과는
  ToolMessage다.

effect 모델에 대응시키면:

| 요구 | 해결 |
|---|---|
| 부모가 어느 자식을 기다리는지 | transcript 꼬리의 미응답 delegate call (`_unanswered_tool_calls` 원리 그대로) |
| 선택적 재시도 | call-id별 ledger 상태 — done인 자식 재사용, 나머지만 재실행. **멱등성에서 공짜** |
| 자식 interrupt의 부모 전파 | suspend 타입 tool result → `RoundSuspended`. **이미 구현됨** |
| join | 라운드 배리어. 배리어 아닌 합류는 input ledger의 background `tool_result` |
| 취소 전파 | `aborted` + 자식 run의 lease 만료 |

LangGraph 자신도 동적 팬아웃에서 `Send` API로 수렴했다 — 정적 그래프가 이 문제의 표현이
아니라는 자백이다.

**범위: delegation은 1단만.** 손자 에이전트(2단 이상 중첩)는 레퍼런스 구현도 업계도 패턴이
아니다. 1단이면 resume 라우팅은 suspension 레코드에 자식 run_id 필드 하나로 끝난다.

공짜가 아닌 것 — 전부 평범한 코드이며 그래프의 근거가 아니다:

- 배리어 아닌 join 정책 (first-wins + 나머지 취소)
- 자식 이벤트의 상향 스트리밍 채널

## 논증 3: transcript는 append-only log다 — 쓰기 증폭은 스냅샷형의 문제

대화 히스토리는 제품상 어차피 저장한다. 선형 루프에서 program counter는 transcript 꼬리에서
유도되므로([orchestrator.py](../../../src/nexora/orchestrator.py)의 `recover_pending`이 이미
그 원리로 동작한다), **그 저장이 곧 state 저장이다.**

쓰기 증폭은 매 라운드 전체 히스토리를 복사하는 스냅샷형 영속화의 문제다
(`record_pending`이 transcript 복사를 거부하는 이유). 메시지당 1회 append는 전체 O(n)이고
증폭이 없다. **즉 ADR-005 조건 5는 자체 store의 장벽이 아니라 checkpointer 쪽 숙제였다.**

"그냥 채팅 히스토리 테이블"과 다른 계약은 둘뿐이다:

1. **model-facing 손실 없는 직렬화.** `tool_calls` 원형(id·args), ToolMessage의
   멀티모달/artifact 왕복, admitted-input 순서. 실측된 함정이다 — 적합성 실험의 "ToolMessage가
   문자열이라 왕복이 안 됨"(`content_and_artifact`로 회복), 그리고 UI 데모의 `capture()`가
   채팅 로그가 아니라 이벤트에서 AIMessage를 재조립해야 했던 것. 화면용 스키마와 겸용할 수 없다.
2. **ledger보다 앞서지 않는 append 순서.** 뒤처짐은 안전하다 — `recover_pending`이 ledger의
   done 결과로 따라잡는다. 금지는 앞서는 것뿐: effect 커밋 후 append.

## 남은 유일한 LangGraph 논거: Platform 운영 아웃소싱

| Platform 제공 | Nexora 현재 |
|---|---|
| run queue, 백그라운드 실행, 워커 스케일링 | 없음 — 범용 인프라(큐+워커)로 대체 가능 |
| threads/runs/HITL HTTP 표면 | `ui/` 콘솔이 초기 형태 |
| run lease | 있음 — `StepLog` lease + fencing |
| double-texting 처리 | 있음 — cancel-and-switch + input ledger |
| managed Postgres | `steps_postgres.py` 있음, 운영만 남음 |
| Studio / LangSmith | `langchain-core` 유지로 모델 계층 트레이싱은 langgraph 없이 가능 |

Platform이 파는 것은 관리형 에이전트 백엔드이고, 가격은 런타임이 LangGraph가 되는 것이다.
에이전트 특화 항목(lease, double-texting, HITL)은 이미 소유했고, 없는 항목은 전부 에이전트
특화가 아닌 범용 웹 인프라다. 따라서 이 구매가 정당해지는 유일한 시나리오는 "출시 속도가
실행 계약 소유권보다 중요하다"는 사업 판단이며, 그것은 기술 ADR이 아니라 별도 결정이다.

## 결정

- model invocation을 durable effect step으로 만든다.
- subagent delegation을 durable effect step으로 만든다. 1단만 설계한다.
- append-only transcript store를 만든다. 계약: model-facing 손실 없는 직렬화,
  effect-커밋-후-append 순서 규율.
- 구현 순서: transcript store → model-as-effect-step → `recover()`의 `history` 인자 제거.
  UI 데모의 `capture()` 재조립은 이 시점에 자연 소멸한다.
- ADR-005의 뒤집힘 조건 중 2·3·5는 이 문서로 닫는다. 남는 재검토 방아쇠는 Platform 구매라는
  사업 결정뿐이다.

## 반증 조건

- 위 구현에서 step key 결정성 또는 admitted-input 순서 재생이 깨지는 실측 사례가 나올 때.
- 1단 delegation으로 부족한 실제 제품 요구가 생길 때.
- 운영 표면 전체를 구매하기로 하는 사업 결정이 내려질 때 — 그 경우에도 논증 1~3은 유효하므로,
  구매 결정이 기술 결론을 소급 변경하지 않는다.
