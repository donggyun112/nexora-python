# LangGraph ordered tool barriers 제안

등록 위치: https://github.com/langchain-ai/langchain/issues/33832#issuecomment-5163604212

## 요구 계약

```text
read_a, read_b, write_a, write_b, read_c
→ (read_a ∥ read_b) → write_a → write_b → read_c
```

- 연속된 read/shared 호출만 병렬 실행한다.
- write/exclusive 호출은 앞 호출이 끝날 때까지 기다리고 단독 실행한다.
- 뒤 호출은 write를 추월하지 않는다.
- write끼리는 모델이 발행한 순서를 유지한다.
- `max_concurrency`는 read/shared 구간 내부의 동시 실행 수 제한으로만 사용한다.

## 제출 판정

이것은 현재 LangGraph가 문서로 보장하는 계약의 위반이 아니라 새로운 실행 정책이므로 Bug가 아닌 Feature 제안이다.

LangGraph 저장소의 Issue Form은 비관리자의 기능요청과 설계 논의를 LangChain Forum으로 보내며 blank issue도 비활성화한다. 또한 `langchain-ai/langchain#33832`가 전체 순차 실행과 state propagation을 이미 요청하고 있다. 따라서 새 Bug 이슈를 우회 생성하지 않고 #33832에 차별화된 shared/exclusive barrier 제안으로 댓글을 추가한다.

## 기존 요청과 차이

- #33832: 모든 호출을 순차 실행하고 앞 도구의 `Command` state update를 다음 도구에 전파
- 이 제안: 외부 부작용의 순서 장벽을 보장하면서 연속 read는 병렬성 유지
- 이 제안만으로 같은 ToolNode 안의 LangGraph state snapshot 전파를 약속하지 않음
