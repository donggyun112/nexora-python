# LangGraph ToolNode `max_concurrency` 이슈 제출 자료

## 결론

유효한 버그 후보다. 다만 “병렬 툴 호출의 실행 순서가 보장되지 않는다”가 아니라 다음과 같이 제보해야 한다.

> Direct `ToolNode`의 async 실행 경로가 공식 `RunnableConfig.max_concurrency` 제한을 무시한다.

정식 등록 이슈: https://github.com/langchain-ai/langgraph/issues/8517

`max_concurrency`는 결정적 실행 순서를 보장하는 옵션이 아니라 최대 동시 실행 수를 제한하는 옵션이다. 따라서 일반적인 “순서 보장”을 주장하지 않는다. `max_concurrency=1`인데도 두 async 도구가 동시에 실행된다는 계약 위반을 주장한다.

## 다운로드 자료에서 확인한 제출 규칙

1. 이슈는 영어로만 제출한다.
2. 제목은 증상과 깨진 계약을 구체적으로 적는다.
3. 재현 코드 입력란에는 바깥쪽 Markdown code fence 없이 `repro.py` 내용을 붙인다. GitHub 양식이 Python 코드로 자동 렌더링한다.
4. 코드는 유지보수자가 그대로 복사해 즉시 실행할 수 있어야 한다.
5. 최신 stable에서 재현하고 `python -m langchain_core.sys_info` 결과를 첨부한다.
6. 공개 API의 실제 그래프 흐름에서 재현하며, 내부 함수만 직접 호출하는 재현은 피한다.
7. 중복 이슈와 인접 이슈를 검색하고, 기능요청과 버그를 구분한다.
8. 원인과 수정 방향은 제안할 수 있지만, 관찰된 public API 계약 위반이 본문 중심이어야 한다.
9. 구현 PR은 유지보수자가 접근법을 승인하고 이슈를 할당한 뒤 여는 것이 안전하다.

중요: `gh issue create --body-file ...`로 직접 생성하면 Issue Form의 `type: bug`와 `labels: ["bug"]`가 적용되지 않는다. 실제 Bug Report 웹 폼을 사용하거나, GraphQL `createIssue`의 `issueTemplate`에 파일명이 아닌 폼 표시 이름 `🐛 Bug Report`를 지정해야 한다. 생성 직후 실제 `issueType`과 `labels`를 다시 조회한다.

## 제출할 때 붙여 넣는 위치

- Title: `ToolNode async execution ignores RunnableConfig.max_concurrency for multiple tool calls`
- Checked other resources: 모두 확인 후 체크
- Related Issues / PRs: 해당 섹션의 영문
- Reproduction Steps / Example Code: `repro.py` 전체를 fence 없이 붙여 넣기
- Error Message and Stack Trace: 실제 실행 결과를 붙여 넣기
- Description: `issue_body_en.md`의 Description
- System Info: 제출 직전에 `python -m langchain_core.sys_info`를 다시 실행해 붙여 넣기

## 검증 범위

- `langgraph==1.2.10`, `langchain-core==1.5.3`, Python 3.12.10에서 재현
- sync `graph.invoke(..., config={"max_concurrency": 1})`: 최대 동시 실행 수 1
- async `graph.ainvoke(..., config={"max_concurrency": 1})`: 최대 동시 실행 수 2
- upstream 소스에서도 sync는 `get_executor_for_config(config)`, async는 제한 없는 `asyncio.gather(*coros)` 사용
- 별도 `create_agent` 대조군은 같은 설정을 지켰으므로 보고 범위는 direct multi-call `ToolNode` async 경로로 한정

## 제출 전 마지막 확인

- GitHub에서 `ToolNode max_concurrency`, `ToolNode asyncio.gather`, `ToolNode concurrency limit`를 다시 검색한다.
- 최신 stable 버전이 바뀌었으면 업그레이드 후 `repro.py`를 다시 실행한다.
- 실제 출력의 파일명과 줄 번호가 초안과 다르면 Stack Trace를 실제 출력으로 교체한다.
- “strict ordering”, “deterministic order”, “all parallel tool calls are broken”처럼 재현보다 넓은 표현은 사용하지 않는다.
