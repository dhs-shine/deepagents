# 🚀🧠 Deep Agents (딥 에이전트)

에이전트는 점점 더 긴 호흡의 작업을 처리할 수 있게 되었으며, [에이전트의 작업 길이는 7개월마다 두 배로 늘어나고 있습니다](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)! 하지만 긴 호흡의 작업은 종종 수십 번의 도구 호출(tool calls)을 필요로 하며, 이는 비용과 신뢰성 문제를 야기합니다. [Claude Code](https://code.claude.com/docs)나 [Manus](https://www.youtube.com/watch?v=6_BcCthVvb8)와 같은 인기 있는 에이전트들은 이러한 문제를 해결하기 위해 **계획 수립(planning)** (작업 실행 전), **컴퓨터 액세스(computer access)** (셸 및 파일 시스템 접근 권한 부여), **서브 에이전트 위임(sub-agent delegation)** (격리된 작업 실행)과 같은 공통된 원칙을 사용합니다. `deepagents`는 이러한 도구들을 구현한 간단한 에이전트 하네스(harness)이면서도, 오픈 소스이며 여러분만의 커스텀 도구와 지침으로 쉽게 확장할 수 있습니다.

<img src="deepagents_banner.png" alt="deep agent" width="100%"/>

## 📚 리소스

- **[문서 (Documentation)](https://docs.langchain.com/oss/python/deepagents/overview)** - 전체 개요 및 API 참조
- **[퀵스타트 저장소 (Quickstarts Repo)](https://github.com/langchain-ai/deepagents-quickstarts)** - 예제 및 사용 사례

## 🚀 퀵스타트 (Quickstart)

`deepagents`에 커스텀 도구를 제공할 수 있습니다. 아래에서는 선택적으로 웹 검색을 위한 `tavily` 도구를 제공해 보겠습니다. 이 도구는 `deepagents`의 내장 도구(아래 참조)에 추가됩니다.

```bash
pip install deepagents tavily-python
```

환경 변수에 `TAVILY_API_KEY`를 설정하세요 ([여기서 발급 가능](https://www.tavily.com/)):

```python
import os
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(query: str, max_results: int = 5):
    """웹 검색 실행"""
    return tavily_client.search(query, max_results=max_results)

agent = create_deep_agent(
    tools=[internet_search],
    system_prompt="조사를 수행하고 잘 다듬어진 보고서를 작성하세요.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "LangGraph가 무엇인가요?"}]})
```

`create_deep_agent`로 생성된 에이전트는 컴파일된 [LangGraph StateGraph](https://docs.langchain.com/oss/python/langgraph/overview)이므로, 다른 LangGraph 에이전트와 마찬가지로 스트리밍, Human-in-the-loop, 메모리 또는 Studio와 함께 사용할 수 있습니다. 더 많은 예제는 [퀵스타트 저장소](https://github.com/langchain-ai/deepagents-quickstarts)를 참조하세요.

## Deep Agents 커스터마이징

`create_deep_agent`에 전달할 수 있는 몇 가지 매개변수가 있습니다.

### `model`

기본적으로 `deepagents`는 `"claude-sonnet-4-5-20250929"`를 사용합니다. [LangChain 모델 객체](https://python.langchain.com/docs/integrations/chat/)를 전달하여 이를 변경할 수 있습니다.

```python
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

model = init_chat_model("openai:gpt-4o")
agent = create_deep_agent(
    model=model,
)
```

### `system_prompt`

`create_deep_agent()`에 `system_prompt` 매개변수를 제공할 수 있습니다. 이 커스텀 프롬프트는 미들웨어에 의해 자동으로 주입되는 기본 지침 **뒤에 추가**됩니다.

커스텀 시스템 프롬프트를 작성할 때 해야 할 일:
- ✅ 도메인별 워크플로우 정의 (예: 연구 방법론, 데이터 분석 단계)
- ✅ 사용 사례에 대한 구체적인 예시 제공
- ✅ 전문적인 가이드 추가 (예: "유사한 연구 작업은 하나의 할 일(TODO)로 일괄 처리")
- ✅ 중단 기준 및 리소스 제한 정의
- ✅ 워크플로우에서 도구들이 어떻게 함께 작동하는지 설명

**하지 말아야 할 일:**
- ❌ 표준 도구가 하는 일 다시 설명하기 (이미 미들웨어에서 다룸)
- ❌ 도구 사용법에 대한 미들웨어 지침 중복 작성
- ❌ 기본 지침과 모순되게 작성 (반대하지 말고 함께 작동하도록 작성)

```python
from deepagents import create_deep_agent
research_instructions = """여러분의 커스텀 시스템 프롬프트"""
agent = create_deep_agent(
    system_prompt=research_instructions,
)
```

더 많은 예제는 [퀵스타트 저장소](https://github.com/langchain-ai/deepagents-quickstarts)를 참조하세요.

### `tools`

에이전트에 커스텀 도구를 제공하세요 ([내장 도구](#built-in-tools) 외에 추가):

```python
from deepagents import create_deep_agent

def internet_search(query: str) -> str:
    """웹 검색 실행"""
    return tavily_client.search(query)

agent = create_deep_agent(tools=[internet_search])
```

[langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters)를 통해 MCP 도구를 연결할 수도 있습니다:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent

async def main():
    mcp_client = MultiServerMCPClient(...)
    mcp_tools = await mcp_client.get_tools()
    agent = create_deep_agent(tools=mcp_tools)

    async for chunk in agent.astream({"messages": [{"role": "user", "content": "..."}]}):
        chunk["messages"][-1].pretty_print()
```

### `middleware`

Deep agents는 확장성을 위해 [미들웨어](https://docs.langchain.com/oss/python/langchain/middleware)를 사용합니다 (기본값은 [내장 도구](#built-in-tools) 참조). 커스텀 미들웨어를 추가하여 도구를 주입하거나, 프롬프트를 수정하거나, 에이전트 수명 주기에 연결(hook)할 수 있습니다.

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware

@tool
def get_weather(city: str) -> str:
    """도시의 날씨를 가져옵니다."""
    return f"{city}의 날씨는 맑음입니다."

class WeatherMiddleware(AgentMiddleware):
    tools = [get_weather]

agent = create_deep_agent(middleware=[WeatherMiddleware()])
```

### `subagents`

메인 에이전트는 `task` 도구를 통해 서브 에이전트에게 작업을 위임할 수 있습니다 ([내장 도구](#built-in-tools) 참조). 컨텍스트 격리 및 커스텀 지침을 위해 커스텀 서브 에이전트를 제공할 수 있습니다.

```python
from deepagents import create_deep_agent

research_subagent = {
    "name": "research-agent",
    "description": "심층적인 질문을 조사하는 데 사용됨",
    "prompt": "당신은 전문 연구원입니다",
    "tools": [internet_search],
    "model": "openai:gpt-4o",  # 선택 사항, 기본값은 메인 에이전트 모델
}

agent = create_deep_agent(subagents=[research_subagent])
```

복잡한 경우에는 미리 빌드된 LangGraph 그래프를 전달하세요:

```python
from deepagents import CompiledSubAgent, create_deep_agent

custom_graph = create_agent(model=..., tools=..., prompt=...)

agent = create_deep_agent(
    subagents=[CompiledSubAgent(
        name="data-analyzer",
        description="데이터 분석을 위한 전문 에이전트",
        runnable=custom_graph
    )]
)
```

자세한 내용은 [서브 에이전트 문서](https://docs.langchain.com/oss/python/deepagents/subagents)를 참조하세요.

### `interrupt_on`

일부 도구는 민감하여 실행 전 사람의 승인이 필요할 수 있습니다. Deepagents는 LangGraph의 인터럽트 기능을 통해 Human-in-the-loop 워크플로우를 지원합니다. 체크포인터를 사용하여 승인이 필요한 도구를 구성할 수 있습니다.

이러한 도구 설정은 미리 빌드된 [HITL 미들웨어](https://docs.langchain.com/oss/python/langchain/middleware#human-in-the-loop)로 전달되어, 에이전트가 실행을 일시 중지하고 구성된 도구를 실행하기 전에 사용자의 피드백을 기다리게 합니다.

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent

@tool
def get_weather(city: str) -> str:
    """도시의 날씨를 가져옵니다."""
    return f"{city}의 날씨는 맑음입니다."

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[get_weather],
    interrupt_on={
        "get_weather": {
            "allowed_decisions": ["approve", "edit", "reject"]
        },
    }
)
```

자세한 내용은 [Human-in-the-loop 문서](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)를 참조하세요.

### `backend`

Deep agents는 플러그형 백엔드를 사용하여 파일 시스템 작업 방식을 제어합니다. 기본적으로 파일은 에이전트의 임시(ephemeral) 상태에 저장됩니다. 로컬 디스크 액세스, 영구적인 대화 간 저장소 또는 하이브리드 라우팅을 위해 다른 백엔드를 구성할 수 있습니다.

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="/path/to/project"),
)
```

사용 가능한 백엔드:
- **StateBackend** (기본값): 에이전트 상태에 저장되는 임시 파일
- **FilesystemBackend**: 루트 디렉터리 하위의 실제 디스크 작업
- **StoreBackend**: LangGraph Store를 사용한 영구 저장소
- **CompositeBackend**: 경로별로 다른 백엔드로 라우팅

자세한 내용은 [백엔드 문서](https://docs.langchain.com/oss/python/deepagents/backends)를 참조하세요.

### 장기 기억 (Long-term Memory)

Deep agents는 특정 경로를 영구 저장소로 라우팅하는 `CompositeBackend`를 사용하여 대화 간에 지속되는 메모리를 유지할 수 있습니다.

이를 통해 작업 파일은 임시로 유지하면서 중요한 데이터(사용자 선호도 또는 지식 베이스 등)는 스레드 간에 지속되는 하이브리드 메모리가 가능해집니다.

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(store=InMemoryStore())},
    ),
)
```

`/memories/` 하위의 파일들은 모든 대화에서 지속되지만, 다른 경로들은 임시로 유지됩니다. 사용 사례:
- 세션 간 사용자 선호도 보존
- 여러 대화에서 지식 베이스 구축
- 피드백을 기반으로 한 자체 개선 지침
- 세션 간 연구 진행 상황 유지

자세한 내용은 [장기 기억 문서](https://docs.langchain.com/oss/python/deepagents/long-term-memory)를 참조하세요.

## 내장 도구 (Built-in Tools)

<img src="deepagents_tools.png" alt="deep agent" width="600"/>

`create_deep_agent`로 생성된 모든 딥 에이전트는 표준 도구 세트와 함께 제공됩니다:

| 도구 이름 | 설명 | 제공자 |
|-----------|-------------|-------------|
| `write_todos` | 복잡한 워크플로우 진행 상황을 추적하기 위한 구조화된 작업 목록 생성 및 관리 | TodoListMiddleware |
| `read_todos` | 현재 할 일 목록 상태 읽기 | TodoListMiddleware |
| `ls` | 디렉터리의 모든 파일 나열 (절대 경로 필요) | FilesystemMiddleware |
| `read_file` | 선택적 페이지네이션(offset/limit 매개변수)으로 파일 내용 읽기 | FilesystemMiddleware |
| `write_file` | 새 파일 생성 또는 기존 파일 완전히 덮어쓰기 | FilesystemMiddleware |
| `edit_file` | 파일 내 정확한 문자열 교체 수행 | FilesystemMiddleware |
| `glob` | 패턴과 일치하는 파일 찾기 (예: `**/*.py`) | FilesystemMiddleware |
| `grep` | 파일 내 텍스트 패턴 검색 | FilesystemMiddleware |
| `execute`* | 샌드박스 환경에서 셸 명령 실행 | FilesystemMiddleware |
| `task` | 격리된 컨텍스트 윈도우를 가진 전문 서브 에이전트에게 작업 위임 | SubAgentMiddleware |

`execute` 도구는 백엔드가 `SandboxBackendProtocol`을 구현한 경우에만 사용할 수 있습니다. 기본적으로 명령 실행을 지원하지 않는 인메모리 상태 백엔드를 사용합니다. 보시는 바와 같이, 이러한 도구들(및 다른 기능들)은 기본 미들웨어에 의해 제공됩니다.

내장 도구 및 기능에 대한 자세한 내용은 [에이전트 하네스 문서](https://docs.langchain.com/oss/python/deepagents/harness)를 참조하세요.

## 내장 미들웨어 (Built-in Middleware)

`deepagents`는 내부적으로 미들웨어를 사용합니다. 사용되는 미들웨어 목록은 다음과 같습니다.

| 미들웨어 | 목적 |
|------------|---------|
| **TodoListMiddleware** | 작업 계획 및 진행 상황 추적 |
| **FilesystemMiddleware** | 파일 작업 및 컨텍스트 오프로딩 (대용량 결과 자동 저장) |
| **SubAgentMiddleware** | 격리된 서브 에이전트에 작업 위임 |
| **SummarizationMiddleware** | 컨텍스트가 170k 토큰을 초과할 때 자동 요약 |
| **AnthropicPromptCachingMiddleware** | 시스템 프롬프트를 캐시하여 비용 절감 (Anthropic 전용) |
| **PatchToolCallsMiddleware** | 중단(interruption)으로 인한 댕글링(dangling) 도구 호출 수정 |
| **HumanInTheLoopMiddleware** | 사람의 승인을 위해 실행 일시 중지 (`interrupt_on` 설정 필요) |

## 내장 프롬프트 (Built-in prompts)

미들웨어는 표준 도구에 대한 지침을 자동으로 추가합니다. 여러분의 커스텀 지침은 이러한 기본값을 **중복하지 않고 보완**해야 합니다.

#### [TodoListMiddleware](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/middleware/todo.py)에서 제공
- `write_todos`와 `read_todos`를 언제 사용해야 하는지 설명
- 작업을 완료로 표시하는 방법에 대한 안내
- 할 일 목록 관리 모범 사례
- 할 일 목록을 사용하지 말아야 할 때 (간단한 작업)

#### [FilesystemMiddleware](libs/deepagents/deepagents/middleware/filesystem.py)에서 제공
- 모든 파일 시스템 도구 나열 (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`*)
- 파일 경로가 `/`로 시작해야 함을 설명
- 각 도구의 목적과 매개변수 설명
- 대용량 도구 결과에 대한 컨텍스트 오프로딩 관련 참고 사항

#### [SubAgentMiddleware](libs/deepagents/deepagents/middleware/subagents.py)에서 제공
- 서브 에이전트에 위임하기 위한 `task()` 도구 설명
- 서브 에이전트를 사용해야 할 때와 사용하지 말아야 할 때
- 병렬 실행에 대한 안내
- 서브 에이전트 수명 주기 (생성 → 실행 → 반환 → 조정)
