# Cencori Python SDK

Official Python SDK for Cencori - AI Infrastructure for Production.

One SDK for AI Gateway, Agents, Memory, Compute, Workflow, and Storage.
Every operation is secured, logged, and tracked.

## Installation

```bash
pip install cencori
```

## Quick Start

```python
from cencori import Cencori

cencori = Cencori(api_key="your-api-key")

# Chat
response = cencori.ai.chat(
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content)

# Embeddings
embedding = cencori.ai.embeddings(
    input="Hello world",
    model="text-embedding-3-small"
)
print(len(embedding.embeddings[0]))

# Generate structured output
result = cencori.ai.generate_object(
    model="gpt-4o",
    prompt="Generate a user profile",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
        },
    },
)
print(result.object)  # {"name": "John", "age": 30}

# Generate images
images = cencori.ai.generate_image(
    prompt="A futuristic city at sunset",
    model="dall-e-3",
)
print(images.images[0].url)
```

## Async Support

All methods have async counterparts:

```python
import asyncio
from cencori import Cencori

async def main():
    cencori = Cencori()

    # Async chat
    response = await cencori.ai.async_chat(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.content)

    # Async generate object
    result = await cencori.ai.async_generate_object(
        model="gpt-4o",
        prompt="Generate a JSON object",
        schema={"type": "object", "properties": {"key": {"type": "string"}}},
    )
    print(result.object)

asyncio.run(main())
```

## Streaming

```python
for chunk in cencori.ai.chat_stream(
    messages=[{"role": "user", "content": "Tell me a story"}],
    model="gpt-4o"
):
    print(chunk.delta, end="", flush=True)
```

## RAG (Retrieval-Augmented Generation)

```python
# Chat with automatic memory context
response = cencori.ai.rag(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What are our company policies?"}],
    namespace="company-docs",
)
print(response.message["content"])
print(response.sources)  # Retrieved context

# Streaming RAG
for chunk in cencori.ai.rag_stream(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me about policies"}],
    namespace="company-docs",
):
    if chunk.type == "content":
        print(chunk.delta, end="", flush=True)
```

## Agents

```python
# Create an agent
agent = cencori.agents.create(
    name="support-agent",
    description="Customer support assistant",
    config={
        "model": "gpt-4o",
        "system_prompt": "You are a helpful support agent",
        "temperature": 0.7,
    },
)

# List agents
agents = cencori.agents.list()

# Get agent
agent = cencori.agents.get("ag_123")

# Update agent
from cencori import UpdateAgentParams
agent = cencori.agents.update_config(
    "ag_123",
    UpdateAgentParams(name="updated-name"),
)

# Create agent key
key = cencori.agents.create_key(
    "ag_123",
    name="prod-key",
    environment="production",
)

# Delete agent
cencori.agents.delete("ag_123")
```

## Memory (Vector Storage)

```python
# Create a namespace
ns = cencori.memory.create_namespace(
    name="conversations",
    description="User conversation history",
)

# Store a memory
memory = cencori.memory.store(
    namespace="conversations",
    content="User asked about pricing plans",
    metadata={"userId": "user_123"},
)

# Search memories
results = cencori.memory.search(
    namespace="conversations",
    query="what did we discuss about pricing?",
    limit=5,
)
for mem in results.results:
    print(mem.content)

# List namespaces
namespaces = cencori.memory.list_namespaces()

# Delete memory
cencori.memory.delete("mem_123")

# Delete by filter
cencori.memory.delete_by_filter(
    namespace="conversations",
    filter={"userId": "user_123"},
)
```

## Telemetry

```python
# Report web requests to the Cencori dashboard
cencori.telemetry.report_web_request(
    host="myapp.vercel.app",
    method="GET",
    path="/api/chat",
    status_code=200,
    user_agent="Mozilla/5.0",
    latency_ms=150,
)
```

## Responses API (OpenAI-compatible)

```python
# Use built-in tools like web search
response = cencori.ai.responses(
    model="gpt-4o",
    input="What is the weather in San Francisco?",
    tools=[{"type": "web_search_preview"}],
)
print(response.output[0].content[0].text)

# Streaming responses
for event in cencori.ai.responses_stream(
    model="gpt-4o",
    input="Tell me about AI",
):
    if event["type"] == "response.output_text.delta":
        print(event["data"]["delta"], end="", flush=True)
```

## Project Management

```python
from cencori.types import CreateProjectParams

# List projects
projects = cencori.projects.list(org_slug="my-org")

# Create project
project = cencori.projects.create(
    org_slug="my-org",
    params=CreateProjectParams(name="New Project")
)
```

## API Key Management

```python
from cencori.types import CreateAPIKeyParams

# Create API key
key = cencori.api_keys.create(
    project_id="proj_123",
    params=CreateAPIKeyParams(name="Dev Key", environment="dev")
)
print(f"Secret Key: {key.key}") # Only shown once!

# Get key stats
stats = cencori.api_keys.get_stats(project_id="proj_123", key_id=key.id)
```

## Metrics & Analytics

```python
# Get usage metrics for last 24 hours
metrics = cencori.metrics.get(period="24h")

print(f"Total Requests: {metrics.requests.total}")
print(f"Total Cost: ${metrics.cost.total_usd}")
```

## Error Handling

```python
from cencori import (
    Cencori,
    AuthenticationError,
    RateLimitError,
    SafetyError,
    CencoriError,
)

try:
    response = cencori.ai.chat(messages=[{"role": "user", "content": "Hello"}])
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Too many requests")
except SafetyError as e:
    print(f"Content blocked: {e.reasons}")
except CencoriError as e:
    print(f"Error: {e.message}")
```

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo` |
| Anthropic | `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku` |
| Google | `gemini-2.5-flash`, `gemini-2.0-flash` |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, workflow, and guidelines.

## License

MIT © FohnAI
