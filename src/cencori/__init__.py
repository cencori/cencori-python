"""
Cencori Python SDK

Official Python SDK for Cencori - AI Infrastructure for Production.

One SDK for AI Gateway, Compute, Workflow, and Storage.
Every operation is secured, logged, and tracked.

Example:
    >>> from cencori import Cencori
    >>> cencori = Cencori(api_key="csk_...")
    >>>
    >>> # Chat completion
    >>> response = cencori.ai.chat(
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
    >>> print(response.content)
    >>>
    >>> # Streaming
    >>> for chunk in cencori.ai.chat_stream(messages=[...]):
    ...     print(chunk.delta, end="")
    >>>
    >>> # Embeddings
    >>> embeddings = cencori.ai.embeddings(input="Hello world")
    >>> print(len(embeddings.embeddings[0]))
    >>>
    >>> # Generate structured output
    >>> result = cencori.ai.generate_object(
    ...     model="gpt-4o",
    ...     prompt="Generate a user profile",
    ...     schema={"type": "object", "properties": {"name": {"type": "string"}}},
    ... )
    >>> print(result.object)
    >>>
    >>> # Generate images
    >>> images = cencori.ai.generate_image(
    ...     prompt="A futuristic city at sunset",
    ...     model="dall-e-3",
    ... )
    >>>
    >>> # RAG
    >>> rag_response = cencori.ai.rag(
    ...     model="gpt-4o",
    ...     messages=[{"role": "user", "content": "What are our policies?"}],
    ...     namespace="company-docs",
    ... )
    >>>
    >>> # Agents
    >>> agent = cencori.agents.create(name="my-agent", config={"model": "gpt-4o"})
    >>>
    >>> # Memory
    >>> memory = cencori.memory.store(
    ...     namespace="conversations",
    ...     content="User asked about pricing",
    ... )
    >>>
    >>> # Telemetry
    >>> cencori.telemetry.report_web_request(
    ...     host="myapp.com",
    ...     method="GET",
    ...     path="/api/chat",
    ...     status_code=200,
    ... )
"""

from .client import Cencori
from .errors import (
    AuthenticationError,
    CencoriError,
    InsufficientCreditsError,
    ProviderError,
    RateLimitError,
    SafetyError,
)
from .types import (
    # Chat types
    Agent,
    AgentConfig,
    AgentKey,
    AgentListItem,
    APIKey,
    Breakdown,
    ChatRequest,
    ChatResponse,
    CodeInterpreterTool,
    CompletionRequest,
    CostMetrics,
    CreateAgentKeyParams,
    CreateAgentParams,
    CreateAPIKeyParams,
    CreateNamespaceOptions,
    CreateProjectParams,
    DailyStat,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    FileSearchTool,
    GeneratedImage,
    GenerateObjectRequest,
    GenerateObjectResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    KeyUsageStats,
    LatencyMetrics,
    Memory,
    MemoryNamespace,
    Message,
    MetricsResponse,
    Project,
    RagRequest,
    RagResponse,
    RagSource,
    RagStreamChunk,
    RequestMetrics,
    ResponseInputItem,
    ResponsesOutputItem,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesTool,
    SearchMemoryOptions,
    SearchResult,
    Stats,
    StoreMemoryOptions,
    StreamChunk,
    TokenMetrics,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    UpdateAgentParams,
    UrlCitation,
    Usage,
    WebSearchTool,
    WebTelemetryPayload,
)

__version__ = "1.2.1"
__all__ = [
    # Client
    "Cencori",
    # Errors
    "CencoriError",
    "AuthenticationError",
    "RateLimitError",
    "SafetyError",
    "InsufficientCreditsError",
    "ProviderError",
    # Chat types
    "Message",
    "ChatRequest",
    "ChatResponse",
    "StreamChunk",
    "Usage",
    "ToolDefinition",
    "ToolCall",
    "ToolChoice",
    # Completion types
    "CompletionRequest",
    # Embedding types
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingUsage",
    # Generate Object types
    "GenerateObjectRequest",
    "GenerateObjectResponse",
    # Image types
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "GeneratedImage",
    # RAG types
    "RagRequest",
    "RagResponse",
    "RagStreamChunk",
    "RagSource",
    # Responses API types
    "ResponsesRequest",
    "ResponsesResponse",
    "ResponsesOutputItem",
    "ResponseInputItem",
    "ResponsesTool",
    "WebSearchTool",
    "FileSearchTool",
    "CodeInterpreterTool",
    "UrlCitation",
    # Agent types
    "Agent",
    "AgentConfig",
    "AgentListItem",
    "CreateAgentParams",
    "UpdateAgentParams",
    "AgentKey",
    "CreateAgentKeyParams",
    # Memory types
    "MemoryNamespace",
    "CreateNamespaceOptions",
    "Memory",
    "StoreMemoryOptions",
    "SearchMemoryOptions",
    "SearchResult",
    # Telemetry types
    "WebTelemetryPayload",
    # Project types
    "Project",
    "CreateProjectParams",
    "Stats",
    # API Key types
    "APIKey",
    "CreateAPIKeyParams",
    "DailyStat",
    "KeyUsageStats",
    # Metrics types
    "MetricsResponse",
    "RequestMetrics",
    "CostMetrics",
    "TokenMetrics",
    "LatencyMetrics",
    "Breakdown",
]
