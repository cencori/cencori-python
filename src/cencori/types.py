"""Type definitions for Cencori SDK."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

# =============================================================================
# Chat Types
# =============================================================================


@dataclass
class ToolDefinition:
    """A tool/function definition for AI models."""

    type: Literal["function"] = "function"
    function: Optional[Dict[str, Any]] = None  # { name, description?, parameters? }


@dataclass
class ToolCall:
    """A tool call made by the model."""

    id: str
    type: Literal["function"] = "function"
    function: Optional[Dict[str, Any]] = None  # { name, arguments }


ToolChoice = Union[
    Literal["auto", "none", "required"],
    Dict[Literal["type", "function"], Any],
]


@dataclass
class Message:
    """A chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


@dataclass
class ChatRequest:
    """Parameters for chat completion."""

    model: str = "gemini-2.5-flash"
    messages: List[Dict[str, Any]] = field(default_factory=list)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[ToolChoice] = None
    prompt: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None


@dataclass
class Usage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    """Response from chat completion."""

    id: str = ""
    model: str = ""
    content: str = ""
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None
    provider: Optional[str] = None
    cost_usd: Optional[float] = None


@dataclass
class StreamChunk:
    """A chunk from streaming response."""

    delta: str = ""
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    error: Optional[str] = None


# =============================================================================
# Completion Types
# =============================================================================


@dataclass
class CompletionRequest:
    """Parameters for text completion."""

    prompt: str
    model: str = "gemini-2.5-flash"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# =============================================================================
# Embedding Types
# =============================================================================


@dataclass
class EmbeddingRequest:
    """Parameters for embedding generation."""

    input: Union[str, List[str]]
    model: str = "text-embedding-3-small"


@dataclass
class EmbeddingUsage:
    """Token usage for embeddings."""

    total_tokens: int = 0


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""

    model: str = ""
    embeddings: List[List[float]] = field(default_factory=list)
    usage: EmbeddingUsage = field(default_factory=EmbeddingUsage)


# =============================================================================
# Generate Object Types
# =============================================================================


@dataclass
class GenerateObjectRequest:
    """Request for structured JSON output."""

    model: str
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    schema: Dict[str, Any] = field(default_factory=dict)
    schema_name: Optional[str] = None
    schema_description: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class GenerateObjectResponse:
    """Response with structured JSON output."""

    object: Any = None
    usage: Optional[Usage] = None


# =============================================================================
# Image Generation Types
# =============================================================================


@dataclass
class GeneratedImage:
    """A single generated image."""

    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


@dataclass
class ImageGenerationRequest:
    """Request for image generation."""

    prompt: str
    model: Optional[str] = None
    n: Optional[int] = None
    size: Optional[str] = None
    quality: Optional[str] = None
    style: Optional[str] = None
    response_format: Optional[str] = None


@dataclass
class ImageGenerationResponse:
    """Response from image generation."""

    images: List[GeneratedImage] = field(default_factory=list)
    model: str = ""
    provider: str = ""


# =============================================================================
# RAG Types
# =============================================================================


@dataclass
class RagRequest:
    """Request for RAG (Retrieval-Augmented Generation)."""

    model: str
    messages: List[Dict[str, str]]
    namespace: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    limit: int = 5
    threshold: float = 0.5
    include_sources: bool = True


@dataclass
class RagSource:
    """A source document retrieved by RAG."""

    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity: float = 0.0


@dataclass
class RagResponse:
    """Response from RAG."""

    message: Dict[str, str] = field(default_factory=dict)
    model: str = ""
    provider: str = ""
    usage: Optional[Usage] = None
    sources: Optional[List[RagSource]] = None
    latency_ms: int = 0


@dataclass
class RagStreamChunk:
    """A chunk from streaming RAG response."""

    type: str = ""  # 'sources' | 'content'
    delta: Optional[str] = None
    finish_reason: Optional[str] = None
    sources: Optional[List[RagSource]] = None


# =============================================================================
# Agents Types
# =============================================================================


@dataclass
class AgentConfig:
    """Configuration for an AI agent."""

    model: str = ""
    system_prompt: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    temperature: Optional[float] = None


@dataclass
class Agent:
    """An AI agent."""

    id: str = ""
    name: str = ""
    description: Optional[str] = None
    is_active: bool = True
    shadow_mode: bool = False
    created_at: str = ""
    updated_at: Optional[str] = None
    config: Optional[AgentConfig] = None


@dataclass
class AgentListItem:
    """An AI agent list item (summary)."""

    id: str = ""
    name: str = ""
    description: Optional[str] = None
    is_active: bool = True
    shadow_mode: bool = False
    created_at: str = ""


@dataclass
class CreateAgentParams:
    """Parameters for creating an agent."""

    name: str
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


@dataclass
class UpdateAgentParams:
    """Parameters for updating an agent."""

    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    shadow_mode: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None


@dataclass
class AgentKey:
    """An API key for an agent."""

    id: str = ""
    name: str = ""
    key_prefix: str = ""
    full_key: Optional[str] = None
    environment: str = ""
    key_type: str = ""
    agent_id: str = ""
    created_at: str = ""


@dataclass
class CreateAgentKeyParams:
    """Parameters for creating an agent key."""

    name: Optional[str] = None
    environment: Optional[str] = None
    key_type: Optional[str] = None
    allowed_domains: Optional[List[str]] = None


# =============================================================================
# Memory Types
# =============================================================================


@dataclass
class MemoryNamespace:
    """A memory namespace for vector storage."""

    id: str = ""
    name: str = ""
    description: Optional[str] = None
    embedding_model: str = ""
    dimensions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_count: Optional[int] = None
    created_at: str = ""


@dataclass
class CreateNamespaceOptions:
    """Options for creating a memory namespace."""

    name: str
    description: Optional[str] = None
    embedding_model: Optional[str] = None
    dimensions: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Memory:
    """A stored memory."""

    id: str = ""
    namespace: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity: Optional[float] = None
    expires_at: Optional[str] = None
    created_at: str = ""
    updated_at: Optional[str] = None


@dataclass
class StoreMemoryOptions:
    """Options for storing a memory."""

    namespace: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    expires_at: Optional[Union[str, Any]] = None  # str (ISO) or datetime


@dataclass
class SearchMemoryOptions:
    """Options for searching memories."""

    namespace: str
    query: str
    limit: Optional[int] = None
    threshold: Optional[float] = None
    filter: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Result of a memory search."""

    results: List[Memory] = field(default_factory=list)
    query: str = ""
    namespace: str = ""
    count: int = 0
    latency_ms: int = 0


# =============================================================================
# Telemetry Types
# =============================================================================


@dataclass
class WebTelemetryPayload:
    """Payload for reporting a web request."""

    host: str
    method: str
    path: str
    status_code: int
    request_id: Optional[str] = None
    query_string: Optional[str] = None
    message: Optional[str] = None
    user_agent: Optional[str] = None
    referer: Optional[str] = None
    ip_address: Optional[str] = None
    country_code: Optional[str] = None
    latency_ms: Optional[int] = None


# =============================================================================
# Responses API Types
# =============================================================================


ResponseInputItem = Union[
    Dict[str, Any],  # message, function_call, function_call_output, file
]


WebSearchTool = Dict[str, Any]  # { type: 'web_search_preview', ... }


FileSearchTool = Dict[str, Any]  # { type: 'file_search', ... }


CodeInterpreterTool = Dict[str, Any]  # { type: 'code_interpreter' }


ResponsesTool = Union[
    WebSearchTool,
    FileSearchTool,
    CodeInterpreterTool,
    ToolDefinition,
]

UrlCitation = Dict[str, Any]


@dataclass
class ResponsesOutputItem:
    """A single output item from a Responses API response."""

    id: str = ""
    type: str = "message"
    status: Optional[str] = None
    role: Optional[str] = None
    content: Optional[List[Dict[str, Any]]] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class ResponsesRequest:
    """Request for the Responses API."""

    model: str
    input: Union[str, List[ResponseInputItem]]
    instructions: Optional[str] = None
    tools: Optional[List[ResponsesTool]] = None
    tool_choice: Optional[Any] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    store: Optional[bool] = None
    metadata: Optional[Dict[str, str]] = None
    previous_response_id: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    truncation: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    include: Optional[List[str]] = None
    stream: Optional[bool] = None
    user: Optional[str] = None


@dataclass
class ResponsesResponse:
    """Response from the Responses API."""

    id: str = ""
    object: str = "response"
    created: int = 0
    model: str = ""
    output: List[ResponsesOutputItem] = field(default_factory=list)
    usage: Optional[Dict[str, int]] = None
    status: str = "completed"
    metadata: Optional[Dict[str, str]] = None


# =============================================================================
# Project Types (keep existing)
# =============================================================================


@dataclass
class Project:
    """A Cencori project."""

    id: str
    name: str
    slug: str
    description: str
    status: str
    visibility: str
    created_at: str
    updated_at: str
    stats: Optional["Stats"] = None


@dataclass
class CreateProjectParams:
    """Parameters for creating a project."""

    name: str
    description: Optional[str] = None
    visibility: Optional[str] = None


@dataclass
class Stats:
    """Project statistics."""

    total_requests: int
    total_cost_usd: float
    last_used_at: Optional[str] = None


# =============================================================================
# API Key Types (keep existing)
# =============================================================================


@dataclass
class APIKey:
    """A Cencori API key."""

    id: str
    name: str
    environment: str
    created_at: str
    prefix: Optional[str] = None
    key: Optional[str] = None
    last_used_at: Optional[str] = None
    usage_count: Optional[int] = None


@dataclass
class CreateAPIKeyParams:
    """Parameters for creating an API key."""

    name: str
    environment: str


@dataclass
class DailyStat:
    """Daily usage statistic."""

    date: str
    count: int
    cost_usd: float


@dataclass
class KeyUsageStats:
    """Usage statistics for an API key."""

    key_id: str
    total_requests: int
    total_cost_usd: float
    last_used_at: str
    requests_by_day: List[DailyStat]
    requests_by_model: Dict[str, int]


# =============================================================================
# Metrics Types (keep existing)
# =============================================================================


@dataclass
class RequestMetrics:
    total: int = 0
    success: int = 0
    error: int = 0
    filtered: int = 0
    success_rate: float = 0.0


@dataclass
class CostMetrics:
    total_usd: float = 0.0
    average_per_request_usd: float = 0.0


@dataclass
class TokenMetrics:
    prompt: int = 0
    completion: int = 0
    total: int = 0


@dataclass
class LatencyMetrics:
    avg_ms: int = 0
    p50_ms: int = 0
    p90_ms: int = 0
    p99_ms: int = 0


@dataclass
class Breakdown:
    requests: int = 0
    cost_usd: float = 0.0


@dataclass
class MetricsResponse:
    """Response from metrics API."""

    period: str = ""
    start_date: str = ""
    end_date: str = ""
    requests: RequestMetrics = field(default_factory=RequestMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    providers: Dict[str, Breakdown] = field(default_factory=dict)
    models: Dict[str, Breakdown] = field(default_factory=dict)
