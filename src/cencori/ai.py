"""AI module for chat completions, embeddings, streaming, and more."""

import json
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

import httpx

from .errors import AuthenticationError, CencoriError, RateLimitError
from .types import (
    ChatResponse,
    EmbeddingResponse,
    EmbeddingUsage,
    GeneratedImage,
    GenerateObjectResponse,
    ImageGenerationResponse,
    RagResponse,
    RagSource,
    RagStreamChunk,
    ResponsesOutputItem,
    ResponsesResponse,
    StreamChunk,
    ToolCall,
    Usage,
)

if TYPE_CHECKING:
    from .client import Cencori


class AIModule:
    """
    AI module for chat completions, embeddings, streaming, and more.

    Provides access to OpenAI, Anthropic, and Google models through
    Cencori's unified API with built-in security, logging, and cost tracking.
    """

    def __init__(self, client: "Cencori") -> None:
        self._client = client

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    @staticmethod
    def _parse_tool_calls(raw: Any) -> Optional[List[ToolCall]]:
        """Parse tool calls from API response data."""
        if not raw:
            return None
        return [
            ToolCall(
                id=tc.get("id", ""),
                type="function",
                function={
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", ""),
                },
            )
            for tc in raw
        ]

    def _parse_chat_response(self, data: Dict[str, Any], model: str) -> ChatResponse:
        """Parse a chat response from the API, supporting both direct and OpenAI-compat formats."""
        choice = None
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]

        content = (
            data.get("content")
            or (choice.get("message", {}).get("content") if choice else None)
            or ""
        )

        finish_reason = data.get("finish_reason") or (
            choice.get("finish_reason") if choice else None
        )

        raw_tool_calls = (
            data.get("toolCalls")
            or data.get("tool_calls")
            or (choice.get("message", {}).get("tool_calls") if choice else None)
        )
        tool_calls = self._parse_tool_calls(raw_tool_calls)

        provider = data.get("provider")
        cost_usd = data.get("cost_usd")

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0) or usage_data.get("promptTokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0)
            or usage_data.get("completionTokens", 0),
            total_tokens=usage_data.get("total_tokens", 0) or usage_data.get("totalTokens", 0),
        )

        return ChatResponse(
            id=data.get("id", f"chatcmpl-{id(data)}"),
            model=data.get("model", model),
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            provider=provider,
            cost_usd=cost_usd,
        )

    # =========================================================================
    # Chat Methods
    # =========================================================================

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-2.5-flash",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        prompt: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        """
        Send a chat completion request (non-streaming).

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: AI model to use (default: gemini-2.5-flash)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            tools: Tool definitions for function calling
            tool_choice: How the model chooses to call tools
            prompt: Reference a named prompt from the Prompt Registry
            user_id: Optional user ID for rate limiting

        Returns:
            ChatResponse with content, usage, and tool calls

        Raises:
            AuthenticationError: Invalid API key
            RateLimitError: Rate limit exceeded
            SafetyError: Content blocked by safety filters

        Example:
            >>> response = cencori.ai.chat(
            ...     messages=[{"role": "user", "content": "Hello!"}],
            ...     model="gpt-4o"
            ... )
            >>> print(response.content)
        """
        payload: Dict[str, Any] = {
            "messages": messages,
            "model": model,
            "stream": False,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["maxTokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["toolChoice"] = tool_choice
        if prompt is not None:
            payload["prompt"] = prompt
        if user_id is not None:
            payload["userId"] = user_id

        data = self._client._request("POST", "/api/ai/chat", json=payload)

        return self._parse_chat_response(data, model)

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-2.5-flash",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        prompt: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Iterator[StreamChunk]:
        """
        Send a chat completion request with streaming.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: AI model to use (default: gemini-2.5-flash)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            tools: Tool definitions for function calling
            tool_choice: How the model chooses to call tools
            prompt: Reference a named prompt from the Prompt Registry
            user_id: Optional user ID for rate limiting

        Yields:
            StreamChunk objects with delta text

        Raises:
            AuthenticationError: Invalid API key
            RateLimitError: Rate limit exceeded
            SafetyError: Content blocked by safety filters

        Example:
            >>> for chunk in cencori.ai.chat_stream(
            ...     messages=[{"role": "user", "content": "Tell me a story"}]
            ... ):
            ...     print(chunk.delta, end="", flush=True)
        """
        payload: Dict[str, Any] = {
            "messages": messages,
            "model": model,
            "stream": True,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["maxTokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["toolChoice"] = tool_choice
        if prompt is not None:
            payload["prompt"] = prompt
        if user_id is not None:
            payload["userId"] = user_id

        url = f"{self._client._base_url}/api/ai/chat"
        headers = {
            "Content-Type": "application/json",
            "CENCORI_API_KEY": self._client._api_key,
        }

        with httpx.Client(timeout=60.0) as http_client:
            with http_client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code == 401:
                    raise AuthenticationError()
                if response.status_code == 429:
                    raise RateLimitError()
                if not response.is_success:
                    raise CencoriError(f"Request failed with status {response.status_code}")

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix

                    if data_str == "[DONE]":
                        return

                    try:
                        data = json.loads(data_str)

                        # Check for error in stream
                        if "error" in data:
                            yield StreamChunk(
                                delta="",
                                error=data.get("error"),
                            )
                            return

                        raw_tool_calls = data.get("toolCalls") or data.get("tool_calls")
                        tool_calls = self._parse_tool_calls(raw_tool_calls)

                        yield StreamChunk(
                            delta=data.get("delta", ""),
                            finish_reason=data.get("finish_reason"),
                            tool_calls=tool_calls,
                        )
                    except json.JSONDecodeError:
                        continue

    # =========================================================================
    # Completions Method
    # =========================================================================

    def completions(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """
        Create a text completion (wraps chat internally).

        Args:
            prompt: The text prompt to complete
            model: AI model to use (default: gemini-2.5-flash)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response

        Returns:
            ChatResponse with the completion

        Example:
            >>> response = cencori.ai.completions(
            ...     prompt="Write a haiku about coding",
            ...     model="gpt-4o"
            ... )
            >>> print(response.content)
        """
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # =========================================================================
    # Embeddings Method
    # =========================================================================

    def embeddings(
        self,
        input: Union[str, List[str]],
        model: str = "text-embedding-3-small",
    ) -> EmbeddingResponse:
        """
        Generate embeddings for text.

        Args:
            input: Text or list of texts to embed
            model: Embedding model to use (default: text-embedding-3-small)

        Returns:
            EmbeddingResponse with embeddings and usage

        Example:
            >>> response = cencori.ai.embeddings(
            ...     input="Hello world",
            ...     model="text-embedding-3-small"
            ... )
            >>> print(len(response.embeddings[0]))  # Embedding dimension
        """
        payload: Dict[str, Any] = {
            "input": input,
            "model": model,
        }

        data = self._client._request("POST", "/api/ai/embeddings", json=payload)

        # Extract embeddings from response
        raw_data = data.get("data", [])
        embeddings_list = [item["embedding"] for item in raw_data]

        return EmbeddingResponse(
            model=data.get("model", model),
            embeddings=embeddings_list,
            usage=EmbeddingUsage(
                total_tokens=data.get("usage", {}).get("total_tokens", 0),
            ),
        )

    # =========================================================================
    # Generate Object (Structured Output)
    # =========================================================================

    def generate_object(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        schema_name: Optional[str] = None,
        schema_description: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerateObjectResponse:
        """
        Generate structured JSON output using function calling.

        Args:
            model: AI model to use
            prompt: Text prompt (alternative to messages)
            messages: Chat-style input (alternative to prompt)
            schema: JSON Schema for the expected output
            schema_name: Schema name for the model
            schema_description: Schema description
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            GenerateObjectResponse with parsed object and usage

        Example:
            >>> response = cencori.ai.generate_object(
            ...     model="gpt-4o",
            ...     prompt="Generate a user profile",
            ...     schema={
            ...         "type": "object",
            ...         "properties": {
            ...             "name": {"type": "string"},
            ...             "age": {"type": "number"},
            ...         },
            ...         "required": ["name", "age"],
            ...     },
            ... )
            >>> print(response.object)
        """
        resolved_messages = messages or [{"role": "user", "content": prompt or ""}]
        schema_name_resolved = schema_name or "generate_object"
        schema_description_resolved = (
            schema_description or "Generate a structured object matching the schema"
        )

        payload: Dict[str, Any] = {
            "model": model,
            "messages": resolved_messages,
            "stream": False,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["maxTokens"] = max_tokens

        # Use function calling to enforce JSON schema
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": schema_name_resolved,
                    "description": schema_description_resolved,
                    "parameters": schema or {},
                },
            }
        ]
        payload["toolChoice"] = {
            "type": "function",
            "function": {"name": schema_name_resolved},
        }

        data = self._client._request("POST", "/api/ai/chat", json=payload)

        # Extract tool call arguments
        raw_tool_calls = (
            data.get("toolCalls")
            or data.get("tool_calls")
            or (
                data.get("choices", [{}])[0].get("message", {}).get("tool_calls")
                if data.get("choices")
                else None
            )
        )

        if not raw_tool_calls:
            raise CencoriError("Model did not return structured output")

        try:
            parsed_object = json.loads(raw_tool_calls[0]["function"]["arguments"])
        except (json.JSONDecodeError, KeyError, IndexError):
            raise CencoriError("Failed to parse structured output as JSON")

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0) or usage_data.get("promptTokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0)
            or usage_data.get("completionTokens", 0),
            total_tokens=usage_data.get("total_tokens", 0) or usage_data.get("totalTokens", 0),
        )

        return GenerateObjectResponse(
            object=parsed_object,
            usage=usage,
        )

    # =========================================================================
    # Image Generation
    # =========================================================================

    def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        style: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> ImageGenerationResponse:
        """
        Generate images from a text prompt.

        Args:
            prompt: Text prompt describing the image
            model: Model to use (default: dall-e-3)
            n: Number of images to generate
            size: Image size (e.g., '1024x1024')
            quality: Image quality ('standard' or 'hd')
            style: Image style ('vivid' or 'natural')
            response_format: Response format ('url' or 'b64_json')

        Returns:
            ImageGenerationResponse with generated images

        Example:
            >>> response = cencori.ai.generate_image(
            ...     prompt="A futuristic city at sunset",
            ...     model="dall-e-3",
            ... )
            >>> print(response.images[0].url)
        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
        }

        if model is not None:
            payload["model"] = model
        if n is not None:
            payload["n"] = n
        if size is not None:
            payload["size"] = size
        if quality is not None:
            payload["quality"] = quality
        if style is not None:
            payload["style"] = style
        if response_format is not None:
            payload["responseFormat"] = response_format

        data = self._client._request("POST", "/api/ai/images/generate", json=payload)

        images = [
            GeneratedImage(
                url=img.get("url"),
                b64_json=img.get("b64_json"),
                revised_prompt=img.get("revisedPrompt"),
            )
            for img in data.get("images", [])
        ]

        return ImageGenerationResponse(
            images=images,
            model=data.get("model", ""),
            provider=data.get("provider", ""),
        )

    # =========================================================================
    # RAG (Retrieval-Augmented Generation)
    # =========================================================================

    def rag(
        self,
        model: str,
        messages: List[Dict[str, str]],
        namespace: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        limit: int = 5,
        threshold: float = 0.5,
        include_sources: bool = True,
    ) -> RagResponse:
        """
        Chat with automatic memory context (RAG).

        Searches the memory namespace for relevant context and includes it
        in the prompt automatically.

        Args:
            model: AI model to use
            messages: Chat messages
            namespace: Memory namespace to search
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            limit: Number of memories to retrieve
            threshold: Similarity threshold for retrieval
            include_sources: Whether to include source documents in response

        Returns:
            RagResponse with message, sources, and usage

        Example:
            >>> response = cencori.ai.rag(
            ...     model="gpt-4o",
            ...     messages=[{"role": "user", "content": "What are our policies?"}],
            ...     namespace="company-docs",
            ... )
            >>> print(response.message["content"])
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "namespace": namespace,
            "limit": limit,
            "threshold": threshold,
            "include_sources": include_sources,
            "stream": False,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["maxTokens"] = max_tokens

        data = self._client._request("POST", "/api/ai/rag", json=payload)

        sources = None
        if "sources" in data and data["sources"]:
            sources = [
                RagSource(
                    content=s.get("content", ""),
                    metadata=s.get("metadata", {}),
                    similarity=s.get("similarity", 0.0),
                )
                for s in data["sources"]
            ]

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return RagResponse(
            message=data.get("message", {}),
            model=data.get("model", model),
            provider=data.get("provider", ""),
            usage=usage,
            sources=sources,
            latency_ms=data.get("latency_ms", 0),
        )

    def rag_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        namespace: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        limit: int = 5,
        threshold: float = 0.5,
        include_sources: bool = True,
    ) -> Iterator[RagStreamChunk]:
        """
        Stream RAG responses with automatic memory context.

        Args:
            Same as rag() but streams the response.

        Yields:
            RagStreamChunk with type 'sources' or 'content'

        Example:
            >>> for chunk in cencori.ai.rag_stream(
            ...     model="gpt-4o",
            ...     messages=[{"role": "user", "content": "What are our policies?"}],
            ...     namespace="company-docs",
            ... ):
            ...     if chunk.type == "content":
            ...         print(chunk.delta, end="", flush=True)
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "namespace": namespace,
            "limit": limit,
            "threshold": threshold,
            "include_sources": include_sources,
            "stream": True,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["maxTokens"] = max_tokens

        url = f"{self._client._base_url}/api/ai/rag"
        headers = {
            "Content-Type": "application/json",
            "CENCORI_API_KEY": self._client._api_key,
        }

        with httpx.Client(timeout=60.0) as http_client:
            with http_client.stream("POST", url, json=payload, headers=headers) as response:
                if not response.is_success:
                    raise CencoriError(f"Request failed with status {response.status_code}")

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]

                    if data_str == "[DONE]":
                        return

                    try:
                        data_parsed = json.loads(data_str)
                        chunk_type = data_parsed.get("type", "content")

                        sources = None
                        if "sources" in data_parsed and data_parsed["sources"]:
                            sources = [
                                RagSource(
                                    content=s.get("content", ""),
                                    metadata=s.get("metadata", {}),
                                    similarity=s.get("similarity", 0.0),
                                )
                                for s in data_parsed["sources"]
                            ]

                        yield RagStreamChunk(
                            type=chunk_type,
                            delta=data_parsed.get("delta"),
                            finish_reason=data_parsed.get("finish_reason"),
                            sources=sources,
                        )
                    except json.JSONDecodeError:
                        continue

    # =========================================================================
    # Responses API (OpenAI-compatible)
    # =========================================================================

    def responses(
        self,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        instructions: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        store: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
        previous_response_id: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        truncation: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        user: Optional[str] = None,
    ) -> ResponsesResponse:
        """
        Send a request to the OpenAI-compatible Responses API.

        Supports built-in tools: web_search_preview, file_search, code_interpreter.

        Args:
            model: AI model to use
            input: Input string or list of response input items
            instructions: System instructions
            tools: Built-in tools (web_search_preview, file_search, code_interpreter) or function tools
            tool_choice: Tool selection strategy
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens
            top_p: Nucleus sampling parameter
            store: Whether to store the response
            metadata: Metadata to associate with the response
            previous_response_id: ID of the previous response for threading
            parallel_tool_calls: Whether to make parallel tool calls
            truncation: Truncation strategy
            response_format: Response format configuration
            include: Additional data to include
            user: User identifier

        Returns:
            ResponsesResponse with output items

        Example:
            >>> response = cencori.ai.responses(
            ...     model="gpt-4o",
            ...     input="What is the weather in SF?",
            ...     tools=[{"type": "web_search_preview"}],
            ... )
        """
        payload: Dict[str, Any] = {
            "model": model,
            "input": input,
            "stream": False,
        }

        if instructions is not None:
            payload["instructions"] = instructions
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if store is not None:
            payload["store"] = store
        if metadata is not None:
            payload["metadata"] = metadata
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls
        if truncation is not None:
            payload["truncation"] = truncation
        if response_format is not None:
            payload["response_format"] = response_format
        if include is not None:
            payload["include"] = include
        if user is not None:
            payload["user"] = user

        data = self._client._request("POST", "/v1/responses", json=payload)

        output_items = [
            ResponsesOutputItem(
                id=item.get("id", ""),
                type=item.get("type", "message"),
                status=item.get("status"),
                role=item.get("role"),
                content=item.get("content"),
                call_id=item.get("call_id"),
                name=item.get("name"),
                arguments=item.get("arguments"),
                output=item.get("output"),
                error=item.get("error"),
            )
            for item in data.get("output", [])
        ]

        return ResponsesResponse(
            id=data.get("id", ""),
            object=data.get("object", "response"),
            created=data.get("created", 0),
            model=data.get("model", model),
            output=output_items,
            usage=data.get("usage"),
            status=data.get("status", "completed"),
            metadata=data.get("metadata"),
        )

    def responses_stream(
        self,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        instructions: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        store: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
        previous_response_id: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        truncation: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        user: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream responses from the Responses API via SSE.

        Yields dicts with 'type' (event type) and 'data' (parsed JSON).

        Args:
            Same as responses() but streams the response.

        Yields:
            Dicts with 'type' and 'data' keys

        Example:
            >>> for event in cencori.ai.responses_stream(
            ...     model="gpt-4o",
            ...     input="Tell me about AI",
            ... ):
            ...     if event["type"] == "response.output_text.delta":
            ...         print(event["data"]["delta"], end="", flush=True)
        """
        payload: Dict[str, Any] = {
            "model": model,
            "input": input,
            "stream": True,
        }

        if instructions is not None:
            payload["instructions"] = instructions
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if store is not None:
            payload["store"] = store
        if metadata is not None:
            payload["metadata"] = metadata
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls
        if truncation is not None:
            payload["truncation"] = truncation
        if response_format is not None:
            payload["response_format"] = response_format
        if include is not None:
            payload["include"] = include
        if user is not None:
            payload["user"] = user

        url = f"{self._client._base_url}/v1/responses"
        headers = {
            "Content-Type": "application/json",
            "CENCORI_API_KEY": self._client._api_key,
        }

        with httpx.Client(timeout=60.0) as http_client:
            with http_client.stream("POST", url, json=payload, headers=headers) as response:
                if not response.is_success:
                    raise CencoriError(f"Request failed with status {response.status_code}")

                event_type = ""
                for line in response.iter_lines():
                    if line.strip() == "":
                        event_type = ""
                        continue
                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if not data_str:
                            continue
                        try:
                            parsed = json.loads(data_str)
                            yield {"type": event_type or "message", "data": parsed}
                        except json.JSONDecodeError:
                            continue
                        event_type = ""

    # =========================================================================
    # Async Methods
    # =========================================================================

    async def async_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-2.5-flash",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        prompt: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        """
        Send a chat completion request asynchronously.

        Same as chat() but async.
        """
        payload: Dict[str, Any] = {
            "messages": messages,
            "model": model,
            "stream": False,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["maxTokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["toolChoice"] = tool_choice
        if prompt is not None:
            payload["prompt"] = prompt
        if user_id is not None:
            payload["userId"] = user_id

        data = await self._client._async_request("POST", "/api/ai/chat", json=payload)

        return self._parse_chat_response(data, model)

    async def async_completions(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """
        Create a text completion asynchronously.

        Same as completions() but async.
        """
        return await self.async_chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def async_embeddings(
        self,
        input: Union[str, List[str]],
        model: str = "text-embedding-3-small",
    ) -> EmbeddingResponse:
        """
        Generate embeddings asynchronously.

        Same as embeddings() but async.
        """
        payload: Dict[str, Any] = {
            "input": input,
            "model": model,
        }

        data = await self._client._async_request("POST", "/api/ai/embeddings", json=payload)

        raw_data = data.get("data", [])
        embeddings_list = [item["embedding"] for item in raw_data]

        return EmbeddingResponse(
            model=data.get("model", model),
            embeddings=embeddings_list,
            usage=EmbeddingUsage(
                total_tokens=data.get("usage", {}).get("total_tokens", 0),
            ),
        )

    async def async_generate_object(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        schema_name: Optional[str] = None,
        schema_description: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerateObjectResponse:
        """Async version of generate_object."""
        resolved_messages = messages or [{"role": "user", "content": prompt or ""}]
        schema_name_resolved = schema_name or "generate_object"
        schema_description_resolved = (
            schema_description or "Generate a structured object matching the schema"
        )

        payload: Dict[str, Any] = {
            "model": model,
            "messages": resolved_messages,
            "stream": False,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["maxTokens"] = max_tokens

        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": schema_name_resolved,
                    "description": schema_description_resolved,
                    "parameters": schema or {},
                },
            }
        ]
        payload["toolChoice"] = {
            "type": "function",
            "function": {"name": schema_name_resolved},
        }

        data = await self._client._async_request("POST", "/api/ai/chat", json=payload)

        raw_tool_calls = (
            data.get("toolCalls")
            or data.get("tool_calls")
            or (
                data.get("choices", [{}])[0].get("message", {}).get("tool_calls")
                if data.get("choices")
                else None
            )
        )

        if not raw_tool_calls:
            raise CencoriError("Model did not return structured output")

        try:
            parsed_object = json.loads(raw_tool_calls[0]["function"]["arguments"])
        except (json.JSONDecodeError, KeyError, IndexError):
            raise CencoriError("Failed to parse structured output as JSON")

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0) or usage_data.get("promptTokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0)
            or usage_data.get("completionTokens", 0),
            total_tokens=usage_data.get("total_tokens", 0) or usage_data.get("totalTokens", 0),
        )

        return GenerateObjectResponse(object=parsed_object, usage=usage)

    async def async_generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        style: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> ImageGenerationResponse:
        """Async version of generate_image."""
        payload: Dict[str, Any] = {"prompt": prompt}
        if model is not None:
            payload["model"] = model
        if n is not None:
            payload["n"] = n
        if size is not None:
            payload["size"] = size
        if quality is not None:
            payload["quality"] = quality
        if style is not None:
            payload["style"] = style
        if response_format is not None:
            payload["responseFormat"] = response_format

        data = await self._client._async_request("POST", "/api/ai/images/generate", json=payload)

        images = [
            GeneratedImage(
                url=img.get("url"),
                b64_json=img.get("b64_json"),
                revised_prompt=img.get("revisedPrompt"),
            )
            for img in data.get("images", [])
        ]

        return ImageGenerationResponse(
            images=images,
            model=data.get("model", ""),
            provider=data.get("provider", ""),
        )

    async def async_rag(
        self,
        model: str,
        messages: List[Dict[str, str]],
        namespace: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        limit: int = 5,
        threshold: float = 0.5,
        include_sources: bool = True,
    ) -> RagResponse:
        """Async version of rag."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "namespace": namespace,
            "limit": limit,
            "threshold": threshold,
            "include_sources": include_sources,
            "stream": False,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["maxTokens"] = max_tokens

        data = await self._client._async_request("POST", "/api/ai/rag", json=payload)

        sources = None
        if "sources" in data and data["sources"]:
            sources = [
                RagSource(
                    content=s.get("content", ""),
                    metadata=s.get("metadata", {}),
                    similarity=s.get("similarity", 0.0),
                )
                for s in data["sources"]
            ]

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return RagResponse(
            message=data.get("message", {}),
            model=data.get("model", model),
            provider=data.get("provider", ""),
            usage=usage,
            sources=sources,
            latency_ms=data.get("latency_ms", 0),
        )

    async def async_responses(
        self,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        instructions: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        store: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
        previous_response_id: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        truncation: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        user: Optional[str] = None,
    ) -> ResponsesResponse:
        """Async version of responses."""
        payload: Dict[str, Any] = {
            "model": model,
            "input": input,
            "stream": False,
        }

        if instructions is not None:
            payload["instructions"] = instructions
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if store is not None:
            payload["store"] = store
        if metadata is not None:
            payload["metadata"] = metadata
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls
        if truncation is not None:
            payload["truncation"] = truncation
        if response_format is not None:
            payload["response_format"] = response_format
        if include is not None:
            payload["include"] = include
        if user is not None:
            payload["user"] = user

        data = await self._client._async_request("POST", "/v1/responses", json=payload)

        output_items = [
            ResponsesOutputItem(
                id=item.get("id", ""),
                type=item.get("type", "message"),
                status=item.get("status"),
                role=item.get("role"),
                content=item.get("content"),
                call_id=item.get("call_id"),
                name=item.get("name"),
                arguments=item.get("arguments"),
                output=item.get("output"),
                error=item.get("error"),
            )
            for item in data.get("output", [])
        ]

        return ResponsesResponse(
            id=data.get("id", ""),
            object=data.get("object", "response"),
            created=data.get("created", 0),
            model=data.get("model", model),
            output=output_items,
            usage=data.get("usage"),
            status=data.get("status", "completed"),
            metadata=data.get("metadata"),
        )
