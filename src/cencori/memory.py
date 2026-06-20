"""Memory module for vector storage and semantic search."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .types import (
    CreateNamespaceOptions,
    Memory,
    MemoryNamespace,
    SearchMemoryOptions,
    SearchResult,
    StoreMemoryOptions,
)

if TYPE_CHECKING:
    from .client import Cencori


class MemoryModule:
    """
    Module for vector storage operations.

    Provides namespaced storage for RAG, conversation history,
    and semantic search.
    """

    def __init__(self, client: "Cencori") -> None:
        self._client = client

    # =========================================================================
    # Namespace Methods
    # =========================================================================

    def create_namespace(
        self,
        name: str,
        description: Optional[str] = None,
        embedding_model: Optional[str] = None,
        dimensions: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryNamespace:
        """
        Create a new memory namespace.

        Args:
            name: Namespace name
            description: Optional description
            embedding_model: Embedding model to use
            dimensions: Embedding dimensions
            metadata: Additional metadata

        Returns:
            Created MemoryNamespace

        Example:
            >>> ns = cencori.memory.create_namespace(
            ...     name="conversations",
            ...     description="User conversation history",
            ... )
        """
        payload: Dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if embedding_model is not None:
            payload["embeddingModel"] = embedding_model
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if metadata is not None:
            payload["metadata"] = metadata

        data = self._client._request("POST", "/api/memory/namespaces", json=payload)

        return MemoryNamespace(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description"),
            embedding_model=data.get("embeddingModel", ""),
            dimensions=data.get("dimensions", 0),
            metadata=data.get("metadata", {}),
            memory_count=data.get("memoryCount"),
            created_at=data.get("createdAt", ""),
        )

    def list_namespaces(self) -> List[MemoryNamespace]:
        """
        List all memory namespaces.

        Returns:
            List of MemoryNamespace objects

        Example:
            >>> namespaces = cencori.memory.list_namespaces()
        """
        data = self._client._request("GET", "/api/memory/namespaces")
        raw = data.get("namespaces", [])
        return [
            MemoryNamespace(
                id=ns.get("id", ""),
                name=ns.get("name", ""),
                description=ns.get("description"),
                embedding_model=ns.get("embeddingModel", ""),
                dimensions=ns.get("dimensions", 0),
                metadata=ns.get("metadata", {}),
                memory_count=ns.get("memoryCount"),
                created_at=ns.get("createdAt", ""),
            )
            for ns in raw
        ]

    # =========================================================================
    # Memory Methods
    # =========================================================================

    def store(
        self,
        namespace: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[Union[str, datetime]] = None,
    ) -> Memory:
        """
        Store a memory in a namespace.

        Args:
            namespace: Namespace to store in
            content: Memory content text
            metadata: Optional metadata
            expires_at: Optional expiration time (ISO string or datetime)

        Returns:
            Stored Memory

        Example:
            >>> memory = cencori.memory.store(
            ...     namespace="conversations",
            ...     content="User asked about pricing plans",
            ...     metadata={"userId": "user_123"},
            ... )
        """
        payload: Dict[str, Any] = {
            "namespace": namespace,
            "content": content,
        }

        if metadata is not None:
            payload["metadata"] = metadata
        if expires_at is not None:
            if isinstance(expires_at, datetime):
                payload["expiresAt"] = expires_at.isoformat()
            else:
                payload["expiresAt"] = expires_at

        data = self._client._request("POST", "/api/memory/store", json=payload)

        return Memory(
            id=data.get("id", ""),
            namespace=data.get("namespace", namespace),
            content=data.get("content", content),
            metadata=data.get("metadata", {}),
            similarity=data.get("similarity"),
            expires_at=data.get("expiresAt"),
            created_at=data.get("createdAt", ""),
            updated_at=data.get("updatedAt"),
        )

    def search(
        self,
        namespace: str,
        query: str,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """
        Semantic search across memories.

        Args:
            namespace: Namespace to search
            query: Search query text
            limit: Maximum number of results
            threshold: Similarity threshold (0-1)
            filter: Metadata filter

        Returns:
            SearchResult with matching memories

        Example:
            >>> results = cencori.memory.search(
            ...     namespace="conversations",
            ...     query="what did we discuss about pricing?",
            ...     limit=5,
            ... )
            >>> for mem in results.results:
            ...     print(mem.content)
        """
        payload: Dict[str, Any] = {
            "namespace": namespace,
            "query": query,
        }

        if limit is not None:
            payload["limit"] = limit
        if threshold is not None:
            payload["threshold"] = threshold
        if filter is not None:
            payload["filter"] = filter

        data = self._client._request("POST", "/api/memory/search", json=payload)

        results = [
            Memory(
                id=m.get("id", ""),
                namespace=m.get("namespace", namespace),
                content=m.get("content", ""),
                metadata=m.get("metadata", {}),
                similarity=m.get("similarity"),
                expires_at=m.get("expiresAt"),
                created_at=m.get("createdAt", ""),
                updated_at=m.get("updatedAt"),
            )
            for m in data.get("results", [])
        ]

        return SearchResult(
            results=results,
            query=data.get("query", query),
            namespace=data.get("namespace", namespace),
            count=data.get("count", len(results)),
            latency_ms=data.get("latencyMs", 0),
        )

    def get(self, memory_id: str) -> Memory:
        """
        Get a memory by ID.

        Args:
            memory_id: The memory ID

        Returns:
            Memory object

        Example:
            >>> memory = cencori.memory.get("mem_123")
        """
        data = self._client._request("GET", f"/api/memory/{memory_id}")

        return Memory(
            id=data.get("id", memory_id),
            namespace=data.get("namespace", ""),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            similarity=data.get("similarity"),
            expires_at=data.get("expiresAt"),
            created_at=data.get("createdAt", ""),
            updated_at=data.get("updatedAt"),
        )

    def delete(self, memory_id: str) -> Dict[str, Any]:
        """
        Delete a memory by ID.

        Args:
            memory_id: The memory ID

        Returns:
            Response dict with deletion status

        Example:
            >>> result = cencori.memory.delete("mem_123")
            >>> print(result["deleted"])
        """
        return self._client._request("DELETE", f"/api/memory/{memory_id}")

    def store_batch(
        self,
        namespace: str,
        items: List[Dict[str, Any]],
    ) -> List[Memory]:
        """
        Store multiple memories in batch.

        Args:
            namespace: Namespace to store in
            items: List of dicts with 'content' and optional 'metadata'

        Returns:
            List of stored Memory objects

        Example:
            >>> memories = cencori.memory.store_batch(
            ...     namespace="conversations",
            ...     items=[
            ...         {"content": "User message 1", "metadata": {"userId": "123"}},
            ...         {"content": "User message 2"},
            ...     ],
            ... )
        """
        results = []
        for item in items:
            mem = self.store(
                namespace=namespace,
                content=item.get("content", ""),
                metadata=item.get("metadata"),
            )
            results.append(mem)
        return results

    def delete_by_filter(
        self,
        namespace: str,
        filter: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Delete all memories in a namespace matching a filter.

        Args:
            namespace: Namespace to delete from
            filter: Metadata filter to match

        Returns:
            Dict with deleted count

        Example:
            >>> result = cencori.memory.delete_by_filter(
            ...     namespace="conversations",
            ...     filter={"userId": "user_123"},
            ... )
            >>> print(f"Deleted {result['deleted']} memories")
        """
        # Search to find matching memories
        search_result = self.search(
            namespace=namespace,
            query="*",
            limit=1000,
            threshold=0,
            filter=filter,
        )

        # Delete each one
        deleted = 0
        for mem in search_result.results:
            self.delete(mem.id)
            deleted += 1

        return {"deleted": deleted}
