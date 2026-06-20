"""Agents module for managing AI agents."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .types import (
    Agent,
    AgentConfig,
    AgentKey,
    AgentListItem,
    CreateAgentKeyParams,
    CreateAgentParams,
    UpdateAgentParams,
)

if TYPE_CHECKING:
    from .client import Cencori


class AgentsModule:
    """
    Module for managing AI agents.

    Provides CRUD operations for agents and agent API keys.
    """

    def __init__(self, client: "Cencori") -> None:
        self._client = client

    def create(
        self,
        name: str,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """
        Create a new AI agent.

        Args:
            name: Agent name
            description: Optional description
            config: Agent configuration (model, system_prompt, tools, temperature)

        Returns:
            The created Agent

        Example:
            >>> agent = cencori.agents.create(
            ...     name="my-agent",
            ...     config={"model": "gpt-4o", "system_prompt": "You are helpful"},
            ... )
        """
        payload: Dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if config is not None:
            payload["config"] = config

        data = self._client._request("POST", "/v1/agents", json=payload)
        return self._parse_agent(data)

    def list(self) -> List[AgentListItem]:
        """
        List all agents.

        Returns:
            List of AgentListItem objects

        Example:
            >>> agents = cencori.agents.list()
            >>> for agent in agents:
            ...     print(agent.name)
        """
        data = self._client._request("GET", "/v1/agents")
        raw = data.get("data", [])
        return [
            AgentListItem(
                id=a.get("id", ""),
                name=a.get("name", ""),
                description=a.get("description"),
                is_active=a.get("is_active", True),
                shadow_mode=a.get("shadow_mode", False),
                created_at=a.get("created_at", ""),
            )
            for a in raw
        ]

    def get(self, agent_id: str) -> Agent:
        """
        Get an agent by ID.

        Args:
            agent_id: The agent ID

        Returns:
            Agent with full details

        Example:
            >>> agent = cencori.agents.get("ag_123")
        """
        data = self._client._request("GET", f"/v1/agents/{agent_id}")
        return self._parse_agent(data)

    def update_config(
        self,
        agent_id: str,
        params: UpdateAgentParams,
    ) -> Agent:
        """
        Update an agent's configuration.

        Args:
            agent_id: The agent ID
            params: Parameters to update

        Returns:
            Updated Agent

        Example:
            >>> agent = cencori.agents.update_config(
            ...     "ag_123",
            ...     UpdateAgentParams(name="updated-name", config={"model": "gpt-4o"}),
            ... )
        """
        payload: Dict[str, Any] = {}
        if params.name is not None:
            payload["name"] = params.name
        if params.description is not None:
            payload["description"] = params.description
        if params.is_active is not None:
            payload["is_active"] = params.is_active
        if params.shadow_mode is not None:
            payload["shadow_mode"] = params.shadow_mode
        if params.config is not None:
            payload["config"] = params.config

        data = self._client._request("PATCH", f"/v1/agents/{agent_id}", json=payload)
        return self._parse_agent(data)

    def delete(self, agent_id: str) -> None:
        """
        Delete an agent.

        Args:
            agent_id: The agent ID

        Example:
            >>> cencori.agents.delete("ag_123")
        """
        self._client._request("DELETE", f"/v1/agents/{agent_id}")

    def create_key(
        self,
        agent_id: str,
        name: Optional[str] = None,
        environment: Optional[str] = None,
        key_type: Optional[str] = None,
        allowed_domains: Optional[List[str]] = None,
    ) -> AgentKey:
        """
        Create an API key for an agent.

        Args:
            agent_id: The agent ID
            name: Key name
            environment: Environment ('production' or 'test')
            key_type: Key type ('secret' or 'publishable')
            allowed_domains: Allowed domains for the key

        Returns:
            Created AgentKey (includes the full key on creation)

        Example:
            >>> key = cencori.agents.create_key(
            ...     "ag_123",
            ...     name="prod-key",
            ...     environment="production",
            ... )
        """
        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if environment is not None:
            payload["environment"] = environment
        if key_type is not None:
            payload["key_type"] = key_type
        if allowed_domains is not None:
            payload["allowed_domains"] = allowed_domains

        data = self._client._request("POST", f"/v1/agents/{agent_id}/keys", json=payload)

        return AgentKey(
            id=data.get("id", ""),
            name=data.get("name", ""),
            key_prefix=data.get("key_prefix", ""),
            full_key=data.get("full_key"),
            environment=data.get("environment", ""),
            key_type=data.get("key_type", ""),
            agent_id=data.get("agent_id", ""),
            created_at=data.get("created_at", ""),
        )

    @staticmethod
    def _parse_agent(data: Dict[str, Any]) -> Agent:
        """Parse agent data from API response."""
        config_raw = data.get("config")
        config = None
        if config_raw:
            config = AgentConfig(
                model=config_raw.get("model", ""),
                system_prompt=config_raw.get("system_prompt"),
                tools=config_raw.get("tools", []),
                temperature=config_raw.get("temperature"),
            )

        return Agent(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description"),
            is_active=data.get("is_active", True),
            shadow_mode=data.get("shadow_mode", False),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at"),
            config=config,
        )
