"""Base Transport Abstraction for Colab Project Workspaces (V2).

Defines the interface for remote communication, file transfer, and execution.
All transports must enforce session ownership and fail closed on errors.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..provider import ManagedSessionRegistry


class BaseTransport(ABC):
    """Abstract base class for all Colab transports."""

    def __init__(self, registry: ManagedSessionRegistry):
        self.registry = registry

    @abstractmethod
    def upload_file(
        self,
        session_id: str,
        local_path: Union[str, Path],
        remote_path: str,
    ) -> None:
        """Uploads a local file to remote_path in the specified session."""
        pass

    @abstractmethod
    def download_file(
        self,
        session_id: str,
        remote_path: str,
        local_path: Union[str, Path],
    ) -> None:
        """Downloads a remote file from remote_path to local_path."""
        pass

    @abstractmethod
    def list_dir(
        self,
        session_id: str,
        remote_path: str = "content",
    ) -> List[str]:
        """Lists directory entries under remote_path in session."""
        pass

    @abstractmethod
    def remove_file(
        self,
        session_id: str,
        remote_path: str,
    ) -> None:
        """Removes a remote file from session."""
        pass

    @abstractmethod
    def exec_code(
        self,
        session_id: str,
        code: str,
        timeout: float = 60.0,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Executes code in session kernel, returning execution metadata and output."""
        pass

    @abstractmethod
    def check_health(self, session_id: str) -> bool:
        """Checks if session is reachable and responsive."""
        pass
