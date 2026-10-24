"""V2 Error Taxonomy for Colab Project Workspaces.

Establishes typed, deterministic error classes covering manifest validation,
transport abstraction, SSH safety guards, delta sync, and artifact recovery.
"""
from typing import Any, Dict, Optional
from .provider import (
    ArtifactMissingError,
    ColabProviderError,
    DependencyMissingError,
    InvalidJobSpecError,
)


class ColabV2Error(ColabProviderError):
    """Base error class for all V2 Colab Project Workspace operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Return a structured dictionary suitable for JSON serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class UpstreamCliError(ColabV2Error):
    """Error communicating with or invoking the official google-colab-cli."""
    pass


class UpstreamCliNotFoundError(UpstreamCliError, DependencyMissingError):
    """google-colab-cli executable was not found on PATH or specified location."""
    pass


class AutoAllocationBlockedError(ColabV2Error):
    """Raised when an operation would risk implicitly allocating a cloud VM without approval.
    
    Specifically guards against 'colab ssh -s <name>' auto-creating a missing session.
    """
    pass


class ManifestValidationError(ColabV2Error, InvalidJobSpecError):
    """Raised when project manifest fails schema, path traversal, or safety checks."""
    pass


class WorkspaceSyncError(ColabV2Error):
    """Raised when workspace file indexing, staging, diffing, or remote apply fails."""
    pass


class ArtifactValidationError(ColabV2Error, ArtifactMissingError):
    """Raised when expected output files/checkpoints are missing, truncated, or invalid."""
    pass


class TransportError(ColabV2Error):
    """Base error for communication with remote Colab session."""
    pass


class TransportUnavailableError(TransportError):
    """The requested transport mechanism (e.g. SSH proxy) is not available or supported."""
    pass


class SessionNotFoundError(TransportError):
    """The targeted Colab session does not exist locally or remotely."""
    pass


class UnmanagedSessionError(TransportError):
    """The targeted session exists on Colab but is not owned by this adapter instance."""
    pass


class SessionAllocationRequiredError(TransportError):
    """The requested operation requires an active Colab session, but none is allocated."""
    pass


class SshHandshakeError(TransportError):
    """Error occurred during SSH-over-WebSocket upgrade or authentication handshake."""

    def __init__(
        self,
        message: str,
        http_status: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if http_status is not None:
            details["http_status"] = http_status
        super().__init__(message, details=details)
        self.http_status = http_status
