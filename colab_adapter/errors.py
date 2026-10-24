"""V2 Error Taxonomy for Colab Project Workspaces.

Establishes typed, deterministic error classes covering manifest validation,
transport abstraction, SSH safety guards, delta sync, and artifact recovery.
Ensures sensitive credentials, authorization codes, and private keys are redacted.
"""
import re
from typing import Any, Dict, List, Optional, Union
from .provider import (
    AllocationError,
    ArtifactMissingError,
    CleanupError,
    ColabProviderError,
    DependencyMissingError,
    InvalidJobSpecError,
    RemoteExecutionError,
)

_SECRET_PATTERN = re.compile(
    r"(4/[0-9A-Za-z\-_]+|ya29\.[0-9A-Za-z\-_]+|eyJ[0-9A-Za-z\-_]+\.[0-9A-Za-z\-_]+|"
    r"(?:token|secret|password|key|client_secret|access_token|refresh_token)\s*[:=]\s*['\"]?[^\s'\"]+)",
    re.IGNORECASE,
)


def _sanitize_details(val: Any) -> Any:
    """Recursively redact potential secrets from error detail structures."""
    if isinstance(val, str):
        return _SECRET_PATTERN.sub("[REDACTED]", val)
    elif isinstance(val, list):
        return [_sanitize_details(v) for v in val]
    elif isinstance(val, tuple):
        return tuple(_sanitize_details(v) for v in val)
    elif isinstance(val, dict):
        sanitized = {}
        for k, v in val.items():
            if any(s in k.lower() for s in ("token", "secret", "password", "key", "auth")):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = _sanitize_details(v)
        return sanitized
    return val


class ColabV2Error(ColabProviderError):
    """Base error class for all V2 Colab Project Workspace operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        sanitized_msg = _sanitize_details(message)
        sanitized_details = _sanitize_details(details or {})
        super().__init__(sanitized_msg)
        self.message = sanitized_msg
        self.details = sanitized_details

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
    
    Guards against 'colab ssh -s <name>' auto-creating a missing session or any race condition.
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


class ProjectSessionError(ColabV2Error):
    """Base error for project session lifecycle operations."""
    pass


class LeaseExpiredError(ProjectSessionError):
    """Raised when an operation targets a session whose approved lease duration has expired."""
    pass


class InvalidStateTransitionError(ProjectSessionError):
    """Raised when an illegal state machine transition is attempted."""
    pass


class StorageError(ColabV2Error):
    """Base error for artifact storage operations."""
    pass


class ArtifactNotFoundError(StorageError, ArtifactMissingError):
    """Requested artifact was not found in local or remote store."""
    pass


class DriveAuthenticationError(StorageError):
    """Raised when Google Drive credentials are missing, expired, or invalid."""
    pass


class DriveQuotaExceededError(StorageError):
    """Raised when Google Drive storage quota or rate limit is exceeded."""
    pass


class DriveTransferError(StorageError):
    """Raised when resumable upload or download fails after retries."""
    pass


class DriveIntegrityError(StorageError, ArtifactValidationError):
    """Raised when remote Google Drive artifact checksum or size does not match verified local artifact."""
    pass


class BrokerError(ColabV2Error):
    """Base error for Workspace Broker operations."""
    pass


class BrokerAuthenticationError(BrokerError):
    """Raised when a client fails authentication with the Workspace Broker."""
    pass


class CheckpointPolicyError(ColabV2Error):
    """Raised when checkpoint policy validation fails."""
    pass


