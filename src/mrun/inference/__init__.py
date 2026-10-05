"""Chat-facing protocol adapters for the separate compute-host inference service."""

from .app import (
    InferenceAppConfig,
    InferenceHost,
    create_inference_app,
    strict_json_object,
)
from .chat import (
    BoundChatTokenizer,
    ChatMessage,
    ChatValidationError,
    IncrementalTextDecoder,
    RenderedChat,
    TextDelta,
    canonical_chat_template_sha256,
    validate_chat_messages,
)
from .loader import (
    NATIVE_INFERENCE_BACKENDS,
    LoadedMixtralExpertPagedSlice,
    LoadedNativeInference,
    MixtralExpertPagedSliceConfig,
    NativeInferenceConfig,
    load_mixtral_expert_paged_slice,
    load_native_inference,
)
from .openai_protocol import (
    OpenAIChatRequest,
    SamplingOptions,
    StreamOptions,
    Usage,
    completion_chunk,
    completion_response,
    error_envelope,
    parse_chat_completion_request,
)

__all__ = [
    "BoundChatTokenizer",
    "ChatMessage",
    "ChatValidationError",
    "IncrementalTextDecoder",
    "InferenceAppConfig",
    "InferenceHost",
    "LoadedNativeInference",
    "LoadedMixtralExpertPagedSlice",
    "NATIVE_INFERENCE_BACKENDS",
    "NativeInferenceConfig",
    "MixtralExpertPagedSliceConfig",
    "OpenAIChatRequest",
    "RenderedChat",
    "SamplingOptions",
    "StreamOptions",
    "TextDelta",
    "Usage",
    "canonical_chat_template_sha256",
    "completion_chunk",
    "completion_response",
    "create_inference_app",
    "error_envelope",
    "load_native_inference",
    "load_mixtral_expert_paged_slice",
    "parse_chat_completion_request",
    "strict_json_object",
    "validate_chat_messages",
]
