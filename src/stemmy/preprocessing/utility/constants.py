# File format validation
SUPPORTED_FORMATS: list[str] = ["wav"]  # Only WAV confirmed for Sprint 1
OPTIONAL_FORMATS: list[str] = ["flac", "mp3"]  # Stretch goals

#########################
# Change by Ryan:
# Reason:
# Keep preprocessing validation aligned with the centralized audio configuration
# (single source of truth) without changing validator/extractor field names.
# What it does:
# Imports TARGET_SAMPLE_RATE from stemmy.constants while preserving the upstream
# metadata keys and message formats expected by AudioMetadataExtractor/AudioFileValidator.
#########################

# Sample rate validation
MIN_SAMPLE_RATE: int = 8000
MAX_SAMPLE_RATE: int = 192000

# Channel validation
MIN_CHANNELS: int = 1
MAX_CHANNELS: int = 2

# Duration validation
MIN_DURATION_SECONDS: float = 1.0
MAX_DURATION_SECONDS: float = 600.0

# File size validation
MAX_FILE_SIZE_MB: float = 500.0
BYTES_TO_MB: int = 1024 * 1024  # Conversion factor from bytes to megabytes


# Exception error messages
ERROR_FILE_NOT_EXIST: str = "File does not exist."
ERROR_FILE_NOT_READABLE: str = "File is not readable."
ERROR_FILE_TOO_LARGE: str = "File size exceeds the maximum limit of {max_size} MB."
ERROR_UNSUPPORTED_FORMAT: str = (
    "Unsupported file format: {format}. Supported formats are: {supported}."
)
ERROR_CORRUPTED_FILE: str = "Failed to read audio file (possibly corrupted): {error}"
ERROR_AUDIO_TOO_SHORT: str = "Audio too short: {duration:.1f}s (min {min_duration}s)"
ERROR_AUDIO_TOO_LONG: str = "Audio too long: {duration:.1f}s (max {max_duration}s)"
ERROR_INVALID_SAMPLE_RATE: str = (
    "Invalid sample rate: {sample_rate}Hz (range {min_rate}-{max_rate}Hz)"
)
ERROR_INVALID_CHANNELS: str = (
    "Invalid channel count: {channels} (must be {min_channels}-{max_channels})"
)
ERROR_METADATA_RETRIEVAL: str = "Failed to retrieve audio metadata: {error}"
AUDIO_VALIDATION_EXCEPTION: str = "Audio validation error: {error}"
UNKNOWN_VALIDATION_EXCEPTION: str = "Unexpected error during validation: {error}"
SUCCESS_VALIDATION: str = "Audio file is valid"

# Audio metadata dictionary keys
METADATA_DURATION: str = "duration_seconds"
METADATA_SAMPLE_RATE: str = "sample_rate"
METADATA_CHANNELS: str = "channels"
METADATA_FILE_SIZE: str = "file_size_mb"
METADATA_FORMAT: str = "audio_format"
