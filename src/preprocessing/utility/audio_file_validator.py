"""Class-based audio file validator."""

import os

from src.logging_config import get_logger
from src.preprocessing.utility import constants
from src.preprocessing.utility.audio_metadata_extractor import AudioMetadataExtractor

logger = get_logger(__name__)


class AudioValidationException(Exception):
    """Exception raised for audio file validation errors."""

    pass


class AudioFileValidator:
    """Validates audio files for processing."""

    def __init__(self, file_path: str):
        """Initialize the validator with an audio file path.

        Args:
            file_path: Path to the audio file to validate
        """
        self.file_path = file_path
        self.metadata_extractor = AudioMetadataExtractor(file_path)
        self._metadata = None
        self._validation_result = None
        logger.debug(f"Initialized AudioFileValidator for file: {file_path}")

    def validate(self) -> tuple[bool, str]:
        """Validate the audio file and return success status with message.

        Returns:
            Tuple of (success: bool, message: str)
        """
        if self._validation_result is not None:
            logger.debug(f"Returning cached validation result for: {self.file_path}")
            return self._validation_result

        logger.info(f"Starting validation for: {self.file_path}")
        try:
            self._run_all_validations()
            self._validation_result = (True, constants.SUCCESS_VALIDATION)
            logger.info(f"Validation successful for: {self.file_path}")
        except AudioValidationException as e:
            error_msg = constants.AUDIO_VALIDATION_EXCEPTION.format(error=str(e))
            self._validation_result = (False, error_msg)
            logger.warning(f"Validation failed for {self.file_path}: {str(e)}")
        except Exception as e:
            error_msg = constants.UNKNOWN_VALIDATION_EXCEPTION.format(error=str(e))
            self._validation_result = (False, error_msg)
            logger.error(f"Unexpected error during validation of {self.file_path}: {str(e)}")

        return self._validation_result

    def _run_all_validations(self) -> None:
        """Run all validation checks on the audio file."""
        logger.debug("Running file readability check")
        self._validate_readable_file()
        logger.debug("Loading metadata")
        self._load_metadata()
        logger.debug("Validating file size")
        self._validate_file_size()
        logger.debug("Validating file format")
        self._validate_file_format()
        logger.debug("Validating audio properties")
        self._validate_audio_properties()

    def _load_metadata(self) -> None:
        """Load and cache metadata from the audio file."""
        try:
            self._metadata = self.metadata_extractor.get_all_metadata()
        except Exception as e:
            raise AudioValidationException(constants.ERROR_METADATA_RETRIEVAL.format(error=str(e)))

    def _validate_readable_file(self) -> None:
        """Check if file exists and is readable."""
        if not os.path.isfile(self.file_path):
            raise AudioValidationException(constants.ERROR_FILE_NOT_EXIST)
        if not os.access(self.file_path, os.R_OK):
            raise AudioValidationException(constants.ERROR_FILE_NOT_READABLE)

    def _validate_file_size(self) -> None:
        """Check if file size is within acceptable limits."""
        file_size = self._metadata.get(constants.METADATA_FILE_SIZE)
        if file_size > constants.MAX_FILE_SIZE_MB:
            raise AudioValidationException(
                constants.ERROR_FILE_TOO_LARGE.format(max_size=constants.MAX_FILE_SIZE_MB)
            )

    def _validate_file_format(self) -> None:
        """Check if file format is supported."""
        file_format = self._metadata.get(constants.METADATA_FORMAT)
        if file_format not in constants.SUPPORTED_FORMATS:
            raise AudioValidationException(
                constants.ERROR_UNSUPPORTED_FORMAT.format(
                    format=file_format, supported=", ".join(constants.SUPPORTED_FORMATS)
                )
            )

    def _validate_audio_properties(self) -> None:
        """Validate audio duration, sample rate, and channel count."""
        self._validate_duration()
        self._validate_sample_rate()
        self._validate_channels()

    def _validate_duration(self) -> None:
        """Validate audio duration is within acceptable range."""
        duration = self._metadata.get(constants.METADATA_DURATION)

        if duration < constants.MIN_DURATION_SECONDS:
            raise AudioValidationException(
                constants.ERROR_AUDIO_TOO_SHORT.format(
                    duration=duration, min_duration=constants.MIN_DURATION_SECONDS
                )
            )

        if duration > constants.MAX_DURATION_SECONDS:
            raise AudioValidationException(
                constants.ERROR_AUDIO_TOO_LONG.format(
                    duration=duration, max_duration=constants.MAX_DURATION_SECONDS
                )
            )

    def _validate_sample_rate(self) -> None:
        """Validate sample rate is within acceptable range."""
        sample_rate = self._metadata.get(constants.METADATA_SAMPLE_RATE)

        if sample_rate < constants.MIN_SAMPLE_RATE or sample_rate > constants.MAX_SAMPLE_RATE:
            raise AudioValidationException(
                constants.ERROR_INVALID_SAMPLE_RATE.format(
                    sample_rate=sample_rate,
                    min_rate=constants.MIN_SAMPLE_RATE,
                    max_rate=constants.MAX_SAMPLE_RATE,
                )
            )

    def _validate_channels(self) -> None:
        """Validate channel count is within acceptable range."""
        num_channels = self._metadata.get(constants.METADATA_CHANNELS)

        if num_channels < constants.MIN_CHANNELS or num_channels > constants.MAX_CHANNELS:
            raise AudioValidationException(
                constants.ERROR_INVALID_CHANNELS.format(
                    channels=num_channels,
                    min_channels=constants.MIN_CHANNELS,
                    max_channels=constants.MAX_CHANNELS,
                )
            )

    def get_metadata(self) -> dict:
        """Get the metadata of the validated audio file.

        Returns:
            Dictionary containing audio metadata

        Note:
            Metadata is loaded during validation. If validation hasn't run,
            this will trigger metadata loading.
        """
        if self._metadata is None:
            self._load_metadata()
        return self._metadata

    def is_valid(self) -> bool:
        """Check if the audio file is valid.

        Returns:
            True if valid, False otherwise
        """
        success, _ = self.validate()
        return success

    def get_validation_message(self) -> str:
        """Get the validation result message.

        Returns:
            Validation message (success or error description)
        """
        _, message = self.validate()
        return message

    def reset_validation(self) -> None:
        """Reset validation state to allow re-validation."""
        logger.debug(f"Resetting validation state for: {self.file_path}")
        self._validation_result = None
        self._metadata = None
        self.metadata_extractor.clear_cache()
