"""Output validator for separated audio waveforms - Sprint 1 MVP."""

from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf

from stemmy import constants as c
from stemmy.logging_config import get_logger

logger = get_logger(__name__)


class OutputValidationException(Exception):
    """Exception raised for output validation errors."""

    pass


class OutputValidator:
    """Validates separated audio output before/after export.

    Sprint 1 scope: Basic validation only.
    - Data integrity (NaN/Inf)
    - Shape validation ([2, N] stereo)
    - Silence detection (RMS threshold)
    - File I/O verification
    """

    def __init__(self, waveform: np.ndarray):
        """Initialize validator with a waveform to validate.

        Args:
            waveform: Audio waveform array, expected shape [2, N]
        """
        self.waveform = waveform
        self._validation_result = None
        logger.debug(f"Initialized OutputValidator for waveform shape: {waveform.shape}")

    def validate(self) -> tuple[bool, str]:
        """Validate the waveform and return success status with message.

        Returns:
            Tuple of (success: bool, message: str)
        """
        if self._validation_result is not None:
            logger.debug("Returning cached validation result")
            return self._validation_result

        logger.info("Starting output validation")
        try:
            self._run_all_validations()
            self._validation_result = True, "Output validation successful"
            logger.info("Output validation successful")
        except OutputValidationException as e:
            self._validation_result = (False, str(e))
            logger.warning(f"Output validation failed: {e}")

        return self._validation_result

    def _run_all_validations(self) -> None:
        """Run all validation checks on the waveform."""
        logger.debug("Checking data integrity")
        self._validate_data_integrity()
        logger.debug("Checking waveform shape")
        self._validate_shape()
        logger.debug("Checking for silence")
        self._validate_not_silent()

    def _validate_data_integrity(self) -> None:
        """Check for NaN and Inf values."""
        if np.any(np.isnan(self.waveform)):
            raise OutputValidationException(c.ERROR_WAVEFORM_CONTAINS_NAN)
        if np.any(np.isinf(self.waveform)):
            raise OutputValidationException(c.ERROR_WAVEFORM_CONTAINS_INF)

    def _validate_shape(self) -> None:
        """Check waveform has correct shape [2, N]."""
        if self.waveform.ndim != 2:
            raise OutputValidationException(
                c.ERROR_WAVEFORM_WRONG_SHAPE.format(shape=self.waveform.shape)
            )

        if self.waveform.shape[0] != c.EXPECTED_CHANNELS:
            raise OutputValidationException(
                c.ERROR_CHANNEL_MISMATCH.format(
                    actual=self.waveform.shape[0], expected=c.EXPECTED_CHANNELS
                )
            )

        if self.waveform.shape[1] < c.MIN_SAMPLES:
            raise OutputValidationException(
                c.ERROR_WAVEFORM_TOO_SHORT.format(
                    samples=self.waveform.shape[1], min_samples=c.MIN_SAMPLES
                )
            )

    def _validate_not_silent(self) -> None:
        """Check that output is not silent."""
        rms = np.sqrt(np.mean(self.waveform**2))
        if rms < c.MIN_RMS_LEVEL:
            raise OutputValidationException(c.ERROR_OUTPUT_SILENT.format(rms=rms))

    def is_valid(self) -> bool:
        """Check if the waveform is valid.

        Returns:
            True if valid, False otherwise
        """
        success, _ = self.validate()
        return success

    def get_validation_message(self) -> str:
        """Get the validation result message.

        Returns:
            Validation message
        """
        _, message = self.validate()
        return message

    @staticmethod
    def validate_file(
        file_path: Union[str, Path], expected_sr: int = c.TARGET_SAMPLE_RATE
    ) -> tuple[bool, str]:
        """Validate an exported audio file exists and is readable.

        Args:
            file_path: Path to the exported WAV file
            expected_sr: Expected sample rate

        Returns:
            Tuple of (success: bool, message: str)
        """
        file_path = Path(file_path)
        logger.info(f"Validating exported file: {file_path}")

        if not file_path.exists():
            return False, c.ERROR_FILE_NOT_CREATED.format(path=file_path)

        try:
            info = sf.info(file_path)
        except Exception:
            return False, c.ERROR_OUTPUT_FILE_NOT_READABLE.format(path=file_path)

        if info.samplerate != expected_sr:
            return (
                False,
                c.ERROR_SAMPLE_RATE_MISMATCH.format(actual=info.samplerate, expected=expected_sr),
            )

        if info.channels != c.EXPECTED_CHANNELS:
            return (
                False,
                c.ERROR_CHANNEL_MISMATCH.format(actual=info.channels, expected=c.EXPECTED_CHANNELS),
            )

        logger.info(f"File validation successful: {file_path}")
        return True, "File validation successful"
