"""Class-based audio file metadata extractor using librosa and soundfile."""

import os

import librosa
import soundfile as sf

from src.logging_config import get_logger
from src.preprocessing.utility import constants

logger = get_logger(__name__)


class AudioMetadataExtractor:
    """Extracts and manages metadata for audio files."""

    def __init__(self, file_path: str):
        """Initialize the extractor with an audio file path.

        Args:
            file_path: Path to the audio file
        """
        self.file_path = file_path
        self._metadata_cache = None
        logger.debug(f"Initialized AudioMetadataExtractor for file: {file_path}")

    def get_all_metadata(self) -> dict:
        """Get all metadata of the audio file.

        Returns:
            Dictionary containing all audio metadata
        """
        if self._metadata_cache is None:
            logger.info(f"Extracting metadata from: {self.file_path}")
            self._metadata_cache = {
                constants.METADATA_DURATION: self.get_duration(),
                constants.METADATA_SAMPLE_RATE: self.get_sample_rate(),
                constants.METADATA_CHANNELS: self.get_channels(),
                constants.METADATA_FILE_SIZE: self.get_file_size_mb(),
                constants.METADATA_FORMAT: self.get_format(),
            }
            logger.debug(f"Metadata extracted: {self._metadata_cache}")
        else:
            logger.debug("Returning cached metadata")
        return self._metadata_cache

    def get_duration(self) -> float:
        """Get the duration of the audio file in seconds.

        Returns:
            Duration in seconds
        """
        return librosa.get_duration(path=self.file_path)

    def get_sample_rate(self) -> int:
        """Get the sample rate of the audio file.

        Returns:
            Sample rate in Hz
        """
        info = sf.info(self.file_path)
        return info.samplerate

    def get_channels(self) -> int:
        """Get the number of channels in the audio file.

        Returns:
            Number of channels (1 for mono, 2 for stereo, etc.)
        """
        info = sf.info(self.file_path)
        return info.channels

    def get_file_size_mb(self) -> float:
        """Get the file size in megabytes.

        Returns:
            File size in MB
        """
        size_bytes = os.path.getsize(self.file_path)
        return size_bytes / constants.BYTES_TO_MB

    def get_format(self) -> str:
        """Get the file format/extension (without the dot).

        Returns:
            File format as lowercase string
        """
        return os.path.splitext(self.file_path)[1].lstrip(".").lower()

    def clear_cache(self) -> None:
        """Clear the cached metadata."""
        logger.debug(f"Clearing metadata cache for: {self.file_path}")
        self._metadata_cache = None
