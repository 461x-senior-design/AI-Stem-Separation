from src.logging_config import setup_logging
from src.preprocessing.utility.audio_file_validator import AudioFileValidator
from src.preprocessing.utility.audio_metadata_extractor import AudioMetadataExtractor

setup_logging(level="DEBUG")

# TODO: Remove this test code once integration is complete

# Example audio file paths (replace with actual test files as needed)
# Store in /local directory for testing
mp3_path = "test.mp3"
wav_path = "test.wav"
txt_path = "test.txt"

print("=== AudioFileValidator Examples ===")
mp3_validator = AudioFileValidator(mp3_path)
wav_validator = AudioFileValidator(wav_path)
txt_validator = AudioFileValidator(txt_path)

print(f"MP3 validation: {mp3_validator.validate()}")
print(f"WAV validation: {wav_validator.validate()}")
print(f"TXT validation: {txt_validator.validate()}")

print("\n=== AudioMetadataExtractor Examples ===")
mp3_metadata = AudioMetadataExtractor(mp3_path)
wav_metadata = AudioMetadataExtractor(wav_path)

print(f"\nMP3 all metadata: {mp3_metadata.get_all_metadata()}")
print(f"MP3 duration: {mp3_metadata.get_duration()} seconds")
print(f"MP3 sample rate: {mp3_metadata.get_sample_rate()} Hz")
print(f"MP3 channels: {mp3_metadata.get_channels()}")
print(f"MP3 file size: {mp3_metadata.get_file_size_mb()} MB")
print(f"MP3 format: {mp3_metadata.get_format()}")

print(f"\nWAV all metadata: {wav_metadata.get_all_metadata()}")
print(f"WAV duration: {wav_metadata.get_duration()} seconds")
print(f"WAV sample rate: {wav_metadata.get_sample_rate()} Hz")
