import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "")


class SpeechToTextService:
    """Lazy-loaded Azure STT service so import never crashes at startup."""

    def __init__(self):
        self._config = None

    def _get_config(self):
        if self._config is None:
            import azure.cognitiveservices.speech as speechsdk
            if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
                raise ValueError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION not set in .env")
            cfg = speechsdk.SpeechConfig(
                subscription=AZURE_SPEECH_KEY,
                region=AZURE_SPEECH_REGION
            )
            cfg.speech_recognition_language = "en-US"
            self._config = cfg
        return self._config

    def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe an audio file to text using Azure Cognitive Services."""
        if not audio_file_path or not os.path.exists(audio_file_path):
            return "STT Error: Audio file not found."

        try:
            import azure.cognitiveservices.speech as speechsdk
            speech_config = self._get_config()
            
            # Detect format by header if extension is missing/generic
            ext = audio_file_path.lower()
            is_compressed = ext.endswith((".mp3", ".ogg", ".webm", ".m4a", ".aac"))
            
            # Check for common headers if needed
            header = b""
            try:
                with open(audio_file_path, "rb") as f:
                    header = f.read(12)
            except Exception:
                pass

            # Update formats based on signature if extension is misleading
            if header.startswith(b"RIFF") and b"WAVE" in header:
                is_compressed = False
            elif header.startswith(b"\x1a\x45\xdf\xa3"): # WebM/MKV
                is_compressed = True
            elif header.startswith(b"OggS"):
                is_compressed = True

            print(f"DEBUG STT: Processing {audio_file_path} (compressed={is_compressed}, size={os.path.getsize(audio_file_path)})")

            if is_compressed:
                # Determine specific format for Azure
                fmt = speechsdk.audio.AudioStreamContainerFormat.MP3
                if ext.endswith(".ogg") or header.startswith(b"OggS"):
                    print("DEBUG STT: Using OGG_OPUS format")
                    fmt = speechsdk.audio.AudioStreamContainerFormat.OGG_OPUS
                elif ext.endswith((".webm", ".mkv")) or header.startswith(b"\x1a\x45\xdf\xa3"):
                    # NOTE: Azure doesn't officially support WebM container, but many browers' WebM uses Opus
                    # If ffmpeg is missing, this might still fail if not OGG encapsulated.
                    print("DEBUG STT: Detected WebM/Matroska container. Trying OGG_OPUS fallback.")
                    fmt = speechsdk.audio.AudioStreamContainerFormat.OGG_OPUS
                else:
                    print("DEBUG STT: Using MP3/Default compressed format")
                
                stream = speechsdk.audio.PushAudioInputStream(
                    pull_stream_callback=None, 
                    stream_format=speechsdk.audio.AudioStreamFormat.get_compressed_format(fmt)
                )
                with open(audio_file_path, "rb") as f:
                    stream.write(f.read())
                stream.close()
                audio_config = speechsdk.audio.AudioConfig(stream=stream)
            else:
                # Default for WAV
                print("DEBUG STT: Using standard AudioConfig (WAV/PCM)")
                audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)

            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            result = recognizer.recognize_once()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                print(f"DEBUG STT: Successfully recognized: {result.text[:50]}...")
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                print("DEBUG STT: No match found.")
                return "Could not understand audio — please try again."
            elif result.reason == speechsdk.ResultReason.Canceled:
                details = result.cancellation_details
                print(f"DEBUG STT: Canceled: {details.reason} - {details.error_details}")
                if "Format" in str(details.error_details) or "Codec" in str(details.error_details):
                    return "STT Error: Audio format not supported. Please ensure FFmpeg is installed or use a WAV file."
                return f"Speech recognition canceled: {details.reason}"
            return "Speech recognition failed."
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"STT Error: {str(e)}"


stt_service = SpeechToTextService()


def speech_to_text(audio_path: str) -> str:
    return stt_service.transcribe_audio(audio_path)