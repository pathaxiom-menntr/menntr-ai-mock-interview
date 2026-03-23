import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "")


class TextToSpeechService:
    """Lazy-loaded Azure TTS service so import never crashes at startup."""

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
            # Professional female interviewer voice
            cfg.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural"
            self._config = cfg
        return self._config

    def speak_text(self, text: str) -> str | None:
        """Convert text to speech and return the path to the output WAV file."""
        try:
            import azure.cognitiveservices.speech as speechsdk
            speech_config = self._get_config()

            # Write to a temp file so Gradio can serve it
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_file = tmp.name
            tmp.close()

            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            result = synthesizer.speak_text_async(text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return output_file
            return None
        except Exception as e:
            print(f"TTS Error: {e}")
            return None


tts_service = TextToSpeechService()


def text_to_speech(text: str) -> str | None:
    return tts_service.speak_text(text)