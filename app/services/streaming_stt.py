import os
import threading
import queue
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

class StreamingSTTService:
    def __init__(self):
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION")
        
        if not self.speech_key or not self.speech_region:
            print("Azure Speech credentials not found.")
            return

        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key, 
            region=self.speech_region
        )
        self.audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        self.recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, 
            audio_config=self.audio_config
        )
        
        self.phrase_list = speechsdk.PhraseListGrammar.from_recognizer(self.recognizer)
        self._seed_phrase_list()
        
        self.transcript_queue = queue.Queue()
        self.intermediate_queue = queue.Queue()
        self._setup_callbacks()
        
    def _setup_callbacks(self):
        def recognized_cb(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = evt.result.text.strip()
                if text:
                    print(f"RECOGNIZED: {text}")
                    self.transcript_queue.put(text)

        def recognizing_cb(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
                if evt.result.text.strip():
                    self.intermediate_queue.put(evt.result.text)

        self.recognizer.recognized.connect(recognized_cb)
        self.recognizer.recognizing.connect(recognizing_cb)
        self.recognizer.session_stopped.connect(lambda evt: print("Session stopped"))
        self.recognizer.canceled.connect(lambda evt: print(f"Canceled: {evt}"))

    def start(self):
        self.recognizer.start_continuous_recognition_async()

    def stop(self):
        self.recognizer.stop_continuous_recognition_async()

    def push_audio(self, audio_data: bytes):
        """No-op as we use the native system microphone now."""
        pass

    def get_latest_transcript(self):
        """Non-blocking check for new recognized (final) text."""
        try:
            return self.transcript_queue.get_nowait()
        except queue.Empty:
            return None

    def get_intermediate_transcript(self):
        """Non-blocking check for intermediate (partial) text."""
        try:
            latest = None
            while not self.intermediate_queue.empty():
                latest = self.intermediate_queue.get_nowait()
            return latest
        except queue.Empty:
            return None

    def _seed_phrase_list(self):
        """Pre-populate common technical terms to improve STT accuracy."""
        tech_terms = [
            "FastAPI", "Python", "Azure", "OpenAI", "LLM", "GPT", "LangGraph", 
            "Docker", "Kubernetes", "SQL", "NoSQL", "REST API", "GraphQL", 
            "React", "Node.js", "JavaScript", "TypeScript", "Pydantic", "Uvicorn",
            "Purely", "Inheritance", "Polymorphism", "Encapsulation", "Abstraction",
            "Microservices", "Serverless", "Database", "Asynchronous", "Concurrency"
        ]
        for term in tech_terms:
            self.phrase_list.addPhrase(term)

    def add_custom_phrases(self, phrases: list):
        """Dynamically add interview-specific phrases (e.g. skills from resume)."""
        if not phrases: return
        for p in phrases:
            self.phrase_list.addPhrase(str(p))

streaming_stt_service = StreamingSTTService()
