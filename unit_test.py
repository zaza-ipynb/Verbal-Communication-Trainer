import unittest
import os
from no_gradio import process_text, transcribe, text_to_speech, model, tokenizer
from unittest.mock import MagicMock

class TestVerbalTrainer(unittest.TestCase):

    def test_process_text(self):
        """Test LLM text processing function"""
        response = process_text("Hello!", tokenizer, model)
        self.assertIsInstance(response, str, "LLM response is not a string")
        self.assertGreater(len(response), 0, "LLM response is empty")

    def test_transcribe(self):
        """Test speech-to-text transcription"""
        test_audio = "test_1.wav"  # Ensure this file exists in the project directory
        transcription = transcribe(test_audio, None)  # Pass a real whisper model if needed
        
        self.assertIsInstance(transcription, str, "Transcription output is not a string")
        self.assertGreater(len(transcription), 0, "Transcription is empty")

    def test_text_to_speech(self):
        """Test text-to-speech conversion"""
        output_path = text_to_speech("Hello, this is a test!")
        
        self.assertIsInstance(output_path, str, "TTS did not return a file path")
        self.assertTrue(os.path.exists(output_path), "Generated speech file not found!")

if __name__ == "__main__":
    unittest.main()
