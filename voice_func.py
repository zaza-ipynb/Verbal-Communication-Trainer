import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gradio as gr
import whisper
from TTS.api import TTS
from functools import lru_cache
import os
import numpy as np
import os
import tempfile
from gtts import gTTS
import yaml
import argparse

# Placeholder LLM function (currently just returns the same text)
def llm_response(transcribed_text, process_text, tokenizer, model):
    return process_text(transcribed_text, tokenizer, model)  # Replace this with actual LLM inference later


# Function to convert text to speech (TTS)
def text_to_speech(text):
    if not text.strip():
        return "Error: No text provided."
    
    tts = gTTS(text=text, lang="en", slow=False)
    output_audio_path = "output_speech.wav"
    tts.save(output_audio_path)

    return output_audio_path

# Function to process audio in chunks
def transcribe(audio, whisper_model, chunk_size=30):
    if not audio or not os.path.exists(audio):
        return "Error: No valid audio file received."

    try:
        # Load and preprocess the audio
        audio_data = whisper.load_audio(audio)
        sample_rate = whisper_model.dims.n_audio_ctx  # Get Whisper's expected sample rate

        # Convert total duration into chunks
        total_duration = len(audio_data) / sample_rate
        num_chunks = int(np.ceil(total_duration / chunk_size))

        results = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_duration)

            # Extract chunk with slight overlap for better context
            chunk_audio = audio_data[int(start * sample_rate):int(end * sample_rate)]
            chunk_audio = whisper.pad_or_trim(chunk_audio)  # Ensure correct input length

            # Convert chunk to Mel spectrogram
            mel = whisper.log_mel_spectrogram(chunk_audio).to(whisper_model.device)

            # Detect language (only on first chunk for efficiency)
            if i == 0:
                _, probs = whisper_model.detect_language(mel)
                detected_language = max(probs, key=probs.get)

            # Transcribe the chunk
            options = whisper.DecodingOptions()
            result = whisper.decode(whisper_model, mel, options)
            results.append(result.text)

        # Combine all transcribed chunks
        full_transcription = " ".join(results)
        return full_transcription

    except Exception as e:
        return f"Error: {str(e)}"
