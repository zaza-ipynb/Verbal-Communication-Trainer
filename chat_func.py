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

# Process text input with caching and structured feedback
@lru_cache(maxsize=10)
def process_text(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=300)
    response = tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()
    return response



