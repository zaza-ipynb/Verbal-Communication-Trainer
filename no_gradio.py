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


from chat_func import process_text
from voice_func import llm_response, text_to_speech, transcribe
from module_train import start_training, impromptu_speaking, storytelling, conflict_resolution

# Load YAML configuration
def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)
    
# # Parse command-line arguments
# parser = argparse.ArgumentParser(description="Run the Verbal Communication Trainer UI")
# parser.add_argument("--model", type=int, required=True, help="Choose model ID as defined in the config.yaml")
# args = parser.parse_args()

# Load config
config = load_config("config.yaml")


# Load Open LLM Model with Optimizations
def load_model(config):
    model_name = config.get("model_mapping", {}).get(1)  # Updated to Falcon-7B-Instruct
    print(f"Model: {model_name}")
    # Apply 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    return model, tokenizer

# Load model once and reuse
model, tokenizer = load_model(config)

# Load Whisper for speech recognition
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)

# Load Mozilla TTS for text-to-speech
tts = TTS("tts_models/en/ljspeech/glow-tts")

# Store conversation history
voice_history = []


# function for chat interface to have general conversation with coach AI
chat_histories = []  # Global chat history

def chat_interface(user_input):
    global chat_histories

    response = process_text(user_input, tokenizer,model)
    # Append to chat history
    chat_histories.append(("User", user_input))
    chat_histories.append(("AI", response))

    # Format for display
    formatted_chat = ""
    for speaker, message in chat_histories:
        formatted_chat += f"**{speaker}:** {message}\n\n"

    return formatted_chat

# function for audio interface to have general audio conversation with coach AI
def process_audio(audio):
    transcribed_text = transcribe(audio, whisper_model)  # Step 1: Speech to Text
    
    if transcribed_text.startswith("Error"):
        return transcribed_text, "Error", None  # No LLM or TTS if STT failed
    
    llm_result = llm_response(transcribed_text,  process_text, tokenizer, model)  # Step 2: LLM response (placeholder)
    
    speech_output = text_to_speech(llm_result)  # Step 3: Convert LLM response to speech
    print('successfully run process audio')
    return transcribed_text, llm_result, speech_output  # Return all outputs

def evaluate_training_response(user_response, module_type):
    """Process user response and return AI feedback."""
    prompts = {
        "impromptu": config.get("prompt_module", {}).get('impromptu'),
        "storytelling": config.get("prompt_module", {}).get('storytelling'),
        "conflict_resolution": config.get("prompt_module", {}).get('conflict_resolution')
    }
    
    evaluation_prompt = prompts.get(module_type, "Analyze this response.")
    feedback = process_text(f"{evaluation_prompt}: \"{user_response}\"", tokenizer, model)
    return feedback

# transcribed_text, llm_response, speech_output = process_audio("test_1.wav")
# # Gradio UI
# with gr.Blocks() as demo:
#     gr.Markdown("# CoachAI")
    
#     with gr.Tab("Chat"):
#         gr.Markdown("# Chat with CoachAI")
#         chat_output = gr.Markdown(value="", label="Conversation History")  # Display chat history
#         chat_input = gr.Textbox(label="Enter your message")
#         chat_button = gr.Button("Send")

#         chat_button.click(chat_interface, inputs=chat_input, outputs=chat_output)

#     with gr.Tab("Voice"):
#         gr.Interface(
#             title="Speak with CoachAI",
#             fn=process_audio,  
#             inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
#             outputs=[
#                 gr.Textbox(label="Transcribed Text"),  # Shows STT output
#                 gr.Textbox(label="LLM Response"),  # Shows LLM response
#                 gr.Audio(type="filepath", label="Generated Speech")  # Play TTS audio
#             ],
#             live=False,
#             allow_flagging="never"
#         )
#     with gr.Tab("Training"):
#         gr.Markdown("### Training Modules")
#         training_selector = gr.Radio(["Impromptu Speaking", "Storytelling", "Conflict Resolution"], label="Choose a training mode")
#         training_prompt = gr.Textbox(label="Training Task")
#         training_button = gr.Button("Start Training")
#         training_button.click(start_training, inputs=training_selector, outputs=training_prompt)

#         training_input = gr.Textbox(label="Your Response")
#         training_feedback = gr.Textbox(label="AI Feedback")
#         training_evaluate_button = gr.Button("Get Feedback")
#         training_evaluate_button.click(evaluate_training_response, inputs=[training_input, training_selector], outputs=training_feedback)


# # Increase timeout for Gradio requests
# # demo.queue().launch(share=True, max_threads=2)
# demo.queue().launch(max_threads=2)

