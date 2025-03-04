# Verbal-Communication-Trainer - Setting Up the Environment 🚀

## Overview
This project is a wrapper application around an open-source LLM to help learners improve their verbal communication skills. It supports text and voice interactions, provides training activities, and assesses presentations with feedback.

## Model Selection
The models used in this project are:
- **DeepSeek-R1-Distill-Qwen-7B**
- **Falcon-7B-Instruct**

### Why 7B Models Instead of 13B?
1. **Hardware Efficiency**: 13B models require high-end GPUs (24GB+ VRAM), making them difficult to run on most machines. The selected 7B models are optimized to work within reasonable hardware constraints.
2. **Balanced Performance**: These models provide strong conversational ability while being more resource-efficient.
3. **Faster Inference**: Smaller models ensure lower latency, which is crucial for real-time chat and voice interactions.
4. **Flexible Deployment**: They can run on both consumer GPUs and cloud environments with optimizations like 8-bit quantization.

## Features
- **Chat Interface**: A conversational coach that provides feedback on clarity, tone, and improvement suggestions.
- **Voice Interface**: Uses Whisper for speech-to-text and Mozilla TTS for text-to-speech.
- **Training Modules**:
  - Impromptu Speaking
  - Storytelling
  - Conflict Resolution
- **Presentation Assessment**: Analyzes structure, delivery, and content with detailed feedback.

## Deployment & Optimization
- Uses **Hugging Face Transformers** for model inference.
- Supports **quantization** to reduce memory usage.
- Implements **caching** for faster response times.
- Designed for **scalability** with API and UI options.



## Prerequisites
Before running the project, ensure you have the following installed:

### Required Software
- **Python 3.9**
- **pip 25.0.1**

### Additional Requirements for Windows Users
- **Microsoft Visual C++ 14.0 or greater** is required.
- **FFmpeg** must be installed before running the application.

## Installing Dependencies
Run the following commands to install the necessary packages:

```sh
pip install transformers
pip install gradio
pip install openai-whisper
pip install TTS
pip install torch transformers gradio openai-whisper TTS accelerate>=0.26.0
```
or 
 ```bash
   pip install -r requirements.txt
 ```

## GPU Support
To run on a **GPU**, install CUDA by running:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Running on a High-End CPU
If you have a high-end CPU, you can also run the model using:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", device_map="cpu")
```

## Additional Installations
```sh
pip install bitsandbytes
pip install gTTS==2.2.1
```

## Hugging Face Model Access

Run the application:
   ```bash
   python verbal_trainer.py --model 1  # Choose 1 for DeepSeek or 2 for Falcon
   ```
If running `verbal_trainer.py` and the model is not yet downloaded locally, follow these steps:
1. Sign up on [Hugging Face](https://huggingface.co/).
2. Go to **Settings > Access Tokens**.
3. Click **New Token**, name it (e.g., "mistral-access"), and select **Read** permissions.
4. Click **Generate Token** and copy the token.

we also had done checking the core function/logic which shows everything is good to use
``` 
----------------------------------------------------------------------
Ran 3 tests in 25.479s

OK
```


Now you are all set to run the project! 🚀

