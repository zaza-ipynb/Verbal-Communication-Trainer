# Verbal-Communication-Trainer

# Setting Up the Environment 🚀

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
If running `llm.py` and the model is not yet downloaded locally, follow these steps:
1. Sign up on [Hugging Face](https://huggingface.co/).
2. Go to **Settings > Access Tokens**.
3. Click **New Token**, name it (e.g., "mistral-access"), and select **Read** permissions.
4. Click **Generate Token** and copy the token.

Now you are all set to run the project! 🚀

