# main.py
from fastapi import FastAPI, UploadFile, File
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig
import torch
import tempfile
import os
import torchaudio

# ==========================
# APP INIT
# ==========================
app = FastAPI(title="Lexi Whisper INT8 API")

# ==========================
# CONFIG: Load model from Hugging Face
# ==========================
MODEL_HF_REPO = "sesefi/lexi-reading-guide"

# Use INT8 quantization to reduce memory usage
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

print("Loading model in INT8 (CPU)...")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_HF_REPO,
    quantization_config=bnb_config
)

# Move model to CPU explicitly (Render free tier is CPU only)
device = torch.device("cpu")
model.to(device)

processor = WhisperProcessor.from_pretrained(MODEL_HF_REPO)

print("✅ Model loaded successfully!")

# ==========================
# ROUTES
# ==========================
@app.get("/")
async def root():
    return {"message": "Lexi Whisper API is running!"}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Accepts audio file upload and returns transcription.
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        audio_path = tmp.name

    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Process audio
        inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")

        # Move inputs to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(**inputs)

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    finally:
        # Cleanup temp file
        os.remove(audio_path)

    return {"transcription": transcription}
