import torch
from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig
from pathlib import Path

# ================================
# CONFIGURATION
# ================================
INPUT_DIR = Path("./final_model")          # your original model folder
OUTPUT_DIR_FP16 = Path("./final_model_fp16")  # output folder for FP16
OUTPUT_DIR_INT8 = Path("./final_model_int8")  # output folder for INT8

# ================================
# FP16 CONVERSION (CPU-friendly)
# ================================
print("🔹 Loading model in FP16...")
model_fp16 = WhisperForConditionalGeneration.from_pretrained(
    INPUT_DIR.resolve(),           # absolute path ensures no HF repo error
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

print("💾 Saving FP16 model...")
model_fp16.save_pretrained(
    OUTPUT_DIR_FP16.resolve(),
    safe_serialization=True
)

print("✅ FP16 conversion completed!")
print(f"FP16 model size: {sum(f.stat().st_size for f in OUTPUT_DIR_FP16.rglob('*')) / 1024**2:.2f} MB")

# ================================
# INT8 QUANTIZATION (GPU required)
# ================================
try:
    import bitsandbytes as bnb
    print("🔹 Loading model in INT8...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model_int8 = WhisperForConditionalGeneration.from_pretrained(
        INPUT_DIR.resolve(),
        quantization_config=bnb_config,
        device_map="auto"  # automatically moves model to GPU if available
    )

    print("💾 Saving INT8 model...")
    model_int8.save_pretrained(
        OUTPUT_DIR_INT8.resolve(),
        safe_serialization=True
    )

    print("✅ INT8 conversion completed!")
    print(f"INT8 model size: {sum(f.stat().st_size for f in OUTPUT_DIR_INT8.rglob('*')) / 1024**2:.2f} MB")

except ImportError:
    print("⚠️ bitsandbytes not installed. Skipping INT8 conversion.")
    print("Install it with: pip install -U bitsandbytes")
except Exception as e:
    print(f"⚠️ INT8 conversion failed: {e}")
