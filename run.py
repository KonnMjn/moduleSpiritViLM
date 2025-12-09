
# moduleforSpiritViLM_gpt/run.py
import os
import torch
from pathlib import Path

import wandb  # ✅ thêm WandB

from .model.model import ViSpiritLM
from .model.utils import convert_to_wav_tensor, build_interleaved_prompt


# =========================
# 0) Hàm tải checkpoints từ WandB
# =========================
def download_artifacts_from_wandb():
    """
    Tải 3 thứ:
      - HuBERT predictor (.pt)
      - KMeans (.joblib)
      - LoRA directory (adapter + tokenizer)

    Các artifact dùng đúng string bạn cung cấp:
      - HuBERT:  spirit-vilm/uncategorized/vimd_100:v0
      - KMeans:  spirit-vilm/fine-tune-hubert-vietnamese-vimd/kmeans_model:v18
      - LoRA:    spirit-vilm/finetune-Vistral-allModaVietMednViMD-v2/vistral_finetuned_0.33:v3
    """
    print("[INFO] Đang login WandB...")
    # Dùng WANDB_API_KEY trong env, hoặc config local
    wandb.login()

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "spirit-vilm-local"),
        job_type="download_models_for_inference",
    )

    # 1) HuBERT predictor
    print("[INFO] Tải HuBERT từ artifact spirit-vilm/uncategorized/vimd_100:v0 ...")
    artifact_hubert = run.use_artifact(
        "spirit-vilm/uncategorized/vimd_100:v0", type="model"
    )
    hubert_dir = artifact_hubert.download()
    hubert_path = next(Path(hubert_dir).glob("*.pt"))  # lấy file .pt đầu tiên

    # 2) KMeans
    print("[INFO] Tải KMeans từ artifact spirit-vilm/fine-tune-hubert-vietnamese-vimd/kmeans_model:v18 ...")
    artifact_kmeans = run.use_artifact(
        "spirit-vilm/fine-tune-hubert-vietnamese-vimd/kmeans_model:v18",
        type="model",
    )
    kmeans_dir = artifact_kmeans.download()
    kmeans_path = next(Path(kmeans_dir).glob("*.joblib"))  # lấy file .joblib đầu tiên

    # 3) LoRA
    print("[INFO] Tải LoRA từ artifact spirit-vilm/finetune-Vistral-allModaVietMednViMD-v2/vistral_finetuned_0.33:v3 ...")
    artifact_lora = run.use_artifact(
        "spirit-vilm/finetune-Vistral-allModaVietMednViMD-v2/vistral_finetuned_0.33:v3",
        type="model",
    )
    lora_dir = Path(artifact_lora.download())

    run.finish()

    print(f"[INFO] HuBERT ckpt: {hubert_path}")
    print(f"[INFO] KMeans path: {kmeans_path}")
    print(f"[INFO] LoRA dir:    {lora_dir}")

    return str(hubert_path), str(kmeans_path), str(lora_dir)


# =========================
# 1) Thiết lập môi trường
# =========================

# Backend HuBERT: sẽ override bằng path tải từ WandB
os.environ.setdefault("VI_SPIRITLM_BACKEND", "predictor")

# Demo audio (tùy chọn)
DEMO_WAV = os.getenv("DEMO_WAV", "/workspace/examples/00000.flac")

# BASE_LM: có thể là thư mục local hoặc repo_id HF
BASE_LM = os.getenv("BASE_LM", "Viet-Mistral/Vistral-7B-Chat")

# --- Tải artifact từ WandB ---
hubert_ckpt_path, kmeans_path, lora_dir_downloaded = download_artifacts_from_wandb()

# Gán vào ENV để các phần khác (HuBERT backend) dùng được
os.environ["VI_SPIRITLM_HUBERT_PREDICTOR"] = hubert_ckpt_path
os.environ["VI_SPIRITLM_KMEANS"] = kmeans_path

# LORA_DIR: ưu tiên ENV nếu user override, ngược lại dùng dir tải từ WandB
LORA_DIR = os.getenv("LORA_DIR", lora_dir_downloaded)

# Một số check nhẹ cho dễ debug
if not (BASE_LM.startswith("Viet-") or Path(BASE_LM).exists()):
    print(
        f"[WARN] BASE_LM='{BASE_LM}' không phải thư mục local; sẽ load từ Hugging Face Hub (cần internet)."
    )

if not Path(LORA_DIR).exists():
    print(
        f"[WARN] LORA_DIR='{LORA_DIR}' chưa tồn tại. Kiểm tra lại artifact LoRA hoặc biến môi trường LORA_DIR."
    )

pred_ckpt = os.getenv("VI_SPIRITLM_HUBERT_PREDICTOR")
km_env_path = os.getenv("VI_SPIRITLM_KMEANS")
for p, name in [(pred_ckpt, "VI_SPIRITLM_HUBERT_PREDICTOR"), (km_env_path, "VI_SPIRITLM_KMEANS")]:
    if not p or not Path(p).exists():
        print(f"[WARN] {name}='{p}' chưa tồn tại. Hãy kiểm tra lại bước tải artifact HuBERT/KMeans.")

# =========================
# 2) Khởi tạo mô hình
# =========================
use_cuda = torch.cuda.is_available()
dtype = torch.float16 if use_cuda else torch.float32
device = "cuda" if use_cuda else "cpu"

print(f"[INFO] device={device}, dtype={dtype}")
print(f"[INFO] BASE_LM={BASE_LM}")
print(f"[INFO] LORA_DIR={LORA_DIR}")
print(f"[INFO] HuBERT ckpt={pred_ckpt}")
print(f"[INFO] KMeans path={km_env_path}")

import sys
sys.path.insert(0, '/workspace/moduleforSpiritViLM_gpt/speech_tokenizer')  # hoặc đường dẫn đúng tới speech_tokenizer

model = ViSpiritLM(BASE_LM, LORA_DIR, device=device, torch_dtype=dtype)

# =========================
# 3) Encode audio -> units
# =========================
if Path(DEMO_WAV).exists():
    wav = convert_to_wav_tensor(DEMO_WAV, device=torch.device(device))
    units = model.encode_audio_to_units(wav)
    unit_str = model.units_to_string(units)
else:
    print(f"[WARN] DEMO_WAV='{DEMO_WAV}' không tồn tại. Sẽ demo text-only.")
    unit_str = ""

# =========================
# 4) Generate (text-first)
# =========================
chunks = [("text", "Xin chào, đây là ví dụ.")]
if unit_str:
    chunks += [("speech", unit_str)]
chunks += [("text", "Hãy tiếp tục trả lời bằng văn bản: ")]

prompt = build_interleaved_prompt(chunks)

text_out = model.generate_text(
    prompt,
    max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "64")),
    temperature=float(os.getenv("TEMP", "0.8")),
    use_cache=os.getenv("USE_CACHE", "0") == "1",
)
print("TEXT OUT:", text_out)

# # 4b) Sinh SPEECH-UNITS (chưa có vocoder)
# prompt2 = build_interleaved_prompt([("text", "Hãy phát âm câu sau: Xin chào Việt Nam!")])
# speech_units = model.generate_speech_units(prompt2, max_new_tokens=300, temperature=0.9, use_cache=False)
# print("SPEECH UNITS LEN:", len(speech_units))