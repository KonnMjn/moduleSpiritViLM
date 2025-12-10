# vi_spiritlm/speech_tokenizer/__init__.py
import os
import torch
from typing import Optional

from .hubert import (
    VietMedHuBERTPredictor,   # backend A: dùng hubert_best.pt (predictor)
    MHuBERTKMeans             # backend B: dùng mHuBERT-147 + KMeans
)
from .tokenizer import UnitsStringCodec

class SpeechSide:
    """
    Giao diện thống nhất cho phần speech:
      - encode_audio_to_units(wav_1xT)
      - units_to_string(units)
      - string_to_units(s)
    """
    def __init__(self, encoder, vocab_size: int = 100):
        self.encoder = encoder
        self.codec   = UnitsStringCodec(vocab_size=vocab_size)

    def encode_audio_to_units(self, wav_1xT: torch.Tensor):
        units = self.encoder.encode(wav_1xT)    # List[int]
        return self.codec.dedup(units)          # SpiritLM-Base: dedup consecutive

    def units_to_string(self, units):
        return self.codec.units_to_string(units)

    def string_to_units(self, s: str):
        return self.codec.string_to_units(s)

def load_speech_side(mode: str = "base", device: Optional[torch.device] = None) -> SpeechSide:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = os.environ.get("VI_SPIRITLM_BACKEND", "predictor").lower()
    vocab_size = 100

    if backend == "predictor":
        hubert_pt = os.environ.get("VI_SPIRITLM_HUBERT_PREDICTOR") or os.environ.get("VI_SPIRITLM_HUBERT_CKPT")
        try:
            encoder = VietMedHuBERTPredictor(hubert_ckpt=hubert_pt, device=device, vocab_size=vocab_size)
            return SpeechSide(encoder, vocab_size=vocab_size)
        except Exception as e:
            print(f"[WARN] predictor backend failed: {e}\n--> Falling back to mHuBERT+KMeans")
            backend = "mhubert_kmeans"

    if backend == "mhubert_kmeans":
        mhubert_id = os.environ.get("VI_SPIRITLM_MHUBERT_ID", "utter-project/mHuBERT-147")
        kmeans     = os.environ.get("VI_SPIRITLM_KMEANS")
        if not kmeans:
            raise RuntimeError("Missing env VI_SPIRITLM_KMEANS for backend=mhubert_kmeans")
        encoder = MHuBERTKMeans(mhubert_id=mhubert_id, kmeans_path=kmeans, device=device)
        return SpeechSide(encoder, vocab_size=vocab_size)

    raise RuntimeError(f"Unknown VI_SPIRITLM_BACKEND='{backend}'")
