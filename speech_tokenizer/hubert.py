# vi_spiritlm/speech_tokenizer/hubert.py
"""
Hai backend rời rạc hóa audio:
A) VietMedHuBERTPredictor: dùng 'hubert.model.Hubert' + hubert_best.pt => logits per frame -> argmax (0..99)
B) MHuBERTKMeans: dùng HF 'utter-project/mHuBERT-147' + KMeans(100) => predict per frame -> (0..99)

Mặc định dùng A) vì bạn đã train hubert_best.pt theo script train-hubert-vietnamese-v4.py.
"""

from typing import List, Optional
import torch
import numpy as np
import joblib

# =============== Backend A: Predictor ==========================
class VietMedHuBERTPredictor:
    """
    Load class 'Hubert' từ gói local 'hubert.model' và nạp state_dict hubert_best.pt.
    Forward: (B, 1, T) -> logits(B, T', Vocab), (mask) | có thể trả về tuple, ta chỉ cần logits.
    """
    def __init__(self, hubert_ckpt: str, device: torch.device, vocab_size: int = 100):
        self.device = device
        self.vocab_size = vocab_size
        self.model = self._build_model(vocab_size=vocab_size)
        self._load_state(self.model, hubert_ckpt)
        self.model.eval().to(device)

    def _build_model(self, vocab_size: int) -> torch.nn.Module:
        """
        Script train đã dùng: from hubert.model import Hubert
        Ta yêu cầu gói 'hubert' có sẵn trong PYTHONPATH khi chạy inference.
        """
        try:
            from hubert.model import Hubert
        except Exception as e:
            raise RuntimeError(
                "Không import được 'hubert.model.Hubert'. "
                "Hãy đảm bảo package 'hubert' (thư mục chứa model) có trong PYTHONPATH.\n"
                f"Import error: {e}"
            )
        # Theo train script: Hubert(num_label_embeddings=num_clusters, mask=True)
        model = Hubert(num_label_embeddings=vocab_size, mask=True)
        return model

    def _load_state(self, model: torch.nn.Module, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")
        # Lưu ý: train script dùng torch.save(hubert.state_dict(), ...),
        # nên ở đây load_state_dict(state_dict) là đúng.
        model.load_state_dict(sd, strict=True)

    @torch.inference_mode()
    def encode(self, wav_1xT: torch.Tensor) -> List[int]:
        """
        wav_1xT: (1, T) float32 @ 16k
        return: List[int] length ~ T_frames (0..99)
        """
        if wav_1xT.dim() != 2 or wav_1xT.size(0) != 1:
            raise ValueError("Expect wav_1xT shape (1, T)")
        x = wav_1xT.unsqueeze(0)  # (B=1, 1, T)
        x = x.to(self.device)
        out = self.model(x)
        # train script: logits, mask_out = hubert(wavs)
        if isinstance(out, (list, tuple)):
            logits = out[0]
        else:
            logits = out
        # logits: (B, T', V)
        if logits.dim() != 3 or logits.size(0) != 1 or logits.size(-1) != self.vocab_size:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
        codes = torch.argmax(logits, dim=-1)  # (1, T')
        return codes.squeeze(0).detach().cpu().tolist()


# =============== Backend B: mHuBERT + KMeans ====================
class MHuBERTKMeans:
    """
    Dùng HF mHuBERT-147 để trích 'last_hidden_state' -> KMeans predict -> mã 0..99.
    Phù hợp với Stage 1 trong script train-hubert-vietnamese-v4.py.
    """
    def __init__(self, mhubert_id: str, kmeans_path: str, device: torch.device):
        from transformers import Wav2Vec2FeatureExtractor, HubertModel
        self.device = device
        self.model  = HubertModel.from_pretrained(mhubert_id).to(device).eval()
        self.feat_ex = Wav2Vec2FeatureExtractor.from_pretrained(mhubert_id)
        self.kmeans = joblib.load(kmeans_path)
        assert getattr(self.kmeans, "n_clusters", 100) == 100, \
            f"Expect KMeans(100), got n_clusters={getattr(self.kmeans, 'n_clusters', None)}"

    @torch.inference_mode()
    def encode(self, wav_1xT: torch.Tensor) -> List[int]:
        if wav_1xT.dim() != 2 or wav_1xT.size(0) != 1:
            raise ValueError("Expect wav_1xT shape (1, T)")
        wav = wav_1xT.squeeze(0).cpu().numpy()
        inputs = self.feat_ex(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        outputs = self.model(**inputs.to(self.device))
        feats = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()  # (T', D)
        codes = self.kmeans.predict(feats).astype(np.int64).tolist()
        return codes
