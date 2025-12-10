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
    def __init__(self, hubert_ckpt: str, device: torch.device, vocab_size: int = 100):
        self.device = device
        
        # Import class Hubert
        import sys
        sys.path.insert(0, '/workspace/moduleSpiritViLM/hubert')
        try:
            from model import Hubert
        except ImportError:
            from ..hubert.model import Hubert

        # Khởi tạo model (chưa load weight)
        self.model = Hubert(num_label_embeddings=vocab_size)
        
        # Gọi hàm load thông minh mới viết
        self.model.load_weights_from_ckpt(hubert_ckpt)
        
        self.model.eval().to(device)
        
        # Cập nhật vocab_size thực tế từ model (đề phòng resize lên 256)
        self.vocab_size = self.model.proj.out_features
        print(f"[INFO] Final vocab size: {self.vocab_size}")

    @torch.inference_mode()
    def encode(self, wav_1xT: torch.Tensor) -> List[int]:
        if wav_1xT.dim() != 2 or wav_1xT.size(0) != 1:
            raise ValueError("Expect wav_1xT shape (1, T)")
            
        x = wav_1xT.unsqueeze(0).to(self.device) # (1, 1, T)
        
        out = self.model(x)
        # out có thể là (logits, mask) hoặc logits
        logits = out[0] if isinstance(out, tuple) else out
            
        codes = torch.argmax(logits, dim=-1)
        return codes.squeeze(0).detach().cpu().tolist()

# ... (Giữ nguyên phần còn lại của file: MHuBERTKMeans, v.v.) ...


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
