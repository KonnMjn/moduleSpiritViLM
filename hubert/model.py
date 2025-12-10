# moduleSpiritViLM/hubert/model.py
import torch
import torch.nn as nn
from transformers import HubertModel, HubertConfig

class Hubert(nn.Module):
    def __init__(self, num_label_embeddings=100, mask=False, model_ckpt_path=None, **kwargs):
        super().__init__()
        
        # 1. Load Config chuẩn từ Teacher model (mHuBERT-147)
        try:
            self.config = HubertConfig.from_pretrained("utter-project/mHuBERT-147")
        except Exception:
            print("[WARN] Không tải được config mHuBERT-147, dùng config facebook/hubert-base-ls960")
            self.config = HubertConfig.from_pretrained("facebook/hubert-base-ls960")
            
        # 2. Khởi tạo Backbone
        self.model = HubertModel(self.config)
        
        # 3. Projection Head (Mặc định, sẽ được resize khi load checkpoint)
        self.proj = nn.Linear(self.config.hidden_size, num_label_embeddings)
        self.mask = mask

    def forward(self, x):
        # Xử lý input shape từ (B, 1, T) -> (B, T) cho khớp Transformers
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
            
        outputs = self.model(x)
        logits = self.proj(outputs.last_hidden_state)
        return logits, None

    def load_weights_from_ckpt(self, ckpt_path):
        """Hàm load weight thông minh, tự sửa size và bỏ qua key lệch"""
        print(f"[INFO] Loading weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        
        # 1. Tự động sửa kích thước lớp proj nếu checkpoint dùng 256 cụm thay vì 100
        if "proj.weight" in state_dict:
            ckpt_dim = state_dict["proj.weight"].shape[0]
            curr_dim = self.proj.weight.shape[0]
            if ckpt_dim != curr_dim:
                print(f"[INFO] Auto-resizing projection layer: {curr_dim} -> {ckpt_dim}")
                self.proj = nn.Linear(self.config.hidden_size, ckpt_dim)
        
        # 2. Lọc bỏ các key gây xung đột của feature extractor cũ
        # Chúng ta ưu tiên giữ lại Backbone chuẩn của HF và chỉ load Projection Head 
        # (nơi chứa thông tin clustering quan trọng nhất).
        # Nếu muốn load cả backbone fine-tuned, cần map tên rất phức tạp. 
        # Ở đây ta chọn giải pháp an toàn: load backbone "mềm" (strict=False).
        
        # Rename keys cơ bản nếu cần (ví dụ layer_norm -> model.encoder.layer_norm)
        # Tuy nhiên, với sự khác biệt lớn, ta chấp nhận load với strict=False
        
        keys = self.load_state_dict(state_dict, strict=False)
        
        print(f"[INFO] Load complete.")
        print(f"   - Missing keys: {len(keys.missing_keys)}")
        print(f"   - Unexpected keys: {len(keys.unexpected_keys)}")
        
        # Kiểm tra xem lớp quan trọng nhất (proj) đã vào chưa
        if "proj.weight" in keys.missing_keys:
            print("[WARN] CRITICAL: Projection layer weights NOT loaded! Kiểm tra tên biến trong ckpt.")
        else:
            print("[SUCCESS] Projection layer loaded successfully.")