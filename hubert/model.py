import torch
import torch.nn as nn

class Hubert(nn.Module):
    def __init__(self, model_ckpt_path, device='cpu', num_label_embeddings=None, mask=False):
        super(Hubert, self).__init__()
        # Tải mô hình HuBERT (từ checkpoint)
        self.model = torch.load(model_ckpt_path, map_location=device)
        self.device = device
        
        # Lưu trữ num_label_embeddings và mask nếu chúng được truyền vào
        self.num_label_embeddings = num_label_embeddings
        self.mask = mask  # Lưu trữ tham số mask

    def forward(self, x):
        # Mô hình HuBERT nhận đầu vào và trả về output
        return self.model(x)



# Hàm để nạp HuBERT model
def load_hubert(model_ckpt_path, device='cpu'):
    hubert_model = Hubert(model_ckpt_path, device)
    hubert_model.eval()
    return hubert_model
