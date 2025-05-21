# utils.py

import os
import torch

def save_checkpoint(model, filename):
    """
    儲存模型參數到指定檔案。
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)
    print(f"✅ Model saved to {filename}")

def load_checkpoint(model, filename, device='cpu'):
    """
    從指定檔案載入模型參數。
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ Checkpoint file not found: {filename}")
    model.load_state_dict(torch.load(filename, map_location=device))
    print(f"📦 Model loaded from {filename}")
