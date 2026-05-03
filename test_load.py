import mlx.nn as nn
from mlx_lm import load

original_load_weights = nn.Module.load_weights

def custom_load_weights(self, weights, strict=True):
    # 강제로 strict=False 로 로드
    return original_load_weights(self, weights, strict=False)

nn.Module.load_weights = custom_load_weights

try:
    model, tokenizer = load("Jiunsong/supergemma4-e4b-abliterated-mlx")
    print("Successfully loaded!")
except Exception as e:
    print("Error:", e)
