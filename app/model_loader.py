import os
import torch
from models.unet import UNet
from models.deeplab import DeepLab

# This file is ROOT/app/model_loader.py
# So ROOT is two levels up
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_models_cache = {}

def load_model(model_name):

    if model_name in _models_cache:
        return _models_cache[model_name]

    if model_name == "unet":
        path = os.path.join(ROOT_DIR, "models", "unet.pth")
        checkpoint = torch.load(path, map_location=DEVICE)

        model = UNet(in_channels=6, out_channels=2)

    elif model_name == "deeplab":
        path = os.path.join(ROOT_DIR, "models", "deeplab.pth")
        checkpoint = torch.load(path, map_location=DEVICE)

        model = DeepLab(n_channels=6, n_classes=2)

    else:
        raise ValueError("Invalid model name")

    state_dict = checkpoint["model_state_dict"]
    
    if model_name == "deeplab":
        # If the saved checkpoint was from the inner model directly, prefix the keys 
        if not list(state_dict.keys())[0].startswith("model."):
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}
            
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    _models_cache[model_name] = model
    return model