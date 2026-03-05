import numpy as np
import torch
from app.model_loader import load_model, DEVICE
from app.preprocessing import normalize_image

def run_inference(image_np, model_name):

    model = load_model(model_name)

    image = normalize_image(image_np)

    # Ensure contiguous float32 numpy array before converting to tensor
    # Using from_numpy instead of torch.tensor for cross-version compatibility
    image = np.ascontiguousarray(image, dtype=np.float32)
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)

    water_prob = probs[:, 1, :, :]
    prediction = torch.argmax(output, dim=1)

    water_prob_np = water_prob.squeeze().cpu().numpy()
    pred_np = prediction.squeeze().cpu().numpy()

    confidence = float(water_prob_np[pred_np == 1].mean() * 100) \
        if (pred_np == 1).sum() > 0 else 0.0

    return pred_np, water_prob_np, confidence