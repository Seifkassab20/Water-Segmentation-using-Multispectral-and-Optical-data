from flask import Flask, render_template, request
import os
import numpy as np
import cv2
import rasterio
import traceback
from app.inference import run_inference

app = Flask(__name__)

# Define absolute paths based on the file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: we are inside 'app/', so 'static' is a sibling
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ===============================
# Load TIF Image
# ===============================
def load_tif_image(path):
    with rasterio.open(path) as src:
        image = src.read()
    return np.transpose(image, (1, 2, 0))


# ===============================
# Create RGB Preview
# ===============================
def create_rgb(image_np):
    red = image_np[:, :, 3]
    green = image_np[:, :, 2]
    blue = image_np[:, :, 1]

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
    return rgb.astype(np.uint8)


@app.route("/")
def home():
    return render_template("index.html")


# ===============================
# Prediction Route
# ===============================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("image")

        if not file:
            return "No image uploaded", 400

        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        image_np = load_tif_image(image_path)

        # Select same 6 bands used during training
        if image_np.shape[2] >= 12:
            band_indices = [0, 1, 4, 5, 6, 11]
            image_to_model = image_np[:, :, band_indices]
        else:
            image_to_model = image_np

        # Create RGB preview
        rgb = create_rgb(image_np)
        rgb_path = os.path.join(OUTPUT_FOLDER, "rgb.png")
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # ===============================
        # Run UNet
        # ===============================
        pred_mask_unet, water_prob_unet, confidence_unet = run_inference(
            image_to_model, "unet"
        )

        # Save UNet Mask
        unet_mask_path = os.path.join(OUTPUT_FOLDER, "unet_mask.png")
        cv2.imwrite(unet_mask_path, (pred_mask_unet * 255).astype(np.uint8))

        # Save UNet Heatmap
        unet_heatmap = (water_prob_unet * 255).astype(np.uint8)
        unet_heatmap = cv2.applyColorMap(unet_heatmap, cv2.COLORMAP_JET)
        unet_heatmap_path = os.path.join(OUTPUT_FOLDER, "unet_heatmap.png")
        cv2.imwrite(unet_heatmap_path, unet_heatmap)

        # ===============================
        # Run DeepLab
        # ===============================
        pred_mask_dl, water_prob_dl, confidence_dl = run_inference(
            image_to_model, "deeplab"
        )

        # Save DeepLab Mask
        dl_mask_path = os.path.join(OUTPUT_FOLDER, "deeplab_mask.png")
        cv2.imwrite(dl_mask_path, (pred_mask_dl * 255).astype(np.uint8))

        # Save DeepLab Heatmap
        dl_heatmap = (water_prob_dl * 255).astype(np.uint8)
        dl_heatmap = cv2.applyColorMap(dl_heatmap, cv2.COLORMAP_JET)
        dl_heatmap_path = os.path.join(OUTPUT_FOLDER, "deeplab_heatmap.png")
        cv2.imwrite(dl_heatmap_path, dl_heatmap)

        return render_template(
            "index.html",
            # Prefixing with / to be absolute from root of the domain
            rgb_image="/static/outputs/rgb.png",
            unet_mask="/static/outputs/unet_mask.png",
            unet_heatmap="/static/outputs/unet_heatmap.png",
            dl_mask="/static/outputs/deeplab_mask.png",
            dl_heatmap="/static/outputs/deeplab_heatmap.png",
            iou_unet=72.45,
            iou_deeplab=78.82,
            conf_unet=round(confidence_unet, 2),
            conf_deeplab=round(confidence_dl, 2)
        )
    except Exception as e:
        print("CRITICAL ERROR during inference:")
        traceback.print_exc()
        return f"Segmentation Error: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)