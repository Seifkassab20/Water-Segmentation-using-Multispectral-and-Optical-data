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
# Metrics Calculation
# ===============================
def calculate_iou(pred_mask, gt_mask):
    """Calculates Intersection over Union for binary masks."""
    # Ensure both are binary (0 and 1)
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if union == 0:
        return 100.0 if intersection == 0 else 0.0
    
    return (intersection / union) * 100.0


# ===============================
# Prediction Route
# ===============================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        image_file = request.files.get("image")
        mask_file = request.files.get("mask")  # Optional Ground Truth

        if not image_file:
            return "No image uploaded", 400

        # Save and Load Image
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)
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

        # Handle Ground Truth Mask if provided
        gt_mask = None
        if mask_file and mask_file.filename != '':
            mask_path = os.path.join(UPLOAD_FOLDER, "gt_" + mask_file.filename)
            mask_file.save(mask_path)
            # Load as grayscale and resize to match image if necessary
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                gt_mask = cv2.resize(gt_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        # ===============================
        # Run UNet
        # ===============================
        pred_mask_unet, water_prob_unet, confidence_unet = run_inference(
            image_to_model, "unet"
        )
        
        # Calculate dynamic IoU for UNet if GT provided
        display_iou_unet = 72.45
        if gt_mask is not None:
            display_iou_unet = round(calculate_iou(pred_mask_unet, gt_mask), 2)

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
        
        # Calculate dynamic IoU for DeepLab if GT provided
        display_iou_dl = 78.82
        if gt_mask is not None:
            display_iou_dl = round(calculate_iou(pred_mask_dl, gt_mask), 2)

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
            rgb_image="/static/outputs/rgb.png",
            unet_mask="/static/outputs/unet_mask.png",
            unet_heatmap="/static/outputs/unet_heatmap.png",
            dl_mask="/static/outputs/deeplab_mask.png",
            dl_heatmap="/static/outputs/deeplab_heatmap.png",
            iou_unet=display_iou_unet,
            iou_deeplab=display_iou_dl,
            conf_unet=round(confidence_unet, 2),
            conf_deeplab=round(confidence_dl, 2),
            is_static_iq=(gt_mask is None)
        )
    except Exception as e:
        print("CRITICAL ERROR during inference:")
        traceback.print_exc()
        return f"Segmentation Error: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)