# 🌊 Water Segmentation using Multispectral and Optical Data

> A deep learning web application for automated water body segmentation from multispectral satellite imagery, powered by **U-Net** and **DeepLabV3+** architectures.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
  - [U-Net](#u-net)
  - [DeepLabV3+](#deeplabv3)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Web Application](#web-application)
- [Getting Started](#getting-started)
  - [Local Setup](#local-setup)
  - [Docker Deployment](#docker-deployment)
- [Model Performance](#model-performance)
- [Requirements](#requirements)
- [Notebooks](#notebooks)


---

## 🔍 Overview

This project presents an end-to-end pipeline for **water body detection and segmentation** from high-resolution multispectral satellite images (12-band GeoTIFF files). Two state-of-the-art deep learning segmentation models — **U-Net** and **DeepLabV3+ (ResNet-50 backbone)** — are trained and deployed via an interactive **Flask web application**.

Users can upload a `.tif` multispectral image and receive:

- A natural **RGB preview** of the uploaded scene
- Binary **water segmentation masks** from both models
- **Probability heatmaps** highlighting water confidence
- **IoU scores** and **per-model confidence metrics**

---

## ✨ Features

| Feature                  | Description                              |
| ------------------------ | ---------------------------------------- |
| 🛰️ Multispectral Input   | Accepts 12-band GeoTIFF satellite images |
| 🤖 Dual Model Inference  | Runs U-Net and DeepLabV3+ in parallel    |
| 🌈 Heatmap Visualization | Jet-colormap probability overlays        |
| 📊 Metrics Dashboard     | IoU scores and prediction confidence     |
| 🖼️ RGB Preview           | Auto-generated natural color composite   |
| 🐳 Docker Ready          | Fully containerized with GPU support     |
| ⚡ GPU Acceleration      | CUDA-enabled inference via PyTorch       |

---

## 📂 Project Structure

```
Water-Segmentation/
│
├── app/                        # Flask web application
│   ├── app.py                  # Main Flask routes and logic
│   ├── inference.py            # Model inference pipeline
│   ├── model_loader.py         # Model loading & caching
│   ├── preprocessing.py        # Band normalization
│   ├── static/
│   │   ├── uploads/            # Uploaded .tif images
│   │   └── outputs/            # Generated masks & heatmaps
│   └── templates/
│       └── index.html          # Web UI template
│
├── models/
│   ├── unet.py                 # U-Net architecture definition
│   ├── unet.pth                # Trained U-Net weights (~30 MB)
│   ├── deeplab.py              # DeepLabV3+ architecture definition
│   └── deeplab.pth             # Trained DeepLabV3+ weights (~168 MB)
│
├── Data/                       # Training dataset (GeoTIFF + masks)
│
├── Segmentation.ipynb          # Training notebook (from scratch)
├── Seg_pretrained.ipynb        # Training notebook (pretrained backbone)
│
├── normalization.json          # Per-band min/max normalization values
├── run.py                      # Application entry point
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker Compose configuration
└── .dockerignore               # Files excluded from Docker build
```

---

## 🧠 Architecture

### U-Net

A custom **U-Net** implementation with an encoder-decoder structure and skip connections, trained from scratch on multispectral data.

- **Input channels:** 6 (bands: 1, 2, 5, 6, 7, 12)
- **Output classes:** 2 (water / non-water)
- **Encoder depths:** 64 → 128 → 256 → 512
- **Upsampling:** Transposed convolutions with skip connections
- **Activation:** ReLU + Batch Normalization

```
Input (6ch) → [Conv×2 → Pool] × 3 → Bridge → [UpConv + Skip → Conv×2] × 3 → Output (2ch)
```

### DeepLabV3+

A **DeepLabV3+** model built on **ResNet-50** backbone, adapted for 6-channel multispectral input via weight initialization from pretrained 3-channel weights.

- **Backbone:** ResNet-50 (pretrained on ImageNet)
- **Input channels:** 6 (modified first conv layer)
- **Output classes:** 2
- **Atrous Spatial Pyramid Pooling (ASPP):** Multi-scale context aggregation
- **Extra channel initialization:** Mean of RGB weights for bands 4–6

---

## 📡 Dataset

The dataset consists of **12-band multispectral GeoTIFF** satellite images with corresponding binary masks indicating water presence.

The **6 bands** selected for training are:

| Index | Band    | Description         |
| ----- | ------- | ------------------- |
| 0     | Band 1  | Coastal Aerosol     |
| 1     | Band 2  | Blue                |
| 4     | Band 5  | Near-Infrared (NIR) |
| 5     | Band 6  | SWIR-1              |
| 6     | Band 7  | SWIR-2              |
| 11    | Band 12 | Water Probablity    |

> Band selection was guided by their high sensitivity to water bodies (SWIR and NIR are especially effective for water/land discrimination).

---

## ⚙️ Preprocessing

Preprocessing is handled in `app/preprocessing.py` and uses **per-band min-max normalization** based on statistics computed from the training dataset, stored in `normalization.json`.

```python
normalized = (pixel_value - band_min) / (band_max - band_min + 1e-8)
```

After normalization, the image is transposed from `(H, W, C)` to `(C, H, W)` format for PyTorch compatibility.

---

## 🖥️ Web Application

The app is built with **Flask** and served via **Gunicorn** in production. It exposes two routes:

| Route      | Method | Description                                                |
| ---------- | ------ | ---------------------------------------------------------- |
| `/`        | `GET`  | Renders the main upload page                               |
| `/predict` | `POST` | Accepts a `.tif` file, runs inference, and returns results |

### Inference Pipeline

```
Upload .tif → Load with Rasterio → Select 6 Bands → Normalize
    → Run U-Net Inference → Run DeepLabV3+ Inference
    → Save Masks & Heatmaps → Render Results Page
```

### Outputs per Prediction

| Output                | Description                                         |
| --------------------- | --------------------------------------------------- |
| `rgb.png`             | Natural color composite (R=Band4, G=Band3, B=Band2) |
| `unet_mask.png`       | Binary water mask from U-Net                        |
| `unet_heatmap.png`    | Water probability heatmap from U-Net                |
| `deeplab_mask.png`    | Binary water mask from DeepLabV3+                   |
| `deeplab_heatmap.png` | Water probability heatmap from DeepLabV3+           |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- CUDA-capable GPU (recommended) or CPU fallback
- Docker & Docker Compose (for containerized deployment)

---

### Local Setup

**1. Clone the repository**

```bash
git clone https://github.com/your-username/Water-Segmentation-using-Multispectral-and-optical-Data.git
cd Water-Segmentation-using-Multispectral-and-optical-Data
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Ensure model weights are in place**

```
models/
├── unet.pth
└── deeplab.pth
```

**5. Run the application**

```bash
python run.py
```

Then open your browser at **[http://localhost:5000](http://localhost:5000)**

---

### Docker Deployment

#### Using Docker Compose (Recommended)

```bash
docker-compose up --build
```

The app will be accessible at **[http://localhost:5000](http://localhost:5000)**

#### Manual Docker Build

```bash
# Build the image
docker build -t water-segmentation .

# Run the container
docker run -p 5000:5000 --gpus all water-segmentation
```

> **Note:** GPU support in Docker requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to be installed on the host machine.

#### Docker Environment Details

| Property        | Value                                           |
| --------------- | ----------------------------------------------- |
| Base Image      | `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime` |
| WSGI Server     | Gunicorn                                        |
| Request Timeout | 300 seconds                                     |
| Exposed Port    | 5000                                            |
| GPU Support     | NVIDIA CUDA 12.1                                |

---

## 📊 Model Performance

| Model          | IoU Score | Notes                                     |
| -------------- | --------- | ----------------------------------------- |
| **U-Net**      | 72.45%    | Custom architecture, trained from scratch |
| **DeepLabV3+** | 78.82%    | Pretrained ResNet-50 backbone, fine-tuned |

> Confidence scores are dynamically computed per-image based on the mean predicted water probability over all predicted water pixels.

---

## 📦 Requirements

```
flask
numpy<2
opencv-python
rasterio
torch          # Provided by the base Docker image
torchvision    # Provided by the base Docker image
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 📓 Notebooks

Two Jupyter notebooks are included for model training and experimentation:

| Notebook               | Description                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| `Segmentation.ipynb`   | Full training pipeline from scratch — data loading, augmentation, training loop, evaluation |
| `Seg_pretrained.ipynb` | Transfer learning approach using a pretrained DeepLabV3+ ResNet-50 backbone                 |

Both notebooks cover:

- Data loading from GeoTIFF files using `rasterio`
- Band selection and normalization
- Model training with PyTorch
- Loss computation (CrossEntropy)
- IoU metric evaluation
- Visualization of predictions
