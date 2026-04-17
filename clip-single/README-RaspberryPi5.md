# 🚀 DEEPX CLIP Starter

<div align="center">

**CLIP text-image similarity using ONNX and DEEPX NPU**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-1.14+-green.svg)](https://onnx.ai/)

</div>

---

## 📖 Overview

This project demonstrates CLIP (Contrastive Language-Image Pre-training) model inference using:
- **ONNX Runtime** for text encoding
- **DEEPX NPU** for accelerated image encoding
- **Real-time camera streaming** with text matching
- **Async inference** for maximum performance

Perfect for understanding vision-language applications with optimized inference pipelines.

## ✨ Features

- 🔥 **Optimized Inference**: ONNX and DEEPX NPU acceleration
- 🎥 **Real-time Processing**: Live camera streaming with text matching
- ⚡ **Async Support**: Non-blocking inference for high throughput
- 🎯 **Zero-shot Classification**: Pre-computed text embeddings
- 📊 **Similarity Scoring**: Cosine similarity between text and images
- 🛠️ **Easy Export**: Convert PyTorch models to ONNX/DXNN formats

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Text      │────▶│  ONNX Text   │────▶│  Text       │
│  Input      │      │   Encoder    │      │  Features   │
└─────────────┘      └──────────────┘      └─────────────┘
                                                      │
                                                      ▼
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Image     │────▶│  DEEPX NPU   │────▶│  Image      │
│  Input      │      │   Encoder    │      │  Features   │
└─────────────┘      └──────────────┘      └─────────────┘
                                                      │
                                                      ▼
                                              ┌─────────────┐
                                              │  Similarity │
                                              │   Matrix    │
                                              └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- DEEPX NPU SDK (for image encoding)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DEEPX-AI/dx-clip-starter
   cd dx-clip-starter
   ```

2. **Install python3-pyqt5**
   ```bash
   sudo apt update
   sudo apt install python3-pyqt5
   ```

3. **Create virtual environment**
   ```bash
   python3 -m venv --system-site-packages .venv-clip
   source .venv-clip/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements-rpi5.txt
   ```

5. **DEEPX SDK has been installed**
- Refer to https://github.com/DEEPX-AI/dx-all-suite/blob/main/docs/source/installation.md

6. **Install dx-engine**
   ```base
   cd <path of dx-all-suite>/dx-runtime/dx_rt/python_package
   pip install .
   ```

### Model Setup

#### Supported Model List
- ViT-B-16-dfn2b
- ViT-B-16-quickgelu-metaclip_fullcc
- ViT-B-32-256-datacomp_s34b_b86k
- ViT-B-32-quickgelu-metaclip_fullcc
- ViT-L-14-336-openai
- ViT-L-14-datacomp_xl_s13b_b90k
- ViT-L-14-quickgelu-dfn2b (* current default)

#### Download Pre-compiled Models
- How to download and extract:
   ```
   wget https://cs.deepx.ai/_deepx_fae_archive/demo_application/clip-models.tar.gz
   tar xzf clip-models.tar.gz
   ```
- Verify downloaded files:
   ```
   tree -lh dxnn onnx
   [4.0K]  dxnn
   ├── [170M]  ViT-B-16-dfn2b.dxnn
   ├── [170M]  ViT-B-16-quickgelu-metaclip_fullcc.dxnn
   ├── [171M]  ViT-B-32-quickgelu-metaclip_fullcc.dxnn
   └── [594M]  ViT-L-14-quickgelu-dfn2b.dxnn
   [4.0K]  onnx
   ├── [242M]  ViT-B-16-dfn2b-text.onnx
   ├── [242M]  ViT-B-16-dfn2b-text.onnx.data
   ├── [ 279]  ViT-B-16-dfn2b-transform.txt
   ├── [242M]  ViT-B-16-quickgelu-metaclip_fullcc-text.onnx
   ├── [242M]  ViT-B-16-quickgelu-metaclip_fullcc-text.onnx.data
   ├── [ 279]  ViT-B-16-quickgelu-metaclip_fullcc-transform.txt
   ├── [242M]  ViT-B-32-quickgelu-metaclip_fullcc-text.onnx
   ├── [242M]  ViT-B-32-quickgelu-metaclip_fullcc-text.onnx.data
   ├── [ 279]  ViT-B-32-quickgelu-metaclip_fullcc-transform.txt
   ├── [472M]  ViT-L-14-quickgelu-dfn2b-text.onnx
   ├── [472M]  ViT-L-14-quickgelu-dfn2b-text.onnx.data
   └── [ 279]  ViT-L-14-quickgelu-dfn2b-transform.txt
   ```
> If you want to compile the models using DX-Compiler, please refer to [README.md](./README.md) file.


## 📚 Usage Examples

### 1. Individual Encoders

**Text Encoder Only:**
```bash
python3 simple-text-encoder.py \
  --onnx-model onnx/ViT-L-14-quickgelu-dfn2b-text.onnx \
  --texts "a photo of a cat" "a dog" "a diagram" \
  --show-similarity
```

**Image Encoder Only:**
```bash
python3 simple-image-encoder.py \
  --model dxnn/ViT-L-14-quickgelu-dfn2b.dxnn \
  --image assets/sample.jpg
```

### 2. Text-Image Similarity Calculator

Calculate similarity between multiple texts and a single image:

```bash
python3 simple-text-image-similarity.py \
  --texts "a cat" "a dog" "a person" "fine in car" \
  --image assets/img-encoder-sample-1.png
```

**Output:**
```
Text-Image Similarity:
--------------------------------------------------
Text                                    Similarity
--------------------------------------------------
fine in car                              0.1748
a person                                 0.0459
a dog                                    0.0217
a cat                                    0.0016
--------------------------------------------------
```

### 3. Real-time Camera Text Matcher

Match predefined texts with live camera feed:

```bash
python3 camera-text-matcher.py \
  --texts "a person" "thumbs up" "hand heart gesture"
```

**Features:**
- Real-time text matching on camera feed
- Displays top 3 matching texts with confidence scores (sorted by similarity)
- Configurable frame skipping for performance
- Previous results maintained during frame skip to prevent flickering

### 4. Async Camera Text Matcher (High Performance)

Use async inference for maximum throughput:

```bash
python3 camera-text-matcher-async.py \
  --texts "a person" "thumbs up" "hand heart gesture"
```

**Benefits:**
- Non-blocking inference
- Higher FPS with async processing
- Better resource utilization

### 5. Async Camera Text Matcher GUI

Use the GUI app to monitor the live camera feed and ranked text matches:

```bash
python3 camera-text-matcher-async-gui.py \
  --texts "A persion giving a thumbs up" \
          "A person clapping hands" \
          "A person making a hand heart" \
          "A person making a V sign with fingers" \
          "A person holding a cup" \
          "A person signaling OK with fingers" \
  --skip-frames 9
```

To use the different model, refer to the following command:
```bash
python3 camera-text-matcher-async-gui.py \
  --text-encoder onnx/ViT-B-16-dfn2b-text.onnx \
  --image-encoder dxnn/ViT-B-16-dfn2b.dxnn \
  --texts "A persion giving a thumbs up" \
          "A person clapping hands" \
          "A person making a hand heart" \
          "A person making a V sign with fingers" \
          "A person holding a cup" \
          "A person signaling OK with fingers" \
  --skip-frames 9
```

Use a video input with the following sample texts:
```bash
python3 camera-text-matcher-async-gui.py \
  --texts "A car on fire with bright flames and black smoke" \
          "People holding a gun are at the airport and a terrorist attack occurred" \
          "A person lying on the floor after falling down in a warehouse" \
          "Cars are driving on the road" \
          "Car accident occurred on the road" \
          "A massive explosion occurred in a large concrete structure" \
  --skip-frames 9 \
  --input assets/CLIP-demo.mp4
```


**Benefits:**
- Live video preview with async inference
- Score cards for each input text
- Same `--skip-frames` behavior as the async CLI version


## 🛠️ Advanced Usage

### Performance Tuning

**Frame Skipping:**
- `--skip-frames 0`: Process every frame (highest accuracy, lower FPS)
- `--skip-frames 1`: Process 1 out of every 2 frames
- `--skip-frames 2`: Process 1 out of every 3 frames (balanced, default)
- `--skip-frames 4`: Process 1 out of every 5 frames (higher FPS)

**Camera Settings:**
```bash
--width 1920 --height 1080 --fps 30  # Full HD @ 30fps
--width 1280 --height 720 --fps 60   # HD @ 60fps
--width 1280 --height 720 --fps 30   # HD @ 30fps, MJPEG (default) 
```

## 📁 Project Structure

```
dx-clip-starter/
├── 📄 README.md                         # This file
├── 📄 requirements.txt                  # Python dependencies
│
├── 🔧 Export Scripts
│   ├── export_to_onnx_text_encoder.py   # Export text encoder to ONNX
│   └── export_to_onnx.py                # Export image encoder to ONNX
│
├── 🎯 Main Applications
│   ├── simple-text-image-similarity.py  # Text-image similarity calculator
│   ├── simple-text-encoder.py           # Text encoder only
│   ├── simple-image-encoder.py          # Image encoder onl
│   ├── camera-text-matcher.py           # Real-time camera matcher
│   ├── camera-text-matcher-async.py     # Async camera matcher
│   └── camera-text-matcher-async-gui.py # Async camera matcher GUI
│
└── 📦 Model Directories
    ├── onnx/                            # ONNX text encoder models
    └── dxnn/                            # DEEPX NPU models
```

## ⚙️ Configuration

### Model Formats

- **Text Encoder**: ONNX format (`.onnx`)
- **Image Encoder**: DEEPX DXNN format (`.dxnn`)

## 🐛 Troubleshooting

### Common Issues

**1. ONNX Model Loading Error**
```
Error: Duplicate definition of name (embedding)
```
**Solution:** This issue has been resolved in `export_to_onnx_text_encoder.py`. The script now automatically appends `text_encoder` to output names to prevent conflicts. If you encounter this error with an old model, re-export using the latest version of the export script.

**2. Camera Not Found**
```
Error: Failed to open camera: /dev/video0
```
**Solution:** Check camera device path:
```bash
ls -la /dev/video*
# Use the correct device path
```

**3. DEEPX SDK Not Found**
```
ModuleNotFoundError: No module named 'dx_engine'
```
**Solution:** Install DEEPX SDK following their documentation.

**4. Feature Dimension Mismatch**
```
Error: shapes (4,768) and (1,1) not aligned
```
**Solution:** Ensure text and image encoders use the same model architecture.

## 📧 Contact

For questions and support, please open an issue on GitHub or contact dgkim@deepx.ai

---

<div align="center">

</div>
