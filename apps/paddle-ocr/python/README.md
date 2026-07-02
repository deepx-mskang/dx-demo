# PP-OCRv5 DEEPX Baseline

PP-OCRv5 DEEPX benchmarking toolchain with NPU acceleration and automated async pipeline.

## Key Features

- **High Performance**: Async mode delivers **2.5x faster** inference with **3.2x higher** throughput
- **NPU Acceleration**: DEEPX DX-M1 NPU optimization for maximum performance
- **Automated Setup**: One-command pipeline with model auto-download and environment setup
- **User-Controlled Benchmarking**: 10-second prompt to confirm benchmark execution
- **Async Processing**: Optimized parallel processing pipeline for maximum performance
- **Rich Visualization**: Automatic generation of OCR result visualizations

## Performance Results

### Custom Dataset Overview

This project uses a diverse custom Chinese dataset for benchmarking. The dataset consists of various real-world scenarios including street signs, handwritten text, exam papers, textbooks, and newspapers, providing comprehensive coverage of different text recognition challenges with detailed annotations including text content and bounding box coordinates.

**Test Configuration**:
- Dataset: Custom Chinese document dataset (20 images)
- Data Format: PNG images with JSON annotations containing text content
- Model: DXNN-OCR v5 full pipeline (PP-OCRv5 → DEEPX NPU accelerated)
  - Text detection: PP-OCRv5 det → DXNN det_v5 (NPU accelerated)
  - Text classification: PP-OCRv5 cls → DXNN cls_v5 (NPU accelerated)
  - Text recognition: PP-OCRv5 rec → DXNN rec_v5 multi-ratio models (NPU accelerated)
- Hardware configuration:
  - Platform: Rockchip RK3588 IR88MX01 LP4X V10
  - NPU: DEEPX DX-M1 Accelerator Card
    - PCIe: Gen3 X4 interface [01:00:00]
    - Firmware: v2.1.0
  - CPU: ARM Cortex-A55 8-core @ 2.35GHz (8nm process)
  - Memory: 8GB LPDDR4X
  - Operating System: Ubuntu 20.04 LTS or higher
  - Runtime: DXRT v3.0.0 + RT driver v1.7.1 + PCIe driver v1.4.1

### Benchmark Results

#### Async Mode (Parallel Processing)
| Processing Mode | Average Inference Time (ms) | Average FPS | Average CPS (chars/s) | Average Accuracy (%) | Total Processing Time (s) |
|---|---|---|---|---|---|
| `Async (Parallel)` | 523.79 | 1.91 | 908.38 | 92.19 | 11.53 |

**Performance Highlights**:
- **Optimized Pipeline**: Async-only processing for maximum throughput
- **Inference Speed**: 523.79ms average per image
- **High Throughput**: 908.38 characters per second processing rate
- **Excellent Accuracy**: 92.19% recognition accuracy maintained

## System Requirements

### Hardware Requirements
- **NPU**: DEEPX DX-M1 Accelerator Card
  - PCIe: Gen3 X4 interface
  - Memory: LPDDR5 6000 Mbps, 3.92GiB minimum
  - Board: M.2, Rev 1.5 or higher

### Software Requirements
- **DXRT**: v3.0.0 or higher
  - RT Driver version: v1.7.1 or higher
  - PCIe Driver version: v1.4.1 or higher
  - FW version: v2.1.6 or higher
- **Python Package**: dx-engine v1.1.2 or higher

### Version Verification
```bash
# Check DXRT version and device status
dxrt-cli -s

# Check dx-engine version
pip list | grep dx
```

## Quick Start

### Automated Full Pipeline Execution

```
sudo apt install -y libxcb-cursor0 libxkbcommon-x11-0 libxcb-xinerama0
```

**One-Step Automated Benchmark Pipeline:**
```bash
git clone git@github.com:DEEPX-AI/dx_baidu_gui.git
cd dx_baidu_gui
./startup.sh --dx_rt {dx_rt path dir}
```

**What `startup.sh` Does Automatically:**

✨ **Phase 1: Environment Setup**
- Creates and activates Python virtual environment
- Installs all required dependencies from `requirements.txt`
- **Auto-downloads DXNN models** if missing (runs `setup.sh` automatically)
- Verifies dataset and ground truth files

⚡ **Phase 2: RT Optimization** 
- Automatically applies `set_env.sh 1 2 1 3 2 4` for optimal NPU performance
- Configures DXRT environment variables for maximum throughput

🚀 **Phase 3: User-Controlled Benchmark**
- **10-second prompt**: "Run benchmark now? [y/N] (auto-no in 10s)"
- If user confirms 'y': Runs async benchmark pipeline
- If timeout/no: Exits gracefully with environment ready
- Saves results to `output_async/` directory
- Generates comprehensive performance report

**Final Output (if benchmark runs):**
- High-performance async processing results
- Detailed performance metrics and timing analysis
- Comprehensive logs for troubleshooting in `logs/`
- Ready-to-use benchmark reports in `output_async/`

**Model Auto-Setup:**
- Automatically checks for `../../workspace/models/ocr/dxnn_optimized` and `dxnn_mobile_optimized`
- Runs `setup.sh` automatically if models are missing
- No manual model download required

**RT Optimization (Recommended):**
```bash
# Apply DXRT optimization settings for maximum performance
source ./set_env.sh 1 2 1 3 2 4

# Explanation of parameters:
# CUSTOM_INTER_OP_THREADS_COUNT=1     # Inter-operation parallelism
# CUSTOM_INTRA_OP_THREADS_COUNT=2     # Intra-operation parallelism  
# DXRT_DYNAMIC_CPU_THREAD=1           # Dynamic CPU thread management
# DXRT_TASK_MAX_LOAD=3                # Maximum task load
# NFH_INPUT_WORKER_THREADS=2          # Input worker threads
# NFH_OUTPUT_WORKER_THREADS=4         # Output worker threads
```

**Advanced Usage Examples:**
```bash
# Step 1: Apply RT optimization (recommended for best performance)
source ./set_env.sh 1 2 1 3 2 4

# Step 2: Run async benchmark (optimized parallel processing)
python scripts/dxnn_benchmark.py \
    -d images/ \
    --mode async \
    --output results/ \
    --ground-truth images/labels.json \
    --runs 1

# Alternative: Use mobile model variant
python scripts/dxnn_benchmark.py \
    -d images/ \
    --mode async \
    --use-mobile \
    --output results_mobile/ \
    --ground-truth images/labels.json \
    --runs 1
```

**Interactive GUI Demo:**
```bash
# Launch interactive GUI demo (async mode with standard model)
python demo.py --version v5 --mode async

# Launch interactive GUI demo (async mode with mobile model)
python demo.py --version v5 --mode async --use-mobile

# Quick demo launcher with auto-setup and mobile model selection
./run_v5_demo.sh
# - Auto-creates environment if missing
# - 10-second prompt: "Use mobile model? [y/N]"
# - Launches async demo with selected model
```

**GUI Features:**
- **Visual OCR Interface**: Drag & drop multiple images for instant OCR processing
- **Real-time Performance Metrics**: Live FPS and processing statistics
- **Accuracy Comparison**: Side-by-side GPU vs NPU accuracy analysis
- **Result Visualization**: Interactive preview of OCR detection and recognition results
- **Async Processing**: Optimized parallel processing for maximum performance

## 📁 Project Structure

```
├── demo.py                 # Interactive GUI demo with real-time OCR processing
├── startup.sh              # Fully automated async benchmark pipeline
│                           # - Environment setup & dependency installation
│                           # - Auto DXNN model download (setup.sh)
│                           # - RT optimization (set_env.sh) application
│                           # - User confirmation (10s prompt)
│                           # - Async benchmark execution → output_async/
├── run_v5_demo.sh          # Quick demo launcher with mobile model option
├── set_env.sh              # DXRT optimization settings
├── setup.sh                # DXNN model download script (auto-called)
├── scripts/
│   ├── dxnn_benchmark.py   # Async-optimized benchmark tool
│   │                       # - Async mode: Parallel processing (2.5x faster)
│   │                       # - Mobile model support (--use-mobile)
│   ├── calculate_acc.py    # PP-OCRv5 compatible accuracy calculation
│   └── ocr_engine.py       # DXNN NPU engine interface
├── engine/
│   ├── models/             # DXNN NPU models (auto-downloaded)
│   │   ├── dxnn_optimized/       # Standard DXNN models
│   │   └── dxnn_mobile_optimized/ # Mobile-optimized models
│   ├── paddleocr.py       # Core OCR pipeline with async support
│   ├── draw_utils.py       # Visualization utilities
│   ├── utils.py           # Processing utilities
│   └── fonts/             # Chinese fonts (for visualization)
└── images/                 # Custom dataset (20 PNG images + labels.json)
    ├── image_1.png ~ image_20.png  # Test images
    └── labels.json         # Ground truth annotations
```

**Custom Dataset:**
```bash
# Step 1: Apply RT optimization for best performance
source ./set_env.sh 1 2 1 3 2 4

# Step 2: Prepare your own images
mkdir -p images/custom
cp /path/to/your/images/* images/custom/

# Step 3: Run async benchmark (optimized parallel processing)
python scripts/dxnn_benchmark.py \
    --directory images/custom \
    --mode async \
    --ground-truth custom_labels.json \
    --output output_custom \
    --runs 3

# Optional: Run with mobile model for different performance profile
python scripts/dxnn_benchmark.py \
    --directory images/custom \
    --mode async \
    --use-mobile \
    --ground-truth custom_labels.json \
    --output output_custom_mobile \
    --runs 3
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project is forked and developed based on [DEEPX-AI/DXNN-OCR](https://github.com/DEEPX-AI/DXNN-OCR) project
- Thanks to [DEEPX team](https://deepx.ai) for NPU runtime and foundational framework support
- Thanks to the [PaddleOCR team](https://github.com/PaddlePaddle/PaddleOCR) for the excellent OCR framework
- Thanks to [Chris-godz/PP-OCRv5-DeepX-Baseline](https://github.com/Chris-godz/PP-OCRv5-DeepX-Baseline) for the baseline benchmarking tool for DXNN PP-OCR applications  

## 🏆 Recent Improvements

### Automated Async Pipeline (Dec 2025)
- **🚀 One-Command Execution**: Complete automated async pipeline with `./startup.sh`
  - Automatic environment setup and dependency installation
  - Auto DXNN model download (setup.sh)
  - RT optimization application (`set_env.sh`) for maximum NPU performance
  - User-controlled async benchmark execution (2.5x performance boost)
  - Comprehensive performance reporting
- **📊 Async-Only Results**: Optimized output directory (`output_async/`) with complete metrics
- **🔧 Zero Configuration**: No manual intervention required - everything automated from start to finish

### Code Refactoring & Performance Optimization (Nov 2025)
- **🔧 Architecture Refactoring**: Complete modular redesign for better maintainability
  - `BenchmarkConfig`: Centralized configuration management
  - `OCRBenchmark`: Core processing logic with helper methods
  - `BenchmarkReporter`: Result processing and visualization
- **⚡ Async Processing**: Introduced parallel processing mode with **2.5x performance boost**
- **📊 Enhanced Reporting**: Comprehensive performance metrics and visualizations
- **🛡️ Robust Error Handling**: Improved stability and error recovery
- **🧪 Extensive Testing**: Validated performance on diverse datasets
