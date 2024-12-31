# DeepSeek-V3 Windows Installation Guide
Created by Khanfar Systems - Making AI Accessible

## Overview
This repository provides a simplified, step-by-step guide to install and run DeepSeek-V3 on Windows. We've streamlined the process to make it as beginner-friendly as possible.

## System Requirements

### Minimum Requirements:
- Windows 10 or Windows 11
- Python 3.9 or higher
- NVIDIA GPU with 16GB VRAM (for 7B model)
- 32GB RAM
- 50GB free disk space

### Recommended:
- Windows 11
- Python 3.9
- NVIDIA GPU with 24GB VRAM
- 64GB RAM
- 100GB SSD space

## Installation Guide

### Step 1: Install Required Software

1. **Install Python 3.9:**
   - Download Python 3.9 from [Python.org](https://www.python.org/downloads/release/python-3913/)
   - Select "Windows installer (64-bit)"
   - During installation:
     - ✅ Check "Add Python 3.9 to PATH"
     - ✅ Click "Install Now"
   - Verify installation by opening Command Prompt:
     ```bash
     python --version
     ```
   Should show: `Python 3.9.x`

2. **Install Git:**
   - Download from [git-scm.com](https://git-scm.com/download/win)
   - Use default installation options
   - Verify installation:
     ```bash
     git --version
     ```

3. **Install CUDA Toolkit:**
   - Download CUDA 11.8 from [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)
   - Choose:
     - Windows
     - x86_64
     - 11
     - exe (local)
   - Run installer with default options
   - Verify installation:
     ```bash
     nvcc --version
     ```

### Step 2: Download This Repository

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/khanfar/DeepSeek-Windows.git
   cd DeepSeek-Windows
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate Virtual Environment:**
   ```bash
   venv\Scripts\activate
   ```

### Step 3: Install Dependencies

1. **Install PyTorch:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Install Other Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Download Model

1. **Run the Download Script:**
   ```bash
   python download_model.py
   ```
   This will download the 7B parameter model (approximately 14GB)

2. **Convert Model Format (if needed):**
   ```bash
   python fp8_cast_bf16.py --input-fp8-hf-path model_weights --output-bf16-hf-path model_weights_bf16
   ```

### Step 5: Start the Server

1. **Run the Server:**
   ```bash
   python windows_server.py --model model_weights_bf16 --trust-remote-code
   ```
   The server will start at: http://127.0.0.1:30000

2. **Test the Model:**
   ```bash
   python test_client.py
   ```

## Directory Structure
```
DeepSeek-Windows/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── download_model.py         # Script to download model
├── fp8_cast_bf16.py         # Model conversion script
├── kernel.py                # CUDA kernels
├── windows_server.py        # Local server
└── test_client.py           # Test script
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **"CUDA not available" Error:**
   - Ensure NVIDIA drivers are up to date
   - Verify CUDA installation:
     ```bash
     nvidia-smi
     ```
   - Should show your GPU and CUDA version

2. **"Out of Memory" Error:**
   - Close other applications
   - Reduce model parameters in `windows_server.py`:
     ```python
     max_tokens=100  # Reduce this value
     ```

3. **Import Errors:**
   - Ensure virtual environment is activated
   - Reinstall dependencies:
     ```bash
     pip install -r requirements.txt
     ```

4. **Download Issues:**
   - Check internet connection
   - Try using a VPN
   - Manual download option available on HuggingFace

### Performance Optimization

1. **For Better Speed:**
   - Use SSD for model storage
   - Close background applications
   - Update NVIDIA drivers

2. **For Lower Memory Usage:**
   - Enable 4-bit quantization
   - Reduce context length
   - Limit batch size

## Updates and Support

- **GitHub Issues:** Report problems or suggest improvements
- **Version Updates:** Check releases page for latest versions
- **Community:** Join our Discord for support

## Credits
- Original DeepSeek-V3 by DeepSeek-AI
- Windows adaptation by Khanfar Systems
- Community contributors

## License
- Code: MIT License
- Model: DeepSeek License
- Documentation: CC-BY-SA 4.0
