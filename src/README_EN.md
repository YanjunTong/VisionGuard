## VisionGurd

[简体中文](../README.md) | English

**A multimodal learning-based video anomaly detection system, specifically optimized for vehicle collision detection, supporting recognition and localization of various abnormal events.**

## Features

· 🎯 Multimodal Fusion: Combines visual and textual information to improve detection accuracy
· ⏱️ Temporal Localization: Precisely detects start and end times of abnormal events
· 🚀 Efficient Inference: Supports real-time video stream processing
· 🔧 Weakly Supervised Learning: Trains temporal localization models with only video-level labels
· 📊 Multi-Task Learning: Simultaneously performs anomaly detection, event classification, and temporal localization

## Supported Anomalous Events

· 🚗 Vehicle Collision
· 🔥 Fire
· 👊 Fighting
· 🧍 Falling
· ✅ Normal Scenes

## Requirements

· Python 3.8+
· PyTorch 1.12+
· CUDA 11.0+ (GPU recommended)

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/YanjunTong/VisionGuard.git
cd VisionGuard
```

2. **Install dependencies:**

```bash
pip install torch torchvision
pip install opencv-python pillow clip-by-openai
pip install numpy tqdm
```

## Quick Start

1. **Data Preprocessing**

```bash
python process.py
```

2. **Model Training**

```bash
python train.py
```

3. **Inference & Detection**

```bash
python inference.py
```

## Project Structure

```
video-anomaly-detection/
├── process.py          # Data preprocessing and feature extraction
├── train.py           # Model training script
├── inference.py       # Inference and detection script
├── preprocessed_data/ # Preprocessed feature storage
│   ├── video_features/
│   ├── text_features/
│   └── sim_matrices/
├── saved_models/      # Trained model weights
├── pseudo_labels/     # Pseudo-label data
└── README.md
```

## Data Preparation

1. **Video Data**

Place training videos in the train_videos/ directory and test videos in the video/ directory.

2. **Text Descriptions**

Configure video-text descriptions in process.py:

```python
TEST_TEXT_DESC_DICT = {
    "video_001": ["detect collision", "vehicle crash in video", ...],
    "normal_001": ["detect normal", "no abnormal events in video", ...]
}
```

## Model Architecture

**The system employs a three-head network structure:**

· Fusion Module: CLIP features + Attention mechanism
· Anomaly Detection Head: Binary classification for anomaly detection
· Event Classification Head: Multi-class classification for event types
· Temporal Localization Head: Regression for event time offsets
![framework](./framework.png "framework")

## Output Format

**Inference results are saved in submission.txt with the format:**

```
VideoID StartFrame EndFrame EventType
Example: car_01 125 189 Vehicle_Collision
```

## Training Configuration

**Key training parameters:**

· Batch Size: 32
· Learning Rate: 1e-4
· Epochs: 500
· Frames per Clip: 16
· Sliding Stride: 8

**Performance Optimization**

· Uses CLIP ViT-B/32 model to balance accuracy and speed
· Sliding window strategy to avoid missed detections
· Feature pre-computation to accelerate training

## License

[MIT License](https://mit-license.org/)

## Citation

If you use this project, please cite:

```bibtex
@software{VisionGuard2025,
  title = {VisionGuard},
  author = {Tong, Yanjun and Liang, Tianyv},
  year = {2025},
  url = {https://github.com/YanjunTong/VisionGuard}
}

@inproceedings{CLIP,
  title = {Learning Transferable Visual Models From Natural Language Supervision},
  author = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle = {International Conference on Machine Learning},
  pages = {8748--8763},
  year = {2021},
  organization = {PMLR}
}
```

## Contributing

Issues and Pull Requests are welcome!

## Contact

· Email: yanjun_tong@outlook.com
· GitHub: @yanjuntong
