# 🥑 FreshScan AI – Food Freshness Detection Using Computer Vision

A production-grade deep-learning computer vision dashboard designed to classify agricultural produce freshness into **Fresh**, **Semi-Fresh / Moderate**, and **Rotten / Spoiled** tiers across 24 distinct categories, predict remaining shelf life, and provide tailored food safety recommendations.

> **Brand**: FreshScan AI  
> **Tagline**: *Smarter Food • Less Waste*  
> **Interface**: Futuristic Dark Glassmorphic Dashboard  
> **Core Technologies**: Python 3.11, TensorFlow/Keras, OpenCV, Streamlit

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Specification](#-dataset-specification)
- [Project Architecture](#-project-architecture)
- [Directory Structure](#-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Workflow](#-usage-workflow)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Model Training](#2-model-training)
  - [3. Running the Web Application](#3-running-the-web-application)
- [Technology Stack](#-technology-stack)
- [Future Enhancements](#-future-enhancements)

---

## 🔍 Project Overview

Post-harvest food loss is a major challenge in supply chain logistics, retail supermarkets, and household food security. Manual inspection is labor-intensive, subjective, and prone to error.

This project delivers an automated, vision-based quality grading solution that:
1. Identifies the commodity category (e.g., Banana, Orange, Tomato, etc.).
2. Accurately predicts its current condition level (**Fresh**, **Semi-Fresh**, or **Rotten**).
3. Provides visual confidence percentages and class probability breakdowns through an intuitive Streamlit interface.

---

## 📊 Dataset Specification

### **AgriFreshNET Processed_Data**
The model targets **24 distinct classification categories**, representing 8 agricultural commodities across 3 degradation stages:

| Produce Item | Fresh Class | Semi-Fresh Class | Rotten Class |
| :--- | :--- | :--- | :--- |
| **Banana** | `fresh_banana` | `semifresh_banana` | `rotten_banana` |
| **Orange** | `fresh_orange` | `semifresh_orange` | `rotten_orange` |
| **Pineapple** | `fresh_pineapple` | `semifresh_pineapple` | `rotten_pineapple` |
| **Tomato** | `fresh_tomato` | `semifresh_tomato` | `rotten_tomato` |
| **Eggplant** | `fresh_eggplant` | `semifresh_eggplant` | `rotten_eggplant` |
| **Cucumber** | `fresh_cucumber` | `semifresh_cucumber` | `rotten_cucumber` |
| **Papaya** | `fresh_papaya` | `semifresh_papaya` | `rotten_papaya` |
| **Bittermelon** | `fresh_bittermelon` | `semifresh_bittermelon` | `rotten_bittermelon` |

---

## 📁 Directory Structure

```text
Food_Freshness_Detection/
├── dataset/                    # AgriFreshNET dataset directory
│   └── README.md               # Dataset organization instructions
├── models/                     # Saved trained models (.keras) & artifacts
│   └── .gitkeep                # Keeps folder tracked in git
├── notebooks/                  # Jupyter notebooks for EDA and benchmarking
│   └── README.md               # Suggested notebooks guidance
├── src/                        # Modular source code
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Hyperparameters, paths, and class definitions
│   ├── dataset.py              # tf.data loading & data augmentation pipeline
│   ├── model.py                # Transfer learning architectures (MobileNetV2 / EfficientNet)
│   ├── predict.py              # Inference engine for images and webcam inputs
│   ├── train.py                # Standalone training CLI with callbacks
│   └── utils.py                # Metrics plotting, confusion matrix & label parsing
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Python 3.11 pinned dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.11** (Required for TensorFlow 2.15/2.16 binary compatibility)
- Git

### 1. Clone or Open the Repository
```bash
cd Food_Freshness_Detection
```

### 2. Create a Virtual Environment (Python 3.11)

**On Windows (Command Prompt / PowerShell):**
```powershell
# Using the Python 3.11 launcher:
py -3.11 -m venv venv

# Activate the virtual environment:
.\venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage Workflow

### 1. Data Preparation
Place your existing **AgriFreshNET Processed_Data** folder inside the `dataset/` directory.

*(Note: Do not rename or alter existing raw image folder structures; `src/dataset.py` automatically maps and sorts the 24 classes).*

### 2. Model Training
When ready to train the deep learning model:
```bash
# Basic run with default hyperparameters (MobileNetV2, 25 epochs)
python src/train.py

# Or customized with arguments:
python src/train.py --backbone MobileNetV2 --epochs 30 --batch-size 32 --lr 0.0001
```

Training automatically:
- Saves the best checkpoint to `models/food_freshness_best_model.keras`
- Saves `models/class_indices.json` for reliable inference mapping
- Generates loss/accuracy curves at `models/training_curves.png`

### 3. Running the Web Application
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to:
- Test image uploads (JPG/PNG).
- Use live webcam capture.
- Inspect top-3 probability predictions and freshness badges.
- Review dataset composition and class breakdown.

---

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow 2.15 / 2.16, Keras
- **Computer Vision**: OpenCV (`cv2`), Pillow (PIL)
- **Web Interface**: Streamlit
- **Data & Evaluation**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Runtime Target**: Python 3.11 (64-bit)

---

## 📌 Future Enhancements
- [ ] Grad-CAM heatmaps for explainable visual attention.
- [ ] Edge deployment using TensorFlow Lite (TFLite) for Raspberry Pi / Jetson Nano.
- [ ] Remaining shelf-life estimation (days before total decay).
