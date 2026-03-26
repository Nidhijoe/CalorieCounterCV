# 🍽️ AI-Powered Calorie Counter

> Computer Vision based food detection and nutrition tracking using **YOLOv8 + MobileNetV2 + Color Analysis**

---

## 📌 Problem Statement

Manually tracking calorie intake is tedious and error-prone — especially for Indian meals with multiple components on a single plate. This project automates the process using computer vision: upload a photo of your meal and instantly get an itemized nutrition breakdown for every detected food component.

---

## 🚀 Features

- Detects multiple Indian and Western food items from a single meal image
- YOLOv8 for object/region detection + MobileNetV2 for food classification
- Custom color-based analysis for Indian food (dal, roti, sabzi, curd, rajma, etc.)
- Per-item calorie, protein, carbs, and fat table
- Annotated output image with bounding boxes and confidence scores
- Macronutrient distribution pie chart
- Session-based daily calorie log in sidebar
- Local nutrition database (no API key required to run)
- Optional Nutritionix API integration for real-time nutrition data

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI Framework | Streamlit |
| Object Detection | YOLOv8n (Ultralytics) |
| Image Classification | MobileNetV2 (torchvision) |
| Color Analysis | NumPy (custom HSV/RGB heuristics) |
| Image Processing | Pillow |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib |
| Nutrition Data | Local DB / Nutritionix API (optional) |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python **3.10 or higher**
- pip
- Git

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/Nidhijoe/CalorieCounterCV.git
cd CalorieCounterCV
```

---

### Step 2 — Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

---

### Step 3 — Install PyTorch (CPU version)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

### Step 4 — Install remaining dependencies

```bash
pip install streamlit ultralytics pillow pandas matplotlib requests numpy==1.26.4
```

---

### Step 5 — Run the application

```bash
streamlit run app.py
```

The app opens automatically in your browser at: **http://localhost:8501**

---

## 📁 Project Structure

```
CalorieCounterCV/
│
├── app.py              # Main Streamlit UI application
├── detector.py         # Food detection pipeline (YOLO + MobileNet + Color Analysis)
├── nutrition.py        # Nutrition lookup (local DB + optional Nutritionix API)
├── README.md           # Project documentation
└── assets/             # Sample meal images for testing
```

---

## 🧪 How to Use

1. Run the app: `streamlit run app.py`
2. Upload any meal photo (JPG / PNG) using the file uploader
3. The app will:
   - Detect food regions using YOLOv8
   - Classify each region (MobileNetV2 + color analysis)
   - Display an annotated image with labels and confidence scores
   - Show a per-item nutrition table (calories, protein, carbs, fat)
   - Display total macros and a pie chart
4. Click **"Add to Daily Log"** to track the meal in the sidebar

---

## 🔍 Computer Vision Concepts Applied

| Concept | Implementation |
|---|---|
| Object Detection | YOLOv8n detects food regions and containers (bowls, plates) |
| Transfer Learning | Pretrained MobileNetV2 weights (ImageNet) fine-mapped to food labels |
| Color Segmentation | Custom RGB/HSV thresholds to identify Indian foods by color profile |
| Multi-label Detection | Top-K predictions per region to handle mixed plates |
| Image Preprocessing | Resize, normalize, dtype conversion for model compatibility |
| Bounding Box Annotation | Pillow ImageDraw with per-item color coding and confidence scores |

---

## 🍱 Indian Food Support

The system uses a custom color analysis layer specifically tuned for Indian mess food:

- White/light regions → **Rice** or **Curd**
- Bright yellow-orange → **Dal**
- Deep orange-red → **Rajma** / **Tomato curry**
- Golden-tan with texture → **Roti**
- Dark/olive green → **Sabzi**
- Brown with high texture variance → **Chicken curry**
- Yellow-grainy → **Khichdi** / **Poha**

Thali inference rules are also applied — e.g., if rice is detected without dal, dal is inferred as likely present.

---

## (Optional) Nutritionix API Setup

For real-time nutrition data, get free credentials from [nutritionix.com](https://www.nutritionix.com/business/api) and set environment variables:

**Windows:**
```bash
set NUTRITIONIX_APP_ID=your_app_id
set NUTRITIONIX_API_KEY=your_api_key
```

**macOS / Linux:**
```bash
export NUTRITIONIX_APP_ID=your_app_id
export NUTRITIONIX_API_KEY=your_api_key
```

If not set, the app automatically uses the built-in nutrition database — no setup needed.

---

## ⚠️ Known Limitations

- Detection accuracy depends on image lighting and clarity
- YOLOv8n COCO weights cover ~10 food classes natively; Indian food detection relies on the color analysis layer
- Portion size is estimated from standard serving sizes, not actual plate measurements
- MobileNetV2 is trained on ImageNet (not a food-specific dataset); custom label mapping is used to redirect its predictions to food categories

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

Developed by **Nidhi Joshi** as a BYOP submission for the Computer Vision course on VITyarthi — VIT Bhopal University.
