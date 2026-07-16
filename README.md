# 🌾 Crop Recommendation AI Agent

An elegant, production-ready Streamlit application that uses Machine Learning to recommend the optimal crop to plant based on soil nutrient levels and environmental parameters.

## 🚀 Key Features

*   **Multi-Language UI Support**: Choose dynamically between English, Spanish (Español), Hindi (हिन्दी), and French (Français).
*   **Modern Glassmorphic Design**: Clean UI with custom CSS, card layouts, Outfit typography, and animated transitions.
*   **AI Predictor Model**: Loaded dynamically from `crop_model.pkl` with label translation via `label_encoder.pkl`.
*   **Comprehensive Inputs**: Analyzes Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Soil pH, and Rainfall.
*   **Deployment Ready**: Fully prepared for deployment on Streamlit Community Cloud or other platforms.

## 📋 Features Analyzed

The underlying Random Forest model analyzes 7 parameters:
1.  **Nitrogen (N)**: Soil nitrogen ratio (mg/kg).
2.  **Phosphorus (P)**: Soil phosphorus ratio (mg/kg).
3.  **Potassium (K)**: Soil potassium ratio (mg/kg).
4.  **Temperature**: Ambient temperature in °C.
5.  **Humidity**: Atmospheric humidity percentage (%).
6.  **pH**: Soil pH (acidity/alkalinity scale).
7.  **Rainfall**: Average annual rainfall in mm.

Supported crop predictions include: *Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, and Coffee*.

## 🛠️ Local Installation & Setup

Ensure you have Python 3.9+ installed.

### Using standard pip:

1. Clone or download the repository.
2. Open terminal in the project directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

### Using uv (Recommended - Fast & Isolated):

If you have the `uv` tool installed, you can launch the app instantly:
```bash
uv run streamlit run app.py
```

## 🌐 Deploy to Streamlit Community Cloud

1. Push this project repository to GitHub (include `app.py`, `requirements.txt`, `crop_model.pkl`, `label_encoder.pkl`, and `.gitignore`).
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**, select your repository, branch, and set the entry file path to `app.py`.
4. Click **Deploy**!
