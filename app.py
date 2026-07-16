import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Translation System Configuration
# ---------------------------------------------------------
TRANSLATIONS = {
    "en": {
        "title": "🌾 Crop Recommendation AI Agent",
        "subtitle": "Discover the most suitable crop for your farm using advanced Machine Learning",
        "lang_label": "🌐 Select Interface Language",
        "sidebar_title": "⚙️ System Configuration",
        "sidebar_desc": "Adjust values to see predictions. The AI model analyzes soil nutrients and climate parameters.",
        "about_title": "ℹ️ About the Project",
        "about_text": "This AI-driven Crop Recommendation System helps farmers optimize crop selection. By analyzing soil chemistry (Nitrogen, Phosphorus, Potassium), pH levels, and climate factors (Temperature, Humidity, Rainfall), the machine learning model recommends the crop with the highest success rate.",
        "input_section": "🌱 Soil & Environmental Parameters",
        "n_label": "Nitrogen (N) - mg/kg",
        "n_help": "Ratio of Nitrogen content in soil (recommended range: 0-150)",
        "p_label": "Phosphorus (P) - mg/kg",
        "p_help": "Ratio of Phosphorus content in soil (recommended range: 5-145)",
        "k_label": "Potassium (K) - mg/kg",
        "k_help": "Ratio of Potassium content in soil (recommended range: 5-205)",
        "temp_label": "Temperature (°C)",
        "temp_help": "Ambient temperature in Celsius (recommended range: 10°C - 50°C)",
        "hum_label": "Humidity (%)",
        "hum_help": "Relative atmospheric humidity percentage (recommended range: 10% - 100%)",
        "ph_label": "Soil pH Level",
        "ph_help": "pH level of the soil, determining acidity or alkalinity (recommended range: 3.5 - 10.0)",
        "rain_label": "Rainfall (mm)",
        "rain_help": "Average annual rainfall in millimeters (recommended range: 20mm - 300mm)",
        "btn_predict": "Predict Optimal Crop 🌾",
        "btn_predict_loading": "Analyzing conditions...",
        "prediction_header": "Prediction Result",
        "recommended_crop": "Recommended Crop",
        "success_msg": "The AI model recommends planting **{crop}** based on the environmental conditions.",
        "error_validation": "⚠️ Please check your input parameters. Ensure all values are positive.",
        "footer_text": "Built with 💚 using Streamlit & Scikit-Learn | © 2026 Precision Agriculture Inc.",
        "model_details": "📊 Model Metrics",
        "model_details_text": "Algorithm: RandomForestClassifier\nFeatures: 7 Dimensional Soil-Climate Input\nDataset: 2200 Agricultural Records\nAccuracy: ~99%",
        "crop_names": {
            "apple": "Apple 🍎",
            "banana": "Banana 🍌",
            "blackgram": "Black Gram 🍛",
            "chickpea": "Chickpea 🧆",
            "coconut": "Coconut 🥥",
            "coffee": "Coffee ☕",
            "cotton": "Cotton ☁️",
            "grapes": "Grapes 🍇",
            "jute": "Jute 🧺",
            "kidneybeans": "Kidney Beans 🫘",
            "lentil": "Lentil 🍲",
            "maize": "Maize 🌽",
            "mango": "Mango 🥭",
            "mothbeans": "Moth Beans 🫛",
            "mungbean": "Mung Bean 🌱",
            "muskmelon": "Muskmelon 🍈",
            "orange": "Orange 🍊",
            "papaya": "Papaya 🥭",
            "pigeonpeas": "Pigeon Peas 🫛",
            "pomegranate": "Pomegranate 🍎",
            "rice": "Rice 🌾",
            "watermelon": "Watermelon 🍉"
        }
    },
    "es": {
        "title": "🌾 Agente AI de Recomendación de Cultivos",
        "subtitle": "Descubra el cultivo más adecuado para su finca utilizando Machine Learning avanzado",
        "lang_label": "🌐 Seleccionar Idioma de la Interfaz",
        "sidebar_title": "⚙️ Configuración del Sistema",
        "sidebar_desc": "Ajuste los valores para ver las predicciones. El modelo de IA analiza los nutrientes del suelo y los parámetros climáticos.",
        "about_title": "ℹ️ Sobre el Proyecto",
        "about_text": "Este sistema de recomendación de cultivos basado en IA ayuda a los agricultores a optimizar la selección de cultivos. Al analizar la química del suelo (nitrógeno, fósforo, potasio), los niveles de pH y los factores climáticos (temperatura, humedad, precipitaciones), el modelo de aprendizaje automático recomienda el cultivo con la mayor tasa de éxito.",
        "input_section": "🌱 Parámetros del Suelo y del Entorno",
        "n_label": "Nitrógeno (N) - mg/kg",
        "n_help": "Proporción de contenido de nitrógeno en el suelo (rango recomendado: 0-150)",
        "p_label": "Fósforo (P) - mg/kg",
        "p_help": "Proporción de contenido de fósforo en el suelo (rango recomendado: 5-145)",
        "k_label": "Potasio (K) - mg/kg",
        "k_help": "Proporción de contenido de potasio en el suelo (rango recomendado: 5-205)",
        "temp_label": "Temperatura (°C)",
        "temp_help": "Temperatura ambiente en grados Celsius (rango recomendado: 10°C - 50°C)",
        "hum_label": "Humedad (%)",
        "hum_help": "Porcentaje de humedad atmosférica relativa (rango recomendado: 10% - 100%)",
        "ph_label": "Nivel de pH del Suelo",
        "ph_help": "Nivel de pH del suelo, que determina la acidez o alcalinidad (rango recomendado: 3.5 - 10.0)",
        "rain_label": "Precipitaciones (mm)",
        "rain_help": "Precipitación media anual en milímetros (rango recomendado: 20mm - 300mm)",
        "btn_predict": "Predecir Cultivo Óptimo 🌾",
        "btn_predict_loading": "Analizando condiciones...",
        "prediction_header": "Resultado de la Predicción",
        "recommended_crop": "Cultivo Recomendado",
        "success_msg": "El modelo de IA recomienda sembrar **{crop}** según las condiciones ambientales.",
        "error_validation": "⚠️ Verifique los parámetros de entrada. Asegúrese de que todos los valores sean positivos.",
        "footer_text": "Creado con 💚 usando Streamlit y Scikit-Learn | © 2026 Precision Agriculture Inc.",
        "model_details": "📊 Métricas del Modelo",
        "model_details_text": "Algoritmo: RandomForestClassifier\nCaracterísticas: Entrada suelo-clima de 7 dimensiones\nConjunto de datos: 2200 registros agrícolas\nPrecisión: ~99%",
        "crop_names": {
            "apple": "Manzana 🍎",
            "banana": "Plátano 🍌",
            "blackgram": "Grama Negra 🍛",
            "chickpea": "Garbanzo 🧆",
            "coconut": "Coco 🥥",
            "coffee": "Café ☕",
            "cotton": "Algodón ☁️",
            "grapes": "Uvas 🍇",
            "jute": "Yute 🧺",
            "kidneybeans": "Frijoles de Riñón 🫘",
            "lentil": "Lenteja 🍲",
            "maize": "Maíz 🌽",
            "mango": "Mango 🥭",
            "mothbeans": "Frijol Polilla 🫛",
            "mungbean": "Frijol Mungo 🌱",
            "muskmelon": "Melón 🍈",
            "orange": "Naranja 🍊",
            "papaya": "Papaya 🥭",
            "pigeonpeas": "Gandules 🫛",
            "pomegranate": "Granada 🍎",
            "rice": "Arroz 🌾",
            "watermelon": "Sandía 🍉"
        }
    },
    "hi": {
        "title": "🌾 एआई फसल अनुशंसा एजेंट",
        "subtitle": "उन्नत मशीन लर्निंग का उपयोग करके अपने खेत के लिए सबसे उपयुक्त फसल की खोज करें",
        "lang_label": "🌐 इंटरफ़ेस भाषा चुनें",
        "sidebar_title": "⚙️ सिस्टम कॉन्फ़िगरेशन",
        "sidebar_desc": "पूर्वानुमान देखने के लिए मान समायोजित करें। एआई मॉडल मिट्टी के पोषक तत्वों और जलवायु मापदंडों का विश्लेषण करता है।",
        "about_title": "ℹ️ परियोजना के बारे में",
        "about_text": "यह एआई-संचालित फसल अनुशंसा प्रणाली किसानों को फसल चयन को अनुकूलित करने में मदद करती है। मिट्टी के रसायन (नाइट्रोजन, फास्फोरस, पोटेशियम), पीएच स्तर और जलवायु कारकों (तापमान, आर्द्रता, वर्षा) का विश्लेषण करके, मशीन लर्निंग मॉडल उच्चतम सफलता दर वाली फसल की सिफारिश करता है।",
        "input_section": "🌱 मिट्टी और पर्यावरण पैरामीटर",
        "n_label": "नाइट्रोजन (N) - मिलीग्राम/किग्रा",
        "n_help": "मिट्टी में नाइट्रोजन की मात्रा का अनुपात (अनुशंसित सीमा: 0-150)",
        "p_label": "फास्फोरस (P) - मिलीग्राम/किग्रा",
        "p_help": "मिट्टी में फास्फोरस की मात्रा का अनुपात (अनुशंसित सीमा: 5-145)",
        "k_label": "पोटेशियम (K) - मिलीग्राम/किग्रा",
        "k_help": "मिट्टी में पोटेशियम की मात्रा का अनुपात (अनुशंसित सीमा: 5-205)",
        "temp_label": "तापमान (°C)",
        "temp_help": "सेल्सियस में परिवेश का तापमान (अनुशंसित सीमा: 10°C - 50°C)",
        "hum_label": "आर्द्रता (%)",
        "hum_help": "सापेक्ष वायुमंडलीय आर्द्रता प्रतिशत (अनुशंसित सीमा: 10% - 100%)",
        "ph_label": "मिट्टी का पीएच स्तर",
        "ph_help": "मिट्टी का पीएच स्तर, जो अम्लता या क्षारीयता को निर्धारित करता है (अनुशंसित सीमा: 3.5 - 10.0)",
        "rain_label": "वर्षा (मिमी)",
        "rain_help": "औसत वार्षिक वर्षा मिलीमीटर में (अनुशंसित सीमा: 20mm - 300mm)",
        "btn_predict": "इष्टतम फसल की भविष्यवाणी करें 🌾",
        "btn_predict_loading": "स्थितियों का विश्लेषण किया जा रहा है...",
        "prediction_header": "पूर्वानुमान परिणाम",
        "recommended_crop": "अनुशंसित फसल",
        "success_msg": "एआई मॉडल पर्यावरण की स्थिति के आधार पर **{crop}** उगाने की सिफारिश करता है।",
        "error_validation": "⚠️ कृपया इनपुट पैरामीटर जांचें। सुनिश्चित करें कि सभी मान सकारात्मक हैं।",
        "footer_text": "स्ट्रीमलिट और स्किकिट-लर्न का उपयोग करके 💚 के साथ निर्मित | © 2026 प्रिसिजन एग्रीकल्चर इंक.",
        "model_details": "📊 मॉडल मेट्रिक्स",
        "model_details_text": "एल्गोरिथ्म: RandomForestClassifier\nविशेषताएं: 7 आयामी मिट्टी-जलवायु इनपुट\nडेटासेट: 2200 कृषि रिकॉर्ड\nसटीकता: ~99%",
        "crop_names": {
            "apple": "सेब 🍎",
            "banana": "केला 🍌",
            "blackgram": "उड़द दाल 🍛",
            "chickpea": "चना 🧆",
            "coconut": "नारियल 🥥",
            "coffee": "कॉफी ☕",
            "cotton": "कपास ☁️",
            "grapes": "अंगूर 🍇",
            "jute": "जूट 🧺",
            "kidneybeans": "राजमा 🫘",
            "lentil": "मसूर 🍲",
            "maize": "मक्का 🌽",
            "mango": "आम 🥭",
            "mothbeans": "मोठ दाल 🫛",
            "mungbean": "मूंग दाल 🌱",
            "muskmelon": "खरबूजा 🍈",
            "orange": "संतरा 🍊",
            "papaya": "पपीता 🥭",
            "pigeonpeas": "अरहर 🫛",
            "pomegranate": "अनार 🍎",
            "rice": "चावल 🌾",
            "watermelon": "तरबूज 🍉"
        }
    },
    "fr": {
        "title": "🌾 Agent IA de Recommandation de Culture",
        "subtitle": "Découvrez la culture la plus adaptée à votre exploitation grâce au Machine Learning",
        "lang_label": "🌐 Choisir la Langue de l'Interface",
        "sidebar_title": "⚙️ Configuration Système",
        "sidebar_desc": "Ajustez les valeurs pour voir les prédictions. Le modèle d'IA analyse les nutriments du sol et les paramètres climatiques.",
        "about_title": "ℹ️ À propos du Projet",
        "about_text": "Ce système intelligent de recommandation de cultures aide les agriculteurs à optimiser leur sélection. En analysant la composition chimique du sol (azote, phosphore, potassium), l'acidité (pH) et les facteurs climatiques (température, humidité, précipitations), notre modèle de machine learning suggère le type de culture garantissant le meilleur taux de réussite.",
        "input_section": "🌱 Paramètres du Sol & Environnementaux",
        "n_label": "Azote (N) - mg/kg",
        "n_help": "Taux d'azote dans le sol (plage recommandée : 0-150)",
        "p_label": "Phosphore (P) - mg/kg",
        "p_help": "Taux de phosphore dans le sol (plage recommandée : 5-145)",
        "k_label": "Potassium (K) - mg/kg",
        "k_help": "Taux de potassium dans le sol (plage recommandée : 5-205)",
        "temp_label": "Température (°C)",
        "temp_help": "Température ambiante en degrés Celsius (plage recommandée : 10°C - 50°C)",
        "hum_label": "Humidité (%)",
        "hum_help": "Taux d'humidité relative de l'air (plage recommandée : 10% - 100%)",
        "ph_label": "pH du Sol",
        "ph_help": "Le pH du sol, mesurant son acidité ou son alcalinité (plage recommandée : 3.5 - 10.0)",
        "rain_label": "Précipitations (mm)",
        "rain_help": "Pluviométrie annuelle moyenne en millimètres (plage recommandée : 20mm - 300mm)",
        "btn_predict": "Prédire la Culture Optimale 🌾",
        "btn_predict_loading": "Analyse en cours...",
        "prediction_header": "Résultat de la Prédiction",
        "recommended_crop": "Culture Recommandée",
        "success_msg": "Le modèle IA vous conseille de planter du/de la **{crop}** au vu des conditions fournies.",
        "error_validation": "⚠️ Veuillez vérifier vos paramètres d'entrée. Assurez-vous que toutes les valeurs soient positives.",
        "footer_text": "Développé avec 💚 via Streamlit & Scikit-Learn | © 2026 Precision Agriculture Inc.",
        "model_details": "📊 Détails du Modèle",
        "model_details_text": "Algorithme : RandomForestClassifier\nVariables : 7 dimensions (Sol & Climat)\nBase de données : 2200 observations\nPrécision : ~99%",
        "crop_names": {
            "apple": "Pomme 🍎",
            "banana": "Banane 🍌",
            "blackgram": "Haricot Mungo Noir 🍛",
            "chickpea": "Pois Chiche 🧆",
            "coconut": "Noix de Coco 🥥",
            "coffee": "Café ☕",
            "cotton": "Coton ☁️",
            "grapes": "Raisin 🍇",
            "jute": "Jute 🧺",
            "kidneybeans": "Haricots Rouges 🫘",
            "lentil": "Lentille 🍲",
            "maize": "Maïs 🌽",
            "mango": "Mangue 🥭",
            "mothbeans": "Haricot Papillon 🫛",
            "mungbean": "Haricot Mungo 🌱",
            "muskmelon": "Melon Brodé 🍈",
            "orange": "Orange 🍊",
            "papaya": "Papaye 🥭",
            "pigeonpeas": "Pois d'Angole 🫛",
            "pomegranate": "Grenade 🍎",
            "rice": "Riz 🌾",
            "watermelon": "Pastèque 🍉"
        }
    }
}

# ---------------------------------------------------------
# 2. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Crop Recommendation AI Agent",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 3. Model Loading Helper
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    """Load machine learning model and label encoder."""
    model = joblib.load("crop_model.pkl")
    with open("label_encoder.pkl", "rb") as file:
        label_encoder = pickle.load(file)
    return model, label_encoder

try:
    model, label_encoder = load_assets()
except Exception as e:
    st.error(f"Error loading pickle assets: {e}")
    st.stop()

# ---------------------------------------------------------
# 4. Custom Styling (Glassmorphism & Typography)
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Typography reset */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif !important;
}

/* Glassmorphic card styling for outputs */
.prediction-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 30px;
    margin-top: 25px;
    margin-bottom: 25px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    text-align: center;
    animation: fadeInUp 0.6s ease-in-out;
}

.prediction-card h2 {
    color: #38ef7d;
    font-weight: 700;
    margin-bottom: 5px;
}

.crop-result {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
    margin: 15px 0;
    letter-spacing: -0.5px;
}

/* Custom form elements styling */
div.stNumberInput {
    margin-bottom: 15px;
}

/* Customized predict button */
div.stButton > button {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    border: none !important;
    padding: 12px 30px !important;
    border-radius: 12px !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(56, 239, 125, 0.2) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(56, 239, 125, 0.4) !important;
}

div.stButton > button:active {
    transform: translateY(1px) !important;
}

/* Custom footer style */
.custom-footer {
    text-align: center;
    padding: 30px 0;
    margin-top: 60px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 0.9rem;
    color: #888888;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. Sidebar Layout (System Configuration)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Language Selector dropdown
    lang_opt = st.selectbox(
        "🌐 Language / भाषा / Idiome",
        options=["English", "Español", "हिन्दी", "Français"],
        index=0
    )
    
    # Map the display name to language code
    lang_map = {
        "English": "en",
        "Español": "es",
        "हिन्दी": "hi",
        "Français": "fr"
    }
    lang = lang_map[lang_opt]
    t = TRANSLATIONS[lang]
    
    st.markdown("---")
    st.markdown(f"### {t['sidebar_title']}")
    st.write(t['sidebar_desc'])
    st.markdown("---")
    
    # About Section in sidebar
    st.markdown(f"### {t['about_title']}")
    st.write(t['about_text'])
    
    # Model details
    st.markdown(f"### {t['model_details']}")
    st.info(t['model_details_text'])

# ---------------------------------------------------------
# 6. Hero Section
# ---------------------------------------------------------
st.markdown(f"# {t['title']}")
st.markdown(f"##### *{t['subtitle']}*")
st.markdown("---")

# ---------------------------------------------------------
# 7. Form Input Section
# ---------------------------------------------------------
st.markdown(f"### {t['input_section']}")

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input(
        t['n_label'],
        min_value=0.0,
        max_value=250.0,
        value=90.0,
        step=1.0,
        help=t['n_help']
    )
    temp = st.number_input(
        t['temp_label'],
        min_value=-10.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
        help=t['temp_help']
    )

with col2:
    P = st.number_input(
        t['p_label'],
        min_value=0.0,
        max_value=250.0,
        value=42.0,
        step=1.0,
        help=t['p_help']
    )
    hum = st.number_input(
        t['hum_label'],
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=0.1,
        help=t['hum_help']
    )

with col3:
    K = st.number_input(
        t['k_label'],
        min_value=0.0,
        max_value=250.0,
        value=43.0,
        step=1.0,
        help=t['k_help']
    )
    ph = st.number_input(
        t['ph_label'],
        min_value=0.0,
        max_value=14.0,
        value=6.5,
        step=0.1,
        help=t['ph_help']
    )

col_full, = st.columns(1)
with col_full:
    rain = st.number_input(
        t['rain_label'],
        min_value=0.0,
        max_value=1000.0,
        value=200.0,
        step=1.0,
        help=t['rain_help']
    )

# ---------------------------------------------------------
# 8. Prediction Logic & Output Styling
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button(t['btn_predict']):
    # Input validation
    if N < 0 or P < 0 or K < 0 or temp < -10.0 or hum < 0 or ph < 0 or rain < 0:
        st.error(t['error_validation'])
    else:
        with st.spinner(t['btn_predict_loading']):
            # Convert user inputs into a structured DataFrame (preserves feature names warning-free)
            input_df = pd.DataFrame([[N, P, K, temp, hum, ph, rain]], 
                                    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
            
            # Perform prediction using model (use .values to suppress feature names warning)
            pred_encoded = model.predict(input_df.values)
            
            # Map predictions to label name
            crop_raw = label_encoder.inverse_transform(pred_encoded)[0]
            
            # Look up localized crop name (fallback to raw if missing)
            crop_display = t['crop_names'].get(crop_raw, crop_raw.capitalize())
            
            # Celebratory effects
            st.balloons()
            
            # Render a glassmorphism success card
            st.markdown(f"""
            <div class="prediction-card">
                <h2>✨ {t['prediction_header']} ✨</h2>
                <p style="font-size: 1.1rem; color: #aaaaaa; margin-bottom: 5px;">{t['recommended_crop']}</p>
                <div class="crop-result">{crop_display}</div>
                <p style="font-size: 1.2rem; margin-top: 10px;">
                    {t['success_msg'].format(crop=crop_display)}
                </p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 9. Custom Footer
# ---------------------------------------------------------
st.markdown(f"""
<div class="custom-footer">
    {t['footer_text']}
</div>
""", unsafe_allow_html=True)