# pyre-ignore-all-errors
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Page Configuration
st.set_page_config(
    page_title="Multi-Cancer Detection System",
    page_icon="🔬",
    layout="wide"
)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skin_cancer_model.h5')

@st.cache_resource
def load_detection_model():
    if os.path.exists(MODEL_PATH):
        try:
            # Workaround for quantization_config error in saved Dense layers
            def dummy_dense(**kwargs):
                kwargs.pop('quantization_config', None)
                return tf.keras.layers.Dense(**kwargs)
            return tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={'Dense': dummy_dense},
                compile=False
            )
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning(f"Model file not found: {MODEL_PATH}")
        return None

model = load_detection_model()

# Class labels matching the Colab HAM10000 training
CLASS_NAMES = [
    'Actinic Keratosis',       # akiec (0)
    'Basal Cell Carcinoma',     # bcc   (1)
    'Benign Keratosis',         # bkl   (2)
    'Dermatofibroma',           # df    (3)
    'Melanoma',                 # mel   (4)
    'Melanocytic Nevi',         # nv    (5)
    'Vascular Lesion'           # vasc  (6)
]
IMG_SIZE = 64  # Must match Colab training size

# Title & Tagline
st.markdown("<h1 style='text-align: center; font-size: 3.5em; margin-bottom: 0;'>Skin Cancer Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.4em; color: gray; margin-top: 10px;'>AI-powered skin lesion classification using HAM10000 dataset.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Custom CSS for the Chatbot Experience
st.markdown("""
<style>
    /* Main Input Container */
    .stTextArea > div {
        border-radius: 30px !important;
        background-color: white !important;
        border: 1px solid #e0e0e0 !important;
        padding-left: 60px !important; /* Space for the paperclip */
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
        transition: height 0.2s ease;
        position: relative;
    }
    /* Integrated Search Button */
    .stButton {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .main-search-row .stButton > button {
        border-radius: 50% !important;
        width: 50px !important;
        height: 50px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: white !important;
        border: 1px solid #e0e0e0 !important;
        transition: all 0.3s ease;
        margin-bottom: 0 !important;
        font-size: 24px !important;
        margin-left: -15px !important; /* Move button further right */
    }
    .main-search-row .stButton > button:hover {
        background-color: #e9ecef !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }

    /* Blue styled primary buttons for results */
    div.stButton > button[kind="primary"] {
        background-color: #007bff !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0069d9 !important;
        border: none !important;
    }

    /* Centered Icon Overlay */
    .stFileUploader {
        display: flex !important;
        justify-content: center !important;
        width: 50px !important;
        height: 50px !important;
        position: absolute !important;
        left: 15px !important;
        top: 21px !important;
        z-index: 1000 !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
    }
    .stFileUploader section {
        padding: 0 !important;
        background-color: transparent !important;
        border: none !important;
        min-height: unset !important;
        width: 50px !important;
        cursor: pointer !important;
    }
    .stFileUploader section:hover {
        background-color: transparent !important;
    }
    .stFileUploader section > div {
        display: none !important;
    }
    .stFileUploader button {
        display: none !important;
    }
    .stFileUploader section::before {
        content: "📎";
        font-size: 30px;
        cursor: pointer;
        display: block;
        opacity: 0.7;
    }
    .stFileUploader section:hover::before {
        opacity: 1;
    }
    
    /* Layout Centering */
    .main-search-row {
        max-width: 900px;
        margin: 0 auto;
        position: relative;
    }
</style>

<script>
try {
    const textArea = window.parent.document.querySelector('textarea[aria-label=""]');
    if (textArea) {
        const updateHeight = () => {
            textArea.style.height = 'auto';
            textArea.style.height = textArea.scrollHeight + 'px';
        };
        updateHeight();
        textArea.addEventListener('input', updateHeight);
    }
} catch (e) {
    console.error("Custom auto-resize script error:", e);
}
</script>
""", unsafe_allow_html=True)

# Layout: Integrated Chatbot Bar
st.markdown("<div class='main-search-row'>", unsafe_allow_html=True)

# Use 2 columns: [Input | Search]
col_input, col_btn = st.columns([15, 1], gap="small", vertical_alignment="center")

with col_input:
    # Attach icon inside the input box, perfectly centered on left
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "pdf"], key="doc_upload", label_visibility="collapsed")
    user_description = st.text_area("", placeholder="Type a message...", key="search_input", height=70, label_visibility="collapsed")

with col_btn:
    search_clicked = st.button("🔍")

st.markdown("</div>", unsafe_allow_html=True)

# Display Results and Processing
if search_clicked or (uploaded_file is not None and user_description):
    if not uploaded_file and not user_description:
        st.warning("Please provide a medical description or attach an image.")
    else:
        st.divider()
        col_img, col_info = st.columns([1, 1])
        
        with col_img:
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption='Attached Document/Image', use_container_width=True)
            else:
                st.info("No image attached. Analyzing clinical description only...")
                image = None
            
        with col_info:
            if user_description:
                st.subheader("User Description")
                st.info(user_description)
            
            # Preprocess & Prediction logic
            st.subheader("Diagnostic Engine")
            with st.status("Reviewing medical data...", expanded=True) as status:
                # Preprocess
                if image:
                    img = image.resize((IMG_SIZE, IMG_SIZE))
                    img_array = np.array(img.convert('RGB')) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                else:
                    img_array = None
                
                if model and img_array is not None:
                    prediction = model.predict(img_array)
                    
                    pred_idx = np.argmax(prediction[0])
                    confidence = prediction[0][pred_idx]
                    status.update(label="Analysis Complete", state="complete", expanded=False)
                    
                    # Output results
                    st.markdown(f"**Diagnosis:** {CLASS_NAMES[pred_idx]}")
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    
                    # Show all class probabilities
                    st.markdown("**All Probabilities:**")
                    for i, name in enumerate(CLASS_NAMES):
                        prob = prediction[0][i]
                        st.progress(float(prob), text=f"{name}: {prob:.2%}")
                    
                    st.write("Does this result match the patient's record?")
                    
                    btn_col1, btn_col2, btn_spacer = st.columns([1, 1, 4])
                    with btn_col1:
                        st.button("Yes", use_container_width=True, key="btn_yes", type="primary")
                    with btn_col2:
                        st.button("No", use_container_width=True, key="btn_no", type="primary")
                elif model and img_array is None:
                    st.warning("Model loaded, but no clinical image provided for visual analysis.")
                    status.update(label="Incomplete Data", state="error", expanded=True)
                else:
                    # 2-line error message with buttons as requested
                    st.error("**Error:** AI Diagnostic Model failed to load.")
                    st.write("Would you like to try reloading the system configuration?")
                    
                    err_col1, err_col2, err_spacer = st.columns([1, 1, 4])
                    with err_col1:
                        if st.button("Yes", use_container_width=True, key="err_yes"):
                            st.rerun()
                    with err_col2:
                        st.button("No", use_container_width=True, key="err_no")
                    
                    status.update(label="Critical System Error", state="error", expanded=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.caption("Disclaimer: This tool is for educational purposes only and is not a substitute for professional medical advice.")
