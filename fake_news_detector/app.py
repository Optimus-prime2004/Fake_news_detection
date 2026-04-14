import streamlit as st
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from datetime import datetime
import os

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Fake News AI Pro",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "bert_model")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CLEAN FUNCTION ----------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

# ---------------- PREDICTION FUNCTION ----------------
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    score, pred = torch.max(probs, dim=1)

    return pred.item(), score.item()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Model Dashboard")

    st.metric("Accuracy", "92.0%")
    st.metric("Precision", "93.1%")
    st.metric("Recall", "90.9%")
    st.metric("F1 Score", "92.0%")

    st.divider()

    st.write("### 📌 Features")
    st.write("✔ Real-time prediction")
    st.write("✔ Confidence graph")
    st.write("✔ History tracking")
    st.write("✔ Download results")

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;'>📰 Fake News Detector AI Pro</h1>
<p style='text-align:center;color:gray;'>Stable Cloud Version with Advanced UI</p>
""", unsafe_allow_html=True)

st.divider()

# ---------------- INPUT ----------------
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("📝 Enter News Content", height=220)

with col2:
    st.info("💡 Tip: Use complete sentences for better prediction")

# ---------------- ANALYZE ----------------
if st.button("🔍 Analyze News", use_container_width=True):

    if not user_input.strip():
        st.warning("⚠️ Please enter some text")
    else:
        with st.spinner("🤖 AI analyzing..."):

            cleaned = clean_text(user_input)

            pred, score = predict(cleaned[:512])
            confidence = score * 100
            is_real = pred == 0

            # ---------------- STORE HISTORY ----------------
            st.session_state.history.append({
                "text": user_input[:100],
                "result": "REAL" if is_real else "FAKE",
                "confidence": round(confidence, 2),
                "time": datetime.now().strftime("%H:%M:%S")
            })

            st.divider()

            # ---------------- RESULT ----------------
            if is_real:
                st.success("✅ REAL NEWS")
            else:
                st.error("🚨 FAKE NEWS")

            # ---------------- METRICS ----------------
            colA, colB, colC = st.columns(3)

            colA.metric("Confidence", f"{confidence:.2f}%")
            colB.metric("Risk", "Low" if is_real else "High")
            colC.metric("Model", "BERT")

            # ---------------- GRAPH ----------------
            st.markdown("### 📊 Confidence Visualization")

            chart_df = pd.DataFrame({
                "Category": ["Confidence", "Uncertainty"],
                "Value": [confidence, 100 - confidence]
            })

            st.bar_chart(chart_df.set_index("Category"))

            # ---------------- PROGRESS ----------------
            st.progress(confidence / 100)

            # ---------------- TEXT PREVIEW ----------------
            st.markdown("### 🧾 Processed Text")
            st.caption(cleaned[:300] + "...")

            # ---------------- AI INSIGHT ----------------
            st.markdown("### 🧠 AI Insight")

            if is_real:
                st.info("This content follows reliable reporting patterns.")
            else:
                st.warning("This content shows patterns of misinformation.")

            # ---------------- LOW CONFIDENCE ----------------
            if confidence < 70:
                st.warning("⚠️ Low confidence — verify manually")

# ---------------- HISTORY ----------------
st.divider()
st.markdown("### 📜 Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

    # ---------------- DOWNLOAD ----------------
    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download History",
        csv,
        "prediction_history.csv",
        "text/csv"
    )
else:
    st.info("No history yet")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 AI Fake News Detection System | Optimized for Streamlit Cloud")