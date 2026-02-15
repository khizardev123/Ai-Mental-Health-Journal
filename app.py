import streamlit as st
import torch
import json
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_all_resources():
    """
    Load all models, tokenizers, and configuration JSONs.
    """
    emotion_path = "./emotion_model"
    crisis_path = "./Crisis Model" 
    
    # Load Models
    emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_path)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_path)
    emotion_model.eval()
    
    crisis_tokenizer = AutoTokenizer.from_pretrained(crisis_path)
    crisis_model = AutoModelForSequenceClassification.from_pretrained(crisis_path)
    crisis_model.eval()
    
    # Load JSON Files
    with open(os.path.join(emotion_path, "config.json"), 'r') as f:
        emotion_config = json.load(f)
    emotion_labels = {int(k): v for k, v in emotion_config["id2label"].items()}
    
    with open("prompt_templates.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
        
    with open("safety_guidelines.json", "r", encoding="utf-8") as f:
        safety = json.load(f)
        
    with open("insights.json", "r", encoding="utf-8") as f:
        insights = json.load(f)

    with open("dashboard_data.json", "r", encoding="utf-8") as f:
        historical_dashboard = json.load(f)

    return {
        "emotion_tokenizer": emotion_tokenizer,
        "emotion_model": emotion_model,
        "crisis_tokenizer": crisis_tokenizer,
        "crisis_model": crisis_model,
        "emotion_labels": emotion_labels,
        "prompts": prompts,
        "safety": safety,
        "insights": insights,
        "historical_dashboard": historical_dashboard
    }

data = load_all_resources()

if "history" not in st.session_state:
    st.session_state.history = []
    hist_ctx = data["historical_dashboard"].get("weekly_summary", {})
    st.session_state.historical_summary = hist_ctx

st.set_page_config(page_title="AI Mental Health Journal", layout="wide", page_icon="🧠")

# Sidebar: Safety Policy
with st.sidebar:
    st.header("⚖️ Safety & Policy")
    st.info(f"Version: {data['safety']['version']} | Updated: {data['safety']['last_updated']}")
    
    with st.expander("Core Principles"):
        for p in data['safety']['core_principles']:
            st.write(f"- {p}")
            
    st.warning("**Disclaimers:**")
    st.write(data['safety']['general_disclaimer'])
    
    if st.button("View Crisis Resources"):
        st.error(data['safety']['crisis_disclaimer'])

# Main UI
st.title("📖 AI Mental Health Journal")
st.markdown("---")

# Prediction Logic
def predict_emotion(text):
    inputs = data["emotion_tokenizer"](text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = data["emotion_model"](**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    idx = torch.argmax(probs).item()
    return data["emotion_labels"][idx], probs[idx].item()

def predict_crisis(text, threshold=0.7):
    inputs = data["crisis_tokenizer"](text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = data["crisis_model"](**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    crisis_prob = probs[1].item()
    return crisis_prob >= threshold, crisis_prob

def check_safety(response_text):
    issues = []
    lower_res = response_text.lower()
    for phrase in data['safety']['prohibited_phrases']:
        if phrase.lower() in lower_res:
            issues.append(phrase)
    return issues

# ==========================================
# UI Section: Journal Entry
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 New Journal Entry")
    user_input = st.text_area("Share your thoughts...", height=180, placeholder="Example: I'm feeling a bit overwhelmed today, but I'm trying to stay positive.")
    analyze_btn = st.button("Analyze & Respond", type="primary")

    if analyze_btn and user_input:
        with st.spinner("Analyzing emotional context..."):
            emotion, e_conf = predict_emotion(user_input)
            is_crisis, c_prob = predict_crisis(user_input)
            
            # Crisis Logic Override
            if is_crisis:
                response_key = "high_distress"
                risk_level = "High Risk"
            else:
                response_key = emotion
                risk_level = "Medium Risk" if c_prob > 0.4 else "Low Risk"
            
            template = data["prompts"].get(response_key, data["prompts"]["neutral"])
            ai_response = random.choice(template["responses"])
            
            # Safety Check
            safety_issues = check_safety(ai_response)
            if safety_issues:
                ai_response = "I hear you. Let's focus on how you're feeling right now." # Fallback if template fails safety
            
            # Save to session history
            entry_data = {
                "timestamp": datetime.now(),
                "text": user_input,
                "emotion": emotion,
                "emotion_conf": e_conf,
                "is_crisis": is_crisis,
                "crisis_prob": c_prob,
                "risk_level": risk_level,
                "ai_response": ai_response
            }
            st.session_state.history.append(entry_data)
            
            # Display Result
            st.markdown("---")
            st.success(f"Analysis for: *\"{user_input[:50]}...\"*")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Emotion", emotion.capitalize(), f"{e_conf:.0%}")
            metric_col2.metric("Risk Level", risk_level)
            metric_col3.metric("Crisis Prob", f"{c_prob:.2f}")

            st.chat_message("assistant").write(ai_response)
            
            if is_crisis:
                st.error("🚨 **Urgent Help Resources:**")
                st.write(data['safety']['crisis_disclaimer'])

# Dashboard Section (Integrated Data)
with col2:
    st.subheader("📊 Life Analytics")
    
    # Initialize tabs (Always visible)
    tabs = st.tabs(["Trends", "Insights", "Distribution", "Training Metrics"])
    
    history_df = pd.DataFrame(st.session_state.history) if st.session_state.history else pd.DataFrame()

    with tabs[0]:
        st.write("**Mood Stability (Crisis Prob)**")
        if not history_df.empty:
            st.line_chart(history_df.set_index("timestamp")["crisis_prob"])
        else:
            st.info("No new entries yet. Showing historical trend.")
            trend_val = data["historical_dashboard"].get("chart_data", {}).get("trend", data["historical_dashboard"].get("weekly_summary", {}).get("trend", "N/A"))
            st.metric("Historical Trend", trend_val.capitalize())

    with tabs[1]:
        st.write("**Recent Observations**")
        for ins in data["insights"]:
            emoji = ins.get("emoji", "ℹ️") 
            with st.expander(f"{emoji} {ins['title']}"):
                st.write(ins['message'])

    with tabs[2]:
        st.write("**Emotion Breakdown**")
        
        if not history_df.empty:
            counts = history_df["emotion"].value_counts()
        else:
            st.caption("Showing historical weekly data")
            hist_counts = data["historical_dashboard"].get("weekly_summary", {}).get("emotion_counts", {})
            counts = pd.Series(hist_counts)
        
        if not counts.empty:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=plt.cm.Set3.colors)
            st.pyplot(fig)
        else:
            st.write("No data available.")

    # --- TAB 4: TRAINING METRICS ---
    with tabs[3]:
        st.write("**Model Intelligence Overview**")
        if os.path.exists("emotion_charts.png"):
            st.image("emotion_charts.png", caption="Training Metrics & Confusion Matrix")
        else:
            st.info("Training metrics image not found.")
            
    # Summary Box
    st.markdown("---")
    if not history_df.empty:
        st.write(f"**Total Entries Session:** {len(history_df)}")
    else:
        st.write("**Session Status:** Ready for your first entry.")

st.markdown("---")
st.caption("AI Mental Health Journal Prototype | Integrating logic from safety_guidelines.json, insights.json, and prompt_templates.json")
