# Humanoid Mental Health Interface
**Advanced Understanding & Response Agent**

> *"More than a journal. A companion that understands."*

AURA is a diverse, emotionally intelligent **Humanoid AI** designed to provide empathetic companionship and mental health support. Unlike standard chatbots, AURA possesses specific "cognitive" modules for emotional recognition, crisis safety, and long-term memory analysis.

## 🧠 Humanoid Architecture

### 1. Emotional Cortex (Emotion Detection)
AURA doesn't just read text; it *feels* the sentiment behind it. Using advanced Transformer models, it detects 6 distinct emotional states:
- **Joy** 🌟
- **Sadness** 💧
- **Anger** 💢
- **Fear** 😨
- **Surprise** 😲
- **Neutral** 😐

### 2. Safety Protocol Implementation (Crisis Override)
AURA is programmed with a "First Law" of safety. Its **Crisis Detection Module** runs in parallel to every interaction. If high-risk patterns are detected, AURA suppresses standard conversational outputs and engages a specialized **Crisis Intervention Protocol**, providing immediate, verified resources.

### 3. Adaptive Personality Matrix (Dynamic Response)
AURA avoids robotic repetition. Its **Response Generation Engine** pulls from a vast database of empathetic, psychologically grounded templates (`prompt_templates.json`), allowing it to adapt its tone from celebratory to soothing based on the user's emotional state.

### 4. Memory & Insight System (Live Analytics)
AURA remembers. The **Life Analytics Dashboard** acts as its memory visualizer, tracking:
- **Mood Trends**: Are you feeling better than yesterday?
- **Emotional Distribution**: What is your dominant emotion this week?
- **Stability Metrics**: How volatile are your stress levels?

## 🚀 Initialize AURA

### Prerequisites
- Python 3.12+
- A compatible GPU (optional, but recommended for faster cognition)

### Installation
Initialize the neural dependencies:
```bash
pip install -r requirements.txt
```

### Activation
Wake AURA up by running:
```bash
python -m streamlit run "d:/Internship 3 month_unpaid/Text Emotion Classification Model/app.py"
```
*Access the interface at `http://localhost:8501`*

## 📂 System Files
- **`app.py`**: The Central Nervous System (Main Logic).
- **`emotion_model/`**: The Emotional Processing Unit.
- **`Crisis Model/`**: The Safety Override Module.
- **`safety_guidelines.json`**: The Core Ethics Directives.
- **`insights.json`**: The Cognitive Pattern Database.

## 🛡️ Operational Disclaimer
AURA is an Artificial Intelligence, not a human doctor. It provides support based on patterns and data. **In a medical emergency, always contact biological emergency services.**

Note: Trained model weights are not included in this repository due to GitHub file size limitations.
