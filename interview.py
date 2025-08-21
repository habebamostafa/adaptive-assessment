# interview_simulator_streamlit.py
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --- إعداد الموديل ---
model_name = "google/flan-t5-large"

# استخدام التوكين من environment variable
token = True  # يستخدم HUGGINGFACEHUB_API_TOKEN

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=token)

def generate_question(track, difficulty, question_num):
    prompt = f"""
You are an AI interviewer for {track} field.
Generate interview question number {question_num} at {difficulty} difficulty.
Include the model answer.
Format:
Question:
Answer: [model answer]
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=250)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.strip()

def generate_interview(track, difficulty, num_questions):
    all_questions = []
    for i in range(1, num_questions+1):
        q = generate_question(track, difficulty, i)
        all_questions.append(q)
    return "\n\n".join(all_questions)

# --- واجهة Streamlit ---
st.title("AI-Powered Interview Simulation")

track = st.selectbox("Select Track:", ["AI", "Web", "Cyber", "Mobile", "Data"])
difficulty = st.selectbox("Select Difficulty:", ["easy", "medium", "hard"])
num_questions = st.number_input("Number of Questions:", min_value=1, max_value=10, value=3)

if st.button("Generate Interview Questions"):
    with st.spinner("Generating questions..."):
        interview_text = generate_interview(track, difficulty, num_questions)
    st.text_area("Interview Questions & Model Answers", interview_text, height=400)
