# interview_simulator_crew.py
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- إعداد الموديل FLAN-T5 ---
MODEL_NAME = "google/flan-t5-large"
HF_TOKEN = st.secrets["HF_TOKEN"]  # Token مخفي في Streamlit Secrets

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)

def generate_text(prompt, max_len=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_len)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- تحميل dataset ---
df = pd.read_csv("data/Software Questions.csv")

# --- Streamlit UI ---
st.title("AI-Powered Interview Simulation with Crew")

tracks = df['Category'].unique().tolist()
track = st.selectbox("Select Track:", tracks)

difficulties = df['Difficulty'].unique().tolist()
difficulty = st.selectbox("Select Difficulty:", difficulties)

num_questions = st.number_input("Number of Questions:", min_value=1, max_value=10, value=3)

# --- إعداد session_state ---
if 'current_q' not in st.session_state:
    st.session_state.current_q = 0
    st.session_state.user_answers = []
    st.session_state.selected_questions = df[(df['Category']==track) & (df['Difficulty']==difficulty)].sample(n=num_questions)
    st.session_state.chat_history = []  # حفظ الحوار بين Agents

# --- عرض السؤال الحالي ---
if st.session_state.current_q < num_questions:
    q_row = st.session_state.selected_questions.iloc[st.session_state.current_q]
    
    # Agent: Interviewer
    interviewer_prompt = f"Interviewer: Ask this question to the candidate:\n{q_row['Question']}"
    interviewer_text = generate_text(interviewer_prompt)
    st.subheader(f"Question {st.session_state.current_q+1} (Interviewer):")
    st.write(interviewer_text)
    
    user_answer = st.text_area("Candidate Answer:", key=f"answer_{st.session_state.current_q}")
    
    if st.button("Submit Answer"):
        st.session_state.user_answers.append(user_answer)
        # Agent: Coach feedback
        coach_prompt = f"""
Candidate answered: {user_answer}
Correct answer: {q_row['Answer']}
Coach: Give concise feedback and 1-2 tips for improvement.
"""
        feedback = generate_text(coach_prompt)
        st.session_state.chat_history.append({
            "question": interviewer_text,
            "candidate": user_answer,
            "coach_feedback": feedback
        })
        st.session_state.current_q += 1
        st.experimental_rerun()

# --- عرض النتائج بعد نهاية الأسئلة ---
else:
    st.subheader("Interview Completed! Feedback from Coach:")
    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1} (Interviewer):** {chat['question']}")
        st.markdown(f"**Your Answer (Candidate):** {chat['candidate']}")
        st.markdown(f"**Feedback (Coach):** {chat['coach_feedback']}")
