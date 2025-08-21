# interview_simulator_crew.py
import streamlit as st
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- إعداد الموديل FLAN-T5 ---
MODEL_NAME = "google/flan-t5-large"
HF_TOKEN = st.secrets.get("HF_TOKEN", None)  # Token مخفي في Streamlit Secrets

# تحميل الموديل مع معالجة حالة عدم وجود التوكن
try:
    if HF_TOKEN:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def generate_text(prompt, max_len=200):
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=max_len)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Model error: {e}"

# --- تحميل dataset ---
try:
    df = pd.read_csv("data/Software Questions.csv", encoding="latin-1")
except FileNotFoundError:
    st.error("CSV file not found. Please make sure 'data/Software Questions.csv' exists.")
    st.stop()

# --- Streamlit UI ---
st.title("🤖 AI-Powered Interview Simulation with CrewAI")
st.markdown("Experience a realistic interview with multiple AI agents (Interviewer, Coach, and you as Candidate)!")

# Sidebar for configuration
with st.sidebar:
    st.header("Interview Configuration")
    tracks = df['Category'].unique().tolist()
    track = st.selectbox("Select Track:", tracks)
    
    difficulties = df['Difficulty'].unique().tolist()
    difficulty = st.selectbox("Select Difficulty:", difficulties)
    
    num_questions = st.number_input("Number of Questions:", min_value=1, max_value=10, value=3)
    
    # Agent personality options
    st.subheader("Agent Personalities")
    interviewer_style = st.selectbox(
        "Interviewer Style:",
        ["Professional", "Friendly", "Technical", "Strict"]
    )
    
    coach_style = st.selectbox(
        "Coach Style:",
        ["Encouraging", "Constructive", "Direct", "Detailed"]
    )

# --- إعداد session_state ---
if 'current_q' not in st.session_state:
    st.session_state.current_q = 0
    st.session_state.user_answers = []
    st.session_state.conversation = []  # Stores the entire conversation
    st.session_state.selected_questions = df[(df['Category']==track) & (df['Difficulty']==difficulty)].sample(n=num_questions)
    st.session_state.interview_finished = False
    st.session_state.questions_asked = []  # Track which questions have been asked

# Function to add messages to the conversation
def add_to_conversation(role, message, agent_type=None):
    st.session_state.conversation.append({
        "role": role,
        "message": message,
        "agent": agent_type
    })

# Initialize conversation if empty
if len(st.session_state.conversation) == 0:
    add_to_conversation("System", f"Starting a {difficulty} level interview for {track} track with {num_questions} questions.")

# Display conversation
st.subheader("Interview Conversation")
conversation_container = st.container()

# Display the conversation in a chat-like format
with conversation_container:
    for msg in st.session_state.conversation:
        if msg["role"] == "System":
            with st.chat_message("system"):
                st.write(f"📢 {msg['message']}")
        elif msg["role"] == "Interviewer":
            with st.chat_message("assistant", avatar="👔"):
                st.write(f"**Interviewer** ({interviewer_style}): {msg['message']}")
        elif msg["role"] == "Coach":
            with st.chat_message("assistant", avatar="📊"):
                st.write(f"**Coach** ({coach_style}): {msg['message']}")
        elif msg["role"] == "Candidate":
            with st.chat_message("user", avatar="🧑‍💼"):
                st.write(f"**You**: {msg['message']}")

# Interview process
if not st.session_state.interview_finished:
    if st.session_state.current_q < num_questions:
        q_row = st.session_state.selected_questions.iloc[st.session_state.current_q]
        
        # Interviewer asks question (only if not already asked)
        question_already_asked = any(msg.get("question_id") == st.session_state.current_q for msg in st.session_state.conversation)
        
        if not question_already_asked:
            # Agent: Interviewer
            interviewer_prompt = f"""
            As a {interviewer_style} interviewer, ask this technical question in a conversational way:
            {q_row['Question']}
            
            Make it sound natural like a real conversation.
            """
            interviewer_text = generate_text(interviewer_prompt)
            add_to_conversation("Interviewer", interviewer_text, "Interviewer")
            st.session_state.conversation[-1]["question_id"] = st.session_state.current_q
            st.session_state.questions_asked.append(st.session_state.current_q)
            st.experimental_rerun()
        
        # Candidate answers
        with st.form(key="answer_form"):
            user_answer = st.text_area("Your answer:", key=f"answer_{st.session_state.current_q}", height=150)
            submit_answer = st.form_submit_button("Submit Answer")
            
            if submit_answer and user_answer.strip():
                add_to_conversation("Candidate", user_answer, "Candidate")
                st.session_state.user_answers.append(user_answer)
                
                # Coach provides immediate feedback
                coach_prompt = f"""
                As a {coach_style} coach, provide constructive feedback on this interview interaction:
                
                Interview Question: {q_row['Question']}
                Candidate's Answer: {user_answer}
                Expected Answer: {q_row['Answer']}
                
                Provide brief, helpful feedback (1-2 sentences) and one suggestion for improvement.
                Keep it conversational and supportive.
                """
                feedback = generate_text(coach_prompt)
                add_to_conversation("Coach", feedback, "Coach")
                
                st.session_state.current_q += 1
                st.experimental_rerun()
    
    else:
        # Interview finished - provide overall feedback
        st.session_state.interview_finished = True
        
        # Generate overall feedback
        feedback_prompt = """
        As an experienced interview coach, provide overall feedback on the candidate's performance.
        Mention strengths, areas for improvement, and 2-3 key recommendations.
        Keep it professional yet encouraging, about 3-4 sentences long.
        """
        
        overall_feedback = generate_text(feedback_prompt, max_len=300)
        add_to_conversation("Coach", f"Overall Feedback: {overall_feedback}", "Coach")
        st.experimental_rerun()

else:
    # Interview completed
    st.balloons()
    st.success("🎉 Interview completed! Check out your overall feedback above.")
    
    if st.button("Start New Interview"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

# Display tips
with st.expander("💡 Interview Tips"):
    st.markdown("""
    - Take your time to think before answering
    - Be specific with your examples
    - It's okay to say 'I don't know' but explain how you would find out
    - Ask clarifying questions if needed
    - Relate your answers to real-world experiences
    """)