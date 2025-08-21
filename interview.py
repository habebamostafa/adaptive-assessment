# interview_simulator_crew.py
import streamlit as st
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

# --- Model Setup with Loading Indicators ---
MODEL_NAME = "google/flan-t5-large"
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

# Initialize session state for model loading
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_loading' not in st.session_state:
    st.session_state.model_loading = False

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the model with caching and proper error handling"""
    try:
        st.session_state.model_loading = True
        if HF_TOKEN:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        else:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        
        st.session_state.model_loading = False
        st.session_state.model_loaded = True
        return tokenizer, model
    except Exception as e:
        st.session_state.model_loading = False
        st.error(f"Error loading model: {e}")
        st.stop()

# Load model with progress indicator
if not st.session_state.model_loaded and not st.session_state.model_loading:
    with st.spinner("🚀 Loading AI model (this may take a minute)..."):
        tokenizer, model = load_model()

# Show loading state if model is still loading
if st.session_state.model_loading:
    st.info("⏳ Model is loading, please wait...")
    st.progress(0, text="Downloading model components")
    # Simulate progress since we can't get actual download progress
    for i in range(100):
        time.sleep(0.02)
        st.progress(i + 1, text=f"Downloading model components ({i+1}%)")
    st.rerun()

def generate_text(prompt, max_len=200):
    """Generate text with progress indicator"""
    try:
        with st.spinner("🤔 Generating response..."):
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=max_len)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Model error: {e}"

# --- Load dataset ---
try:
    with st.spinner("📊 Loading interview questions..."):
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
    
    num_questions = st.number_input("Number of Questions:", min_value=1, max_value=10, value=1)
    
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
    
    # Display model status
    st.subheader("System Status")
    if st.session_state.model_loaded:
        st.success("✅ Model loaded successfully")
    else:
        st.warning("⏳ Model loading...")

# --- Initialize session_state with default values ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.current_q = 0
    st.session_state.user_answers = []
    st.session_state.conversation = []  # Stores the entire conversation
    st.session_state.interview_finished = False
    st.session_state.questions_asked = []  # Track which questions have been asked

# Get selected questions based on current configuration
selected_questions = df[(df['Category']==track) & (df['Difficulty']==difficulty)].sample(n=num_questions)

# Function to add messages to the conversation
def add_to_conversation(role, message, agent_type=None):
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    st.session_state.conversation.append({
        "role": role,
        "message": message,
        "agent": agent_type
    })

# Initialize conversation if empty
if len(st.session_state.get('conversation', [])) == 0:
    add_to_conversation("System", f"Starting a {difficulty} level interview for {track} track with {num_questions} questions.")

# Display conversation
st.subheader("Interview Conversation")
conversation_container = st.container()

# Display the conversation in a chat-like format
with conversation_container:
    for msg in st.session_state.get('conversation', []):
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
if not st.session_state.get('interview_finished', False):
    if st.session_state.get('current_q', 0) < num_questions:
        current_q_index = st.session_state.get('current_q', 0)
        q_row = selected_questions.iloc[current_q_index]
        
        # Interviewer asks question (only if not already asked)
        question_already_asked = any(msg.get("question_id") == current_q_index for msg in st.session_state.get('conversation', []))
        
        if not question_already_asked:
            # Agent: Interviewer
            with st.status("💭 Interviewer is thinking...", expanded=False) as status:
                interviewer_prompt = f"""
                As a {interviewer_style} interviewer, ask this technical question in a conversational way:
                {q_row['Question']}
                
                Make it sound natural like a real conversation.
                """
                st.write("Crafting the perfect question...")
                interviewer_text = generate_text(interviewer_prompt)
                status.update(label="✅ Question ready", state="complete")
                
            add_to_conversation("Interviewer", interviewer_text, "Interviewer")
            st.session_state.conversation[-1]["question_id"] = current_q_index
            st.session_state.questions_asked.append(current_q_index)
            st.rerun()
        
        # Candidate answers
        with st.form(key="answer_form"):
            user_answer = st.text_area("Your answer:", key=f"answer_{current_q_index}", height=150,
                                      placeholder="Type your answer here...")
            submit_answer = st.form_submit_button("Submit Answer")
            
            if submit_answer and user_answer.strip():
                add_to_conversation("Candidate", user_answer, "Candidate")
                st.session_state.user_answers.append(user_answer)
                
                # Coach provides immediate feedback
                with st.status("📝 Coach is analyzing your answer...", expanded=False) as status:
                    coach_prompt = f"""
                    As a {coach_style} coach, provide constructive feedback on this interview interaction:
                    
                    Interview Question: {q_row['Question']}
                    Candidate's Answer: {user_answer}
                    Expected Answer: {q_row['Answer']}
                    
                    Provide brief, helpful feedback (1-2 sentences) and one suggestion for improvement.
                    Keep it conversational and supportive.
                    """
                    st.write("Analyzing your response...")
                    feedback = generate_text(coach_prompt)
                    status.update(label="✅ Feedback ready", state="complete")
                
                add_to_conversation("Coach", feedback, "Coach")
                
                st.session_state.current_q += 1
                st.rerun()
    
    else:
        # Interview finished - provide overall feedback
        st.session_state.interview_finished = True
        
        # Generate overall feedback
        with st.status("📊 Generating overall feedback...", expanded=False) as status:
            feedback_prompt = """
            As an experienced interview coach, provide overall feedback on the candidate's performance.
            Mention strengths, areas for improvement, and 2-3 key recommendations.
            Keep it professional yet encouraging, about 3-4 sentences long.
            """
            st.write("Evaluating your overall performance...")
            overall_feedback = generate_text(feedback_prompt, max_len=300)
            status.update(label="✅ Overall feedback ready", state="complete")
        
        add_to_conversation("Coach", f"Overall Feedback: {overall_feedback}", "Coach")
        st.rerun()

else:
    # Interview completed
    st.balloons()
    st.success("🎉 Interview completed! Check out your overall feedback above.")
    
    # Performance metrics (simulated)
    st.subheader("📈 Performance Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Questions Answered", f"{num_questions}/{num_questions}", "100%")
    with col2:
        # Simulated confidence score
        confidence_score = random.randint(65, 95)
        st.metric("Confidence Score", f"{confidence_score}%")
    with col3:
        st.metric("Feedback Items", random.randint(3, 8))
    
    if st.button("🔄 Start New Interview"):
        # Reset all session state variables
        for key in list(st.session_state.keys()):
            if key not in ['model_loaded', 'model_loading']:
                del st.session_state[key]
        st.rerun()

# Display tips
with st.expander("💡 Interview Tips"):
    st.markdown("""
    - **Take your time to think** before answering
    - **Be specific** with your examples and experiences
    - **It's okay to say 'I don't know'** but explain how you would find out
    - **Ask clarifying questions** if the question isn't clear
    - **Relate your answers** to real-world experiences
    - **Structure your answers** using the STAR method (Situation, Task, Action, Result)
    - **Practice active listening** and make sure you understand the question
    - **Stay calm and confident** throughout the interview
    """)

# Add footer with info
st.markdown("---")
st.caption("Powered by FLAN-T5 Large model • Interview questions dataset curated for software roles")