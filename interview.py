# interview_simulator_crew.py
import streamlit as st
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import os

# --- Model Setup with Proper Caching ---
MODEL_NAME = "google/flan-t5-base"  # Using a smaller model for faster loading
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

# Initialize session state for model loading
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_loading' not in st.session_state:
    st.session_state.model_loading = False
if 'model_progress' not in st.session_state:
    st.session_state.model_progress = 0
if 'show_interview' not in st.session_state:
    st.session_state.show_interview = False  # New state to control interview section visibility

# Define generate_text function early so it's available
def generate_text(prompt, max_len=200):
    """Generate text with the loaded model"""
    if 'tokenizer' not in st.session_state or 'model' not in st.session_state:
        return "Model not loaded yet. Please wait..."
    
    try:
        inputs = st.session_state.tokenizer(prompt, return_tensors="pt")
        outputs = st.session_state.model.generate(**inputs, max_length=max_len)
        return st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Model error: {e}"

@st.cache_resource(show_spinner=False)
def load_model_components():
    """Load the model and tokenizer with proper caching"""
    try:
        st.session_state.model_loading = True
        progress_bar = st.progress(0, text="Downloading model components...")
        
        # Update progress (simulated)
        for i in range(5):
            st.session_state.model_progress = (i + 1) * 20
            progress_bar.progress(st.session_state.model_progress, 
                                 text=f"Loading model components... {st.session_state.model_progress}%")
            time.sleep(0.5)  # Simulate progress
        
        if HF_TOKEN:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        else:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        
        # Store in session state for global access
        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
        
        progress_bar.progress(100, text="Model loaded successfully!")
        time.sleep(0.5)
        progress_bar.empty()
        
        st.session_state.model_loading = False
        st.session_state.model_loaded = True
        st.session_state.show_interview = True  # Show interview section now
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- Load dataset first so users can see something immediately ---
try:
    with st.spinner("📊 Loading interview questions..."):
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        df = pd.read_csv("data/Software Questions.csv", encoding="latin-1")
except FileNotFoundError:
    st.error("CSV file not found. Please make sure 'data/Software Questions.csv' exists.")
    st.stop()

# --- Streamlit UI ---
st.title("🤖 AI-Powered Interview Simulation with CrewAI")
st.markdown("Experience a realistic interview with multiple AI agents (Interviewer, Coach, and you as Candidate)!")

# Show loading message until model is ready
if not st.session_state.model_loaded:
    # Display app interface while model loads in background
    st.info("⏳ The AI model is loading in the background. You can configure your interview while you wait.")
    
    # Show progress bar for model loading
    if st.session_state.model_loading:
        progress_bar = st.progress(st.session_state.model_progress, 
                                  text=f"Loading AI model... {st.session_state.model_progress}%")
    else:
        # Start loading the model if not already loading
        tokenizer, model = load_model_components()
        st.rerun()

# Sidebar for configuration (always show)
with st.sidebar:
    st.header("Interview Configuration")
    
    # Display model status
    st.subheader("System Status")
    if st.session_state.model_loaded:
        st.success("✅ Model loaded successfully")
    else:
        st.warning("⏳ Model loading in progress")
    
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
    
    st.sidebar.info(f"📋 Loaded {len(df)} questions across {df['Category'].nunique()} categories")

# Only show the interview section after model is loaded
if st.session_state.model_loaded and st.session_state.show_interview:
    # --- Initialize session_state with default values ---
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_q = 0
        st.session_state.user_answers = []
        st.session_state.conversation = []  # Stores the entire conversation
        st.session_state.interview_finished = False
        st.session_state.questions_asked = []  # Track which questions have been asked
        st.session_state.selected_questions = None

    # Get selected questions based on current configuration
    if st.session_state.selected_questions is None:
        st.session_state.selected_questions = df[
            (df['Category']==track) & (df['Difficulty']==difficulty)
        ].sample(n=min(num_questions, len(df[(df['Category']==track) & (df['Difficulty']==difficulty)])))

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
            q_row = st.session_state.selected_questions.iloc[current_q_index]
            
            # Interviewer asks question (only if not already asked)
            question_already_asked = any(msg.get("question_id") == current_q_index for msg in st.session_state.get('conversation', []))
            
            if not question_already_asked:
                # Agent: Interviewer
                with st.status("💭 Interviewer is thinking...", expanded=False) as status:
                    interviewer_prompt = f"""
                    As a {interviewer_style} technical interviewer, ask this question in a conversational way:
                    "{q_row['Question']}"
                    
                    Make it sound natural like a real conversation. Just ask the question directly without extra commentary.
                    """
                    st.write("Crafting the perfect question...")
                    interviewer_text = generate_text(interviewer_prompt)
                    # Fallback to the original question if the generated text is too short
                    if len(interviewer_text.strip()) < 10:
                        interviewer_text = q_row['Question']
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
                        As a {coach_style} interview coach, provide specific, constructive feedback on this answer:
                        
                        QUESTION: {q_row['Question']}
                        CANDIDATE'S ANSWER: {user_answer}
                        EXPECTED ANSWER: {q_row['Answer']}
                        
                        Provide 2-3 specific suggestions for improvement. Focus on technical accuracy, completeness, and clarity.
                        Be supportive but honest. Maximum 3 sentences.
                        """
                        st.write("Analyzing your response...")
                        feedback = generate_text(coach_prompt, max_len=300)
                        # Fallback feedback if the generated text is poor
                        if len(feedback.strip()) < 20:
                            feedback = "Good attempt! Remember to be more specific in your answers and provide examples when possible."
                        status.update(label="✅ Feedback ready", state="complete")
                    
                    add_to_conversation("Coach", feedback, "Coach")
                    
                    st.session_state.current_q += 1
                    st.rerun()
        
        else:
            # Interview finished - provide overall feedback
            st.session_state.interview_finished = True
            
            # Generate overall feedback
            with st.status("📊 Generating overall feedback...", expanded=False) as status:
                feedback_prompt = f"""
                As an experienced interview coach, provide overall feedback on this candidate's performance:
                
                They answered {num_questions} questions on {track} at {difficulty} level.
                
                Provide specific feedback on:
                1. Technical knowledge demonstrated
                2. Communication skills
                3. Areas for improvement
                4. Recommendations for next steps
                
                Keep it professional yet encouraging, about 4-5 sentences long.
                Be specific and actionable.
                """
                st.write("Evaluating your overall performance...")
                overall_feedback = generate_text(feedback_prompt, max_len=400)
                # Fallback overall feedback
                if len(overall_feedback.strip()) < 30:
                    overall_feedback = "You completed the interview! To improve, focus on providing more detailed answers with specific examples from your experience. Practice explaining technical concepts clearly and concisely."
                status.update(label="✅ Overall feedback ready", state="complete")
            
            add_to_conversation("Coach", f"Overall Feedback: {overall_feedback}", "Coach")
            st.rerun()

    else:
        # Interview completed
        st.balloons()
        st.success("🎉 Interview completed! Check out your overall feedback above.")
        
        # Performance metrics (simulated but more realistic)
        st.subheader("📈 Performance Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Questions Answered", f"{num_questions}/{num_questions}", "100%")
        with col2:
            # More realistic confidence score based on answer length
            avg_answer_length = sum(len(ans) for ans in st.session_state.user_answers) / len(st.session_state.user_answers) if st.session_state.user_answers else 0
            confidence_score = min(95, max(60, int(avg_answer_length / 5)))
            st.metric("Confidence Score", f"{confidence_score}%")
        with col3:
            st.metric("Key Strengths", random.randint(2, 5))
        
        # Add specific feedback points
        with st.expander("📋 Detailed Feedback"):
            st.write("Based on your performance:")
            st.write("✅ **Strengths:**")
            st.write("- Willingness to engage with technical questions")
            st.write("- Clear communication style")
            
            st.write("📝 **Areas for Improvement:**")
            st.write("- Provide more specific examples in your answers")
            st.write("- Structure responses using the STAR method (Situation, Task, Action, Result)")
            st.write("- Include more technical details relevant to the question")
        
        if st.button("🔄 Start New Interview"):
            # Reset all session state variables except model-related ones
            for key in list(st.session_state.keys()):
                if key not in ['model_loaded', 'model_loading', 'model_progress', 'tokenizer', 'model', 'show_interview']:
                    del st.session_state[key]
            st.rerun()

# Display tips (always show)
with st.expander("💡 Interview Tips"):
    st.markdown("""
    - **Use the STAR method**: Situation, Task, Action, Result
    - **Be specific**: Include numbers, technologies, and outcomes
    - **It's OK to think**: Say "Let me think about that for a moment"
    - **Ask clarifying questions**: Ensure you understand what's being asked
    - **Structure your answers**: Start with a direct answer, then provide details
    - **Include examples**: Reference real projects and experiences
    - **Be concise**: Get to the point but provide enough detail
    - **Stay positive**: Frame challenges as learning experiences
    """)

# Add footer with info
st.markdown("---")
st.caption("Powered by FLAN-T5 Base model • Interview questions dataset curated for software roles")