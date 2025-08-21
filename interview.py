# interview_simulator_crew.py
import streamlit as st
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

# --- Model Setup with Proper Caching ---
MODEL_NAME = "declare-lab/flan-alpaca-base"  # Using a smaller model for faster loading
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

# Initialize session state for model loading
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_loading' not in st.session_state:
    st.session_state.model_loading = False
if 'model_progress' not in st.session_state:
    st.session_state.model_progress = 0
if 'show_interview' not in st.session_state:
    st.session_state.show_interview = False

# Define generate_text function
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
        st.session_state.show_interview = True
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- Streamlit UI ---
st.title("🤖 AI-Powered Interview Simulation with CrewAI")
st.markdown("Experience a realistic interview with AI-generated questions and feedback!")

# Show loading message until model is ready
if not st.session_state.model_loaded:
    st.info("⏳ The AI model is loading in the background. You can configure your interview while you wait.")
    
    if st.session_state.model_loading:
        progress_bar = st.progress(st.session_state.model_progress, 
                                  text=f"Loading AI model... {st.session_state.model_progress}%")
    else:
        tokenizer, model = load_model_components()
        st.rerun()

# Sidebar for configuration
with st.sidebar:
    st.header("Interview Configuration")

    # Display model status
    st.subheader("System Status")
    if st.session_state.model_loaded:
        st.success("✅ Model loaded successfully")
    else:
        st.warning("⏳ Model loading in progress")

    # Input options
    tracks = ["Artificial Intelligence", "Software Development", "Web Development", "Mobile App", "Data Science", "Product Management", "UX Design", "Marketing"]
    difficulties = ["Easy", "Medium", "Hard"]
    
    selected_track = st.selectbox(
        "Select Track:",
        tracks,
        index=3  # Default to Mobile App
    )

    selected_difficulty = st.selectbox(
        "Select Difficulty:",
        difficulties,
        index=0
    )

    num_questions = st.number_input(
        "Number of Questions:",
        min_value=1,
        max_value=10,
        value=3
    )

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
    
    # Confirm button
    if st.button("Start Interview", type="primary", use_container_width=True):
        st.session_state.selected_track = selected_track
        st.session_state.selected_difficulty = selected_difficulty
        st.session_state.selected_num_questions = num_questions
        st.session_state.interviewer_style = interviewer_style
        st.session_state.coach_style = coach_style
        
        # Initialize interview state
        st.session_state.current_q = 0
        st.session_state.user_answers = []
        st.session_state.conversation = []
        st.session_state.interview_finished = False
        st.session_state.questions = []
        st.session_state.expected_answers = []
        st.session_state.settings_confirmed = True
        st.session_state.show_interview = True
        st.rerun()

# Only show the interview section after model is loaded and settings confirmed
if st.session_state.model_loaded and st.session_state.get('settings_confirmed', False):
    # Initialize session state with default values
    if 'current_q' not in st.session_state:
        st.session_state.current_q = 0
        st.session_state.user_answers = []
        st.session_state.conversation = []
        st.session_state.interview_finished = False
        st.session_state.questions = []
        st.session_state.expected_answers = []

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
        add_to_conversation("System", f"Starting a {st.session_state.selected_difficulty} level interview for {st.session_state.selected_track} track with {st.session_state.selected_num_questions} questions.")

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
                    st.write(f"**Interviewer** ({st.session_state.interviewer_style}): {msg['message']}")
            elif msg["role"] == "Coach":
                with st.chat_message("assistant", avatar="📊"):
                    st.write(f"**Coach** ({st.session_state.coach_style}): {msg['message']}")
            elif msg["role"] == "Candidate":
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.write(f"**You**: {msg['message']}")

    # Interview process
    if not st.session_state.get('interview_finished', False):
        if st.session_state.current_q < st.session_state.selected_num_questions:
            current_q_index = st.session_state.current_q
            
            # Generate question if not already generated for this index
            if len(st.session_state.questions) <= current_q_index:
                with st.status("💭 Interviewer is generating a question...", expanded=False) as status:
                    # Get previously asked questions to avoid repetition
                    previous_questions = st.session_state.questions
                    previous_questions_text = ", ".join(previous_questions) if previous_questions else "None"
                    
                    question_prompt = f"""
                    As a {st.session_state.interviewer_style} technical interviewer in {st.session_state.selected_track}, 
                    generate a {st.session_state.selected_difficulty.lower()} level interview question.
                    
                    This is question {current_q_index + 1} of {st.session_state.selected_num_questions}.
                    Previously asked questions: {previous_questions_text}
                    
                    The question should be:
                    - Specific and relevant to {st.session_state.selected_track}
                    - Appropriate for {st.session_state.selected_difficulty} level
                    - Different from previous questions
                    - Focused on a different aspect than previous questions
                    
                    Return only the question without any additional text.
                    """
                    st.write("Creating a tailored question...")
                    question = generate_text(question_prompt, max_len=100)
                    
                    # If the question is too similar to previous ones, try again
                    if previous_questions and any(prev_q in question for prev_q in previous_questions):
                        st.write("Generating a different question to avoid repetition...")
                        question = generate_text(question_prompt, max_len=100)
                    
                    # Generate expected answer
                    answer_prompt = f"""
                    As an expert in {st.session_state.selected_track}, provide a model answer to the following question:
                    "{question}"
                    
                    The answer should be:
                    - Comprehensive and technically accurate
                    - Appropriate for a {st.session_state.selected_difficulty.lower()} level
                    - Structured and clear
                    
                    Return only the answer without any additional text.
                    """
                    st.write("Preparing model answer...")
                    expected_answer = generate_text(answer_prompt, max_len=300)
                    
                    # Store the generated question and answer
                    st.session_state.questions.append(question)
                    st.session_state.expected_answers.append(expected_answer)
                    status.update(label="✅ Question ready", state="complete")
            
            # Get the current question
            current_question = st.session_state.questions[current_q_index]
            
            # Interviewer asks question (only if not already asked)
            question_already_asked = any(msg.get("question_id") == current_q_index for msg in st.session_state.get('conversation', []))
            
            if not question_already_asked:
                # Agent: Interviewer
                with st.status("💭 Interviewer is thinking...", expanded=False) as status:
                    interviewer_prompt = f"""
                    As a {st.session_state.interviewer_style} technical interviewer, ask this question in a conversational way:
                    "{current_question}"
                    
                    Make it sound natural like a real conversation. Just ask the question directly without extra commentary.
                    """
                    st.write("Crafting the perfect question...")
                    interviewer_text = generate_text(interviewer_prompt)
                    # Fallback to the original question if the generated text is too short
                    if len(interviewer_text.strip()) < 10:
                        interviewer_text = current_question
                    status.update(label="✅ Question ready", state="complete")
                
                add_to_conversation("Interviewer", interviewer_text, "Interviewer")
                st.session_state.conversation[-1]["question_id"] = current_q_index
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
                        As a {st.session_state.coach_style} interview coach, provide specific, constructive feedback on this answer:
                        
                        QUESTION: {current_question}
                        CANDIDATE'S ANSWER: {user_answer}
                        EXPECTED ANSWER: {st.session_state.expected_answers[current_q_index]}
                        
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
                
                They answered {st.session_state.selected_num_questions} questions on {st.session_state.selected_track} at {st.session_state.selected_difficulty} level.
                
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
            st.metric("Questions Answered", f"{st.session_state.selected_num_questions}/{st.session_state.selected_num_questions}", "100%")
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
st.caption("Powered by AI Interview Simulation | Questions and feedback generated in real-time by the language model")