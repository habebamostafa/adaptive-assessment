import streamlit as st
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

# --- Model Setup with Proper Caching ---
MODEL_NAME = "google/flan-t5-small"  # Better model for text generation
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

# Define generate_text function with better parameters
def generate_text(prompt, max_len=150, temperature=0.7):
    """Generate text with the loaded model"""
    if 'tokenizer' not in st.session_state or 'model' not in st.session_state:
        return "Model not loaded yet. Please wait..."
    
    try:
        inputs = st.session_state.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512,
            truncation=True
        )
        
        outputs = st.session_state.model.generate(
            **inputs, 
            max_length=max_len,
            min_length=20,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            no_repeat_ngram_size=2,
            pad_token_id=st.session_state.tokenizer.eos_token_id
        )
        
        generated_text = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        # Ensure we have meaningful content
        if len(generated_text.strip()) < 10:
            return get_fallback_response(prompt)
        
        return generated_text.strip()
    
    except Exception as e:
        st.error(f"Model error: {e}")
        return get_fallback_response(prompt)

def get_fallback_response(prompt):
    """Provide fallback responses when model fails"""
    if "question" in prompt.lower():
        questions = {
            "Artificial Intelligence": [
                "What is machine learning and how does it differ from traditional programming?",
                "Explain the difference between supervised and unsupervised learning.",
                "What are neural networks and how do they work?"
            ],
            "Software Development": [
                "What is the difference between object-oriented and functional programming?",
                "Explain the concept of version control and why it's important.",
                "What are design patterns and can you give an example?"
            ],
            "Web Development": [
                "What is the difference between frontend and backend development?",
                "Explain how HTTP works and the difference between GET and POST requests.",
                "What is responsive design and why is it important?"
            ],
            "Data Science": [
                "What is the difference between correlation and causation?",
                "Explain the steps in a typical data science project.",
                "What is overfitting and how can you prevent it?"
            ]
        }
        track = st.session_state.get('selected_track', 'Software Development')
        return random.choice(questions.get(track, questions['Software Development']))
    
    elif "feedback" in prompt.lower():
        feedbacks = [
            "Good start! Try to be more specific and provide concrete examples in your answer.",
            "Your answer shows understanding. Consider elaborating with real-world applications.",
            "Well thought out response. Adding technical details would strengthen your answer.",
            "Nice explanation! Try to structure your answer with clear points next time."
        ]
        return random.choice(feedbacks)
    
    else:
        return "Thank you for your response. Let's continue with the next question."

@st.cache_resource(show_spinner=False)
def load_model_components():
    """Load the model and tokenizer with proper caching"""
    try:
        st.session_state.model_loading = True
        progress_bar = st.progress(0, text="Downloading model components...")
        
        # Update progress
        for i in range(5):
            st.session_state.model_progress = (i + 1) * 20
            progress_bar.progress(
                st.session_state.model_progress, 
                text=f"Loading model components... {st.session_state.model_progress}%"
            )
            time.sleep(0.3)
        
        if HF_TOKEN:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        else:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        
        # Handle tokenizer padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
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
        st.info("Using fallback responses for the interview.")
        st.session_state.model_loaded = True  # Allow interview to proceed with fallbacks
        st.session_state.model_loading = False
        st.session_state.show_interview = True
        return None, None

# --- Streamlit UI ---
st.title("🤖 AI-Powered Interview Simulation")
st.markdown("Experience a realistic interview with AI-generated questions and feedback!")

# Show loading message until model is ready
if not st.session_state.model_loaded:
    st.info("⏳ The AI model is loading. You can configure your interview while you wait.")
    
    if st.session_state.model_loading:
        progress_bar = st.progress(
            st.session_state.model_progress, 
            text=f"Loading AI model... {st.session_state.model_progress}%"
        )
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
    tracks = [
        "Artificial Intelligence", 
        "Software Development", 
        "Web Development", 
        "Mobile App Development", 
        "Data Science", 
        "Product Management", 
        "UX Design", 
        "Digital Marketing"
    ]
    difficulties = ["Easy", "Medium", "Hard"]
    
    selected_track = st.selectbox(
        "Select Track:",
        tracks,
        index=0
    )

    selected_difficulty = st.selectbox(
        "Select Difficulty:",
        difficulties,
        index=0
    )

    num_questions = st.number_input(
        "Number of Questions:",
        min_value=1,
        max_value=8,
        value=3
    )

    # Agent personality options
    st.subheader("Interview Style")
    interviewer_style = st.selectbox(
        "Interviewer Style:",
        ["Professional", "Friendly", "Technical", "Conversational"]
    )
    
    coach_style = st.selectbox(
        "Coach Style:",
        ["Encouraging", "Constructive", "Direct", "Detailed"]
    )
    
    # Start interview button
    if st.button("Start Interview", type="primary", use_container_width=True):
        # Store configuration
        st.session_state.selected_track = selected_track
        st.session_state.selected_difficulty = selected_difficulty
        st.session_state.selected_num_questions = num_questions
        st.session_state.interviewer_style = interviewer_style
        st.session_state.coach_style = coach_style
        
        # Reset interview state
        st.session_state.current_q = 0
        st.session_state.user_answers = []
        st.session_state.conversation = []
        st.session_state.interview_finished = False
        st.session_state.questions = []
        st.session_state.expected_answers = []
        st.session_state.settings_confirmed = True
        st.session_state.show_interview = True
        st.rerun()

# Only show interview section when ready
if st.session_state.model_loaded and st.session_state.get('settings_confirmed', False):
    
    # Initialize session state variables
    if 'current_q' not in st.session_state:
        st.session_state.current_q = 0
        st.session_state.user_answers = []
        st.session_state.conversation = []
        st.session_state.interview_finished = False
        st.session_state.questions = []
        st.session_state.expected_answers = []

    def add_to_conversation(role, message, agent_type=None):
        """Add message to conversation history"""
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
        st.session_state.conversation.append({
            "role": role,
            "message": message,
            "agent": agent_type,
            "timestamp": time.time()
        })

    # Initialize conversation
    if len(st.session_state.get('conversation', [])) == 0:
        welcome_msg = f"Welcome! Starting your {st.session_state.selected_difficulty.lower()} level interview for {st.session_state.selected_track}. We'll go through {st.session_state.selected_num_questions} questions together."
        add_to_conversation("System", welcome_msg)

    # Display conversation
    st.subheader("Interview in Progress")
    
    # Show progress
    progress = st.session_state.current_q / st.session_state.selected_num_questions
    st.progress(progress, text=f"Question {st.session_state.current_q + 1} of {st.session_state.selected_num_questions}")

    # Conversation display
    conversation_container = st.container()
    with conversation_container:
        for msg in st.session_state.get('conversation', []):
            if msg["role"] == "System":
                with st.chat_message("system"):
                    st.info(f"📢 {msg['message']}")
            elif msg["role"] == "Interviewer":
                with st.chat_message("assistant", avatar="👔"):
                    st.write(f"**Interviewer**: {msg['message']}")
            elif msg["role"] == "Coach":
                with st.chat_message("assistant", avatar="📊"):
                    st.write(f"**Coach**: {msg['message']}")
            elif msg["role"] == "Candidate":
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.write(f"**You**: {msg['message']}")

    # Interview logic
    if not st.session_state.get('interview_finished', False):
        if st.session_state.current_q < st.session_state.selected_num_questions:
            current_q_index = st.session_state.current_q
            
            # Generate question if needed
            if len(st.session_state.questions) <= current_q_index:
                with st.status("💭 Preparing next question...", expanded=False) as status:
                    question_prompt = f"""Generate a {st.session_state.selected_difficulty.lower()} level interview question for {st.session_state.selected_track}. 
                    Question {current_q_index + 1}: Ask about a core concept or skill.
                    Make it specific and practical. Question only:"""
                    
                    st.write("Crafting your question...")
                    question = generate_text(question_prompt, max_len=100)
                    
                    # Generate expected answer for scoring
                    answer_prompt = f"""Provide a comprehensive answer to: {question}
                    Include key points and examples. Answer:"""
                    
                    st.write("Preparing evaluation criteria...")
                    expected_answer = generate_text(answer_prompt, max_len=200)
                    
                    st.session_state.questions.append(question)
                    st.session_state.expected_answers.append(expected_answer)
                    status.update(label="✅ Question ready", state="complete")

            # Ask question if not already asked
            current_question = st.session_state.questions[current_q_index]
            question_asked = any(
                msg.get("question_id") == current_q_index 
                for msg in st.session_state.get('conversation', [])
            )
            
            if not question_asked:
                add_to_conversation("Interviewer", current_question, "Interviewer")
                st.session_state.conversation[-1]["question_id"] = current_q_index
                st.rerun()
            
            # Answer form
            with st.form(key=f"answer_form_{current_q_index}"):
                user_answer = st.text_area(
                    "Your answer:", 
                    key=f"answer_{current_q_index}", 
                    height=120,
                    placeholder="Take your time to provide a detailed answer..."
                )
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption("💡 Tip: Use specific examples and explain your reasoning")
                with col2:
                    submit_answer = st.form_submit_button("Submit Answer", type="primary")
                
                if submit_answer and user_answer.strip():
                    # Add user answer to conversation
                    add_to_conversation("Candidate", user_answer, "Candidate")
                    st.session_state.user_answers.append(user_answer)
                    
                    # Generate coach feedback
                    with st.status("📝 Coach analyzing your response...", expanded=False) as status:
                        feedback_prompt = f"""As an {st.session_state.coach_style.lower()} interview coach, provide brief constructive feedback.
                        Question: {current_question}
                        Answer: {user_answer}
                        
                        Give 2-3 specific improvement suggestions. Be supportive. Feedback:"""
                        
                        st.write("Evaluating your response...")
                        feedback = generate_text(feedback_prompt, max_len=150)
                        
                        status.update(label="✅ Feedback ready", state="complete")
                    
                    add_to_conversation("Coach", feedback, "Coach")
                    st.session_state.current_q += 1
                    st.rerun()

        else:
            # Interview completed
            st.session_state.interview_finished = True
            
            # Generate final feedback
            with st.status("📊 Preparing your final evaluation...", expanded=False) as status:
                final_prompt = f"""Provide overall interview feedback for {st.session_state.selected_track} candidate.
                They completed {st.session_state.selected_num_questions} questions at {st.session_state.selected_difficulty} level.
                
                Summarize: strengths, areas for improvement, and next steps. 
                Keep it professional and encouraging. Overall assessment:"""
                
                st.write("Compiling your performance summary...")
                overall_feedback = generate_text(final_prompt, max_len=200)
                status.update(label="✅ Evaluation complete", state="complete")
            
            add_to_conversation("Coach", f"🎯 **Final Assessment**: {overall_feedback}", "Coach")
            st.rerun()

    else:
        # Show completion
        st.success("🎉 Interview completed successfully!")
        
        # Performance summary
        st.subheader("📈 Your Performance Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Questions Completed", 
                f"{st.session_state.selected_num_questions}/{st.session_state.selected_num_questions}",
                "100%"
            )
        
        with col2:
            # Calculate engagement score based on answer length
            if st.session_state.user_answers:
                avg_length = sum(len(ans.split()) for ans in st.session_state.user_answers) / len(st.session_state.user_answers)
                engagement_score = min(95, max(65, int(avg_length * 2)))
            else:
                engagement_score = 70
            st.metric("Engagement Score", f"{engagement_score}%")
        
        with col3:
            improvement_areas = random.randint(2, 4)
            st.metric("Growth Areas Identified", improvement_areas)
        
        # Detailed feedback section
        with st.expander("📋 Detailed Performance Analysis"):
            st.markdown("### Strengths Observed:")
            st.write("✅ Active participation in the interview process")
            st.write("✅ Willingness to engage with technical questions")
            if any(len(ans.split()) > 30 for ans in st.session_state.user_answers):
                st.write("✅ Provided detailed responses to questions")
            
            st.markdown("### Recommendations for Improvement:")
            st.write("📈 Practice the STAR method (Situation, Task, Action, Result)")
            st.write("📈 Include more specific examples from your experience")
            st.write("📈 Focus on explaining your thought process clearly")
            
            st.markdown("### Next Steps:")
            st.write(f"🎯 Continue practicing {st.session_state.selected_track} concepts")
            st.write("🎯 Work on structuring your responses more effectively")
            st.write("🎯 Practice explaining complex topics in simple terms")
        
        # Reset button
        if st.button("🔄 Start New Interview", type="primary", use_container_width=True):
            # Clear interview-related session state
            keys_to_keep = ['model_loaded', 'model_loading', 'model_progress', 'tokenizer', 'model']
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            st.rerun()

# Interview tips (always visible)
with st.expander("💡 Interview Success Tips"):
    st.markdown("""
    **Before You Start:**
    - Take a moment to understand each question fully
    - Think about real examples from your experience
    - Structure your thoughts before answering
    
    **During the Interview:**
    - Use the STAR method: Situation, Task, Action, Result
    - Be specific with numbers, technologies, and outcomes
    - It's okay to pause and think before answering
    - Ask for clarification if a question is unclear
    
    **Answer Structure:**
    - Start with a direct answer to the question
    - Provide context and background
    - Explain your approach or methodology
    - Share the outcome and what you learned
    
    **Common Mistakes to Avoid:**
    - Being too vague or general in your responses
    - Not providing concrete examples
    - Rushing through your answers
    - Forgetting to explain your reasoning process
    """)

# Footer
st.markdown("---")
st.caption("🤖 AI-Powered Interview Simulator | Realistic practice with instant feedback")