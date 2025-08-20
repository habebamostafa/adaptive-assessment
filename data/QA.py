"""
Enhanced MCQ Generator using Google's FLAN-T5 with Hugging Face Token
Fixed and Improved Version
"""

import streamlit as st
import requests
import json
import random
import time
from typing import List, Dict, Optional
import re

# Try importing Hugging Face Hub
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class SimpleMCQGenerator:
    def __init__(self):
        """Initialize the MCQ generator"""
        self.available_tracks = {
            "web": "Web Development (HTML, CSS, JavaScript, React, Vue, Angular)",
            "ai": "Artificial Intelligence (Machine Learning, Deep Learning, NLP)",
            "cyber": "Cybersecurity (Network Security, Encryption, Ethical Hacking)",
            "data": "Data Science (Data Analysis, Visualization, Statistics, Python)",
            "mobile": "Mobile Development (Android, iOS, Flutter, React Native)",
            "devops": "DevOps (Docker, Kubernetes, CI/CD, Cloud Computing)",
            "backend": "Backend Development (APIs, Databases, Microservices)",
            "frontend": "Frontend Development (UI/UX, Responsive Design)"
        }
        
        # Initialize Hugging Face client
        self.hf_token = None
        self.client = None
        self._setup_hugging_face()
    
    def _setup_hugging_face(self):
        """Setup Hugging Face connection with better error handling"""
        if not HF_AVAILABLE:
            st.warning("⚠️ Hugging Face Hub not available. Install with: pip install huggingface-hub")
            return
            
        try:
            # Get token from multiple sources
            self.hf_token = None
            
            # Try Streamlit secrets first
            if hasattr(st, 'secrets'):
                try:
                    self.hf_token = st.secrets.get("HF_TOKEN") or st.secrets.get("hf_token")
                except:
                    pass
            
            # Try environment variable as backup
            if not self.hf_token:
                import os
                self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            
            if self.hf_token:
                # Test the token by creating a client
                self.client = InferenceClient(token=self.hf_token)
                # Try a simple API call to validate
                try:
                    # Test with a lightweight model first
                    test_response = self.client.text_generation(
                        model="google/flan-t5-base",  # Using base model for testing
                        prompt="Test",
                        max_new_tokens=5
                    )
                    st.success("✅ Hugging Face connected and validated!")
                except Exception as test_error:
                    st.warning(f"⚠️ Token found but API test failed: {str(test_error)[:100]}...")
                    # Keep the client anyway, might work for actual requests
            else:
                st.info("ℹ️ No Hugging Face token found. Using demo mode.")
                st.info("💡 To enable AI generation, add your HF token to Streamlit secrets or environment variables")
                
        except Exception as e:
            st.error(f"❌ Hugging Face setup failed: {str(e)[:200]}...")
            self.client = None
    
    def get_available_tracks(self) -> List[str]:
        """Get list of available technology tracks"""
        return list(self.available_tracks.keys())
    
    def get_demo_question(self, track: str, difficulty: str) -> Dict:
        """Get a demo question for the specified track and difficulty"""
        demo_questions = {
            "web": {
                "easy": {
                    "question": "What does HTML stand for?",
                    "options": [
                        "HyperText Markup Language",
                        "High Tech Modern Language", 
                        "Home Tool Markup Language",
                        "Hyperlink Text Management Language"
                    ],
                    "correct_answer": "HyperText Markup Language",
                    "explanation": "HTML stands for HyperText Markup Language, the standard markup language for creating web pages and web applications."
                },
                "medium": {
                    "question": "What is the primary benefit of React's Virtual DOM?",
                    "options": [
                        "Optimizes rendering by minimizing DOM operations",
                        "Provides direct database connectivity",
                        "Automatically generates CSS styles",
                        "Enables server-side rendering only"
                    ],
                    "correct_answer": "Optimizes rendering by minimizing DOM operations",
                    "explanation": "Virtual DOM creates a virtual representation of the UI in memory and efficiently updates only the parts of the real DOM that have changed, improving performance."
                },
                "hard": {
                    "question": "How does webpack's tree shaking optimize bundle size?",
                    "options": [
                        "Eliminates unused code through static analysis",
                        "Compresses images automatically",
                        "Minifies CSS files only",
                        "Removes HTML comments"
                    ],
                    "correct_answer": "Eliminates unused code through static analysis",
                    "explanation": "Tree shaking removes dead code by analyzing ES6 module imports/exports during build time, eliminating unused functions and variables."
                }
            },
            "ai": {
                "easy": {
                    "question": "What is the main goal of machine learning?",
                    "options": [
                        "Enable computers to learn from data automatically",
                        "Create human-like robots only",
                        "Replace all human workers",
                        "Process images exclusively"
                    ],
                    "correct_answer": "Enable computers to learn from data automatically",
                    "explanation": "Machine learning enables computers to automatically learn and improve their performance on tasks through experience with data, without being explicitly programmed for each specific task."
                },
                "medium": {
                    "question": "What distinguishes supervised from unsupervised learning?",
                    "options": [
                        "Supervised uses labeled data, unsupervised finds patterns in unlabeled data",
                        "Supervised is always faster than unsupervised",
                        "Unsupervised requires constant human oversight",
                        "Supervised only works with image data"
                    ],
                    "correct_answer": "Supervised uses labeled data, unsupervised finds patterns in unlabeled data",
                    "explanation": "Supervised learning learns from labeled training examples to make predictions, while unsupervised learning discovers hidden patterns and structures in unlabeled data."
                },
                "hard": {
                    "question": "What is the key innovation of Transformer architecture in deep learning?",
                    "options": [
                        "Self-attention mechanism for parallel sequence processing",
                        "Convolutional layers optimized for text processing",
                        "Recurrent connections for enhanced memory",
                        "Direct reinforcement learning integration"
                    ],
                    "correct_answer": "Self-attention mechanism for parallel sequence processing",
                    "explanation": "Transformers introduced the self-attention mechanism that allows the model to process all positions in a sequence simultaneously, enabling better parallelization and long-range dependency modeling."
                }
            },
            "cyber": {
                "easy": {
                    "question": "What is a firewall's primary function in network security?",
                    "options": [
                        "Control network traffic based on security rules",
                        "Encrypt all hard drive data automatically",
                        "Prevent physical access to devices",
                        "Create automated data backups"
                    ],
                    "correct_answer": "Control network traffic based on security rules",
                    "explanation": "A firewall monitors and filters incoming and outgoing network traffic based on predetermined security rules to protect against unauthorized access."
                },
                "medium": {
                    "question": "What's the main difference between symmetric and asymmetric encryption?",
                    "options": [
                        "Symmetric uses one key, asymmetric uses public/private key pairs",
                        "Symmetric encryption is always more secure",
                        "Asymmetric encryption is only used for SSL/TLS",
                        "Symmetric encryption only works with text data"
                    ],
                    "correct_answer": "Symmetric uses one key, asymmetric uses public/private key pairs",
                    "explanation": "Symmetric encryption uses the same key for both encryption and decryption, while asymmetric encryption uses mathematically related public and private key pairs."
                },
                "hard": {
                    "question": "What makes zero-day vulnerabilities particularly dangerous in cybersecurity?",
                    "options": [
                        "They're unknown to vendors with no available patches",
                        "They only affect newly released systems",
                        "They can only be exploited during midnight hours",
                        "They require no user interaction to exploit"
                    ],
                    "correct_answer": "They're unknown to vendors with no available patches",
                    "explanation": "Zero-day vulnerabilities are security flaws that are unknown to software vendors and security teams, making them particularly dangerous as there are no patches or defenses available."
                }
            },
            "data": {
                "easy": {
                    "question": "What is the primary purpose of data visualization?",
                    "options": [
                        "Make data insights easier to understand and communicate",
                        "Reduce the size of datasets automatically",
                        "Encrypt sensitive information in databases",
                        "Replace the need for statistical analysis"
                    ],
                    "correct_answer": "Make data insights easier to understand and communicate",
                    "explanation": "Data visualization transforms complex data into visual formats like charts and graphs to make patterns, trends, and insights more accessible and understandable."
                },
                "medium": {
                    "question": "When should you use a median instead of a mean for central tendency?",
                    "options": [
                        "When the data has outliers or is skewed",
                        "When working with categorical data only",
                        "When the dataset is very small",
                        "When all values are the same"
                    ],
                    "correct_answer": "When the data has outliers or is skewed",
                    "explanation": "Median is more robust to outliers and provides a better representation of central tendency when data is skewed, as it's not affected by extreme values."
                },
                "hard": {
                    "question": "What is the curse of dimensionality in machine learning?",
                    "options": [
                        "Performance degrades as feature dimensions increase exponentially",
                        "Models can only handle 2D data effectively",
                        "Training time increases linearly with features",
                        "Memory usage decreases with more dimensions"
                    ],
                    "correct_answer": "Performance degrades as feature dimensions increase exponentially",
                    "explanation": "The curse of dimensionality refers to various phenomena that arise when analyzing data in high-dimensional spaces, where distance metrics become less meaningful and data becomes sparse."
                }
            }
        }
        
        # Add more tracks with demo questions
        default_questions = {
            "mobile": {
                "question": "What is the main advantage of React Native for mobile development?",
                "options": [
                    "Write once, run on both iOS and Android",
                    "Automatic app store deployment",
                    "Built-in payment processing",
                    "No coding required"
                ],
                "correct_answer": "Write once, run on both iOS and Android",
                "explanation": "React Native allows developers to use a single codebase to create apps for both iOS and Android platforms."
            },
            "devops": {
                "question": "What is the main purpose of Docker containers?",
                "options": [
                    "Package applications with their dependencies for consistent deployment",
                    "Only for web application development",
                    "Replace traditional databases",
                    "Automatic code generation"
                ],
                "correct_answer": "Package applications with their dependencies for consistent deployment",
                "explanation": "Docker containers package applications with all their dependencies to ensure consistent behavior across different environments."
            }
        }
        
        # Get the question or return a track-specific one
        if track in demo_questions and difficulty in demo_questions[track]:
            demo = demo_questions[track][difficulty]
            return {
                'text': demo['question'],
                'options': demo['options'],
                'correct_answer': demo['correct_answer'],
                'explanation': demo['explanation'],
                'track': track,
                'difficulty': difficulty,
                'generated_by': 'Demo Mode'
            }
        elif track in default_questions:
            demo = default_questions[track]
            return {
                'text': demo['question'],
                'options': demo['options'],
                'correct_answer': demo['correct_answer'],
                'explanation': demo['explanation'],
                'track': track,
                'difficulty': difficulty,
                'generated_by': 'Demo Mode'
            }
        else:
            return {
                'text': f"What is an important concept in {self.available_tracks[track]}?",
                'options': [
                    "The correct answer for this topic",
                    "An incorrect but plausible option",
                    "Another incorrect alternative",
                    "A clearly wrong choice"
                ],
                'correct_answer': "The correct answer for this topic",
                'explanation': f"This tests {difficulty} level knowledge of {self.available_tracks[track]}.",
                'track': track,
                'difficulty': difficulty,
                'generated_by': 'Generic Demo'
            }
    
    def generate_with_ai(self, track: str, difficulty: str) -> Dict:
        """Generate question using AI (FLAN-T5) with improved error handling"""
        if not self.client or not self.hf_token:
            st.warning("🤖 AI generation not available, using demo question")
            return self.get_demo_question(track, difficulty)
        
        try:
            prompt = self._create_prompt(track, difficulty)
            
            # Try multiple models in order of preference
            models_to_try = [
                "google/flan-t5-large",
                "google/flan-t5-base",
                "microsoft/DialoGPT-medium"
            ]
            
            response_text = None
            used_model = None
            
            for model in models_to_try:
                try:
                    st.info(f"🔄 Trying model: {model}")
                    
                    response = self.client.text_generation(
                        model=model,
                        prompt=prompt,
                        max_new_tokens=500,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1
                    )
                    
                    # Handle different response formats
                    if isinstance(response, str):
                        response_text = response
                    elif isinstance(response, dict) and "generated_text" in response:
                        response_text = response["generated_text"]
                    elif isinstance(response, list) and len(response) > 0:
                        if isinstance(response[0], dict) and "generated_text" in response[0]:
                            response_text = response[0]["generated_text"]
                        elif isinstance(response[0], str):
                            response_text = response[0]
                    
                    if response_text:
                        used_model = model
                        break
                        
                except Exception as model_error:
                    st.warning(f"⚠️ Model {model} failed: {str(model_error)[:100]}...")
                    continue
            
            if response_text:
                st.success(f"✅ Successfully generated with {used_model}")
                return self._parse_response(response_text, track, difficulty, used_model)
            else:
                st.error("❌ All AI models failed, using demo question")
                return self.get_demo_question(track, difficulty)
            
        except Exception as e:
            st.error(f"❌ AI generation error: {str(e)[:200]}...")
            return self.get_demo_question(track, difficulty)
    
    def _create_prompt(self, track: str, difficulty: str) -> str:
        """Create an improved prompt for FLAN-T5"""
        track_desc = self.available_tracks[track]
        
        # More specific prompts based on difficulty
        difficulty_guidelines = {
            "easy": "fundamental concepts, basic definitions, introductory knowledge",
            "medium": "practical applications, intermediate concepts, problem-solving",
            "hard": "advanced topics, complex scenarios, expert-level understanding"
        }
        
        prompt = f"""Generate a {difficulty} level multiple choice question about {track_desc}.

The question should test {difficulty_guidelines[difficulty]}.

Requirements:
- Clear, professional question suitable for technical interviews
- Exactly 4 answer options labeled A, B, C, D
- Only one correct answer
- Plausible but incorrect distractors
- Brief explanation of the correct answer

Format your response as valid JSON:
{{
    "question": "Your question text here",
    "options": {{
        "A": "First option",
        "B": "Second option",
        "C": "Third option",
        "D": "Fourth option"
    }},
    "correct_answer": "A",
    "explanation": "Brief explanation of why this is correct"
}}

Topic: {track_desc}
Difficulty: {difficulty}
Focus: {difficulty_guidelines[difficulty]}

Generate the JSON response now:"""
        
        return prompt
    
    def _parse_response(self, response: str, track: str, difficulty: str, model_name: str = "AI") -> Dict:
        """Enhanced response parsing with better error handling"""
        try:
            # Clean the response
            response = response.strip()
            
            # Try to find JSON in the response
            json_pattern = r'\{.*\}'
            json_matches = re.findall(json_pattern, response, re.DOTALL)
            
            data = None
            if json_matches:
                # Try each JSON match
                for json_match in json_matches:
                    try:
                        data = json.loads(json_match)
                        if "question" in data and "options" in data:
                            break
                    except:
                        continue
            
            if not data:
                # Try parsing the entire response as JSON
                try:
                    data = json.loads(response)
                except:
                    pass
            
            if data and isinstance(data, dict):
                # Extract and validate data
                question_text = data.get("question", f"What is important in {track}?")
                options_dict = data.get("options", {})
                correct_key = data.get("correct_answer", "A")
                explanation = data.get("explanation", f"This tests {difficulty} {track} knowledge.")
                
                # Convert options to list
                if isinstance(options_dict, dict):
                    options_list = [
                        options_dict.get("A", "Option A"),
                        options_dict.get("B", "Option B"),
                        options_dict.get("C", "Option C"),
                        options_dict.get("D", "Option D")
                    ]
                    correct_answer_text = options_dict.get(correct_key, options_list[0])
                else:
                    options_list = ["Option A", "Option B", "Option C", "Option D"]
                    correct_answer_text = "Option A"
                
                return {
                    'text': question_text,
                    'options': options_list,
                    'correct_answer': correct_answer_text,
                    'explanation': explanation,
                    'track': track,
                    'difficulty': difficulty,
                    'generated_by': f'{model_name} AI'
                }
            else:
                raise ValueError("Could not parse AI response")
                
        except Exception as e:
            st.warning(f"⚠️ Parsing failed ({str(e)[:50]}...), using demo question")
            return self.get_demo_question(track, difficulty)
    
    def generate_question_set(self, track: str, num_questions: int, difficulty: str, use_ai: bool = True) -> List[Dict]:
        """Generate a set of questions with improved progress tracking"""
        questions = []
        
        # Create progress tracking
        progress_container = st.empty()
        
        for i in range(num_questions):
            # Update progress
            progress = (i + 1) / num_questions
            progress_container.progress(progress, text=f"Generating question {i + 1} of {num_questions}...")
            
            if use_ai and self.client and self.hf_token:
                question = self.generate_with_ai(track, difficulty)
            else:
                question = self.get_demo_question(track, difficulty)
            
            questions.append(question)
            
            # Small delay to prevent rate limiting
            if use_ai and i < num_questions - 1:
                time.sleep(1)
        
        # Clear progress
        progress_container.empty()
        return questions

def main():
    """Main Streamlit app with improved UI"""
    st.set_page_config(
        page_title="MCQ Generator Pro",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .question-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
    .correct-answer {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 MCQ Generator Pro</h1>
        <p>Generate professional technical interview questions using AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize generator
    if 'generator' not in st.session_state:
        with st.spinner("Initializing MCQ Generator..."):
            st.session_state.generator = SimpleMCQGenerator()
    
    generator = st.session_state.generator
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Connection status
        if generator.hf_token and generator.client:
            st.success("✅ AI Mode Active")
            st.info("🤖 Using Hugging Face API")
        else:
            st.warning("📋 Demo Mode Active")
            st.info("💡 Add HF_TOKEN to secrets for AI generation")
        
        st.divider()
        
        # Settings
        use_ai = st.checkbox(
            "🤖 Use AI Generation", 
            value=bool(generator.hf_token and generator.client),
            disabled=not (generator.hf_token and generator.client),
            help="Enable AI-powered question generation"
        )
        
        # Show token status
        if generator.hf_token:
            masked_token = f"{generator.hf_token[:8]}{'*' * 20}"
            st.caption(f"Token: {masked_token}")
    
    # Main interface
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📚 Question Configuration")
        
        # Track selection with better formatting
        track = st.selectbox(
            "🎯 Technology Track:",
            options=generator.get_available_tracks(),
            format_func=lambda x: f"{x.upper()} - {generator.available_tracks[x].split('(')[0].strip()}",
            help="Select the technology domain for questions"
        )
        
        # Show track description
        st.info(f"📖 **Focus:** {generator.available_tracks[track]}")
        
        # Difficulty and number in columns
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            difficulty = st.selectbox(
                "📊 Difficulty Level:", 
                options=["easy", "medium", "hard"], 
                index=1,
                help="Select question difficulty level"
            )
        with col1_2:
            num_questions = st.number_input(
                "📝 Number of Questions:", 
                min_value=1, 
                max_value=10, 
                value=3,
                help="How many questions to generate"
            )
    
    with col2:
        st.subheader("📊 Generation Summary")
        
        # Summary cards
        summary_data = [
            ("🎯 Track", track.upper()),
            ("📊 Level", difficulty.title()),
            ("📝 Count", str(num_questions)),
            ("🤖 Mode", "AI" if use_ai else "Demo")
        ]
        
        for label, value in summary_data:
            if "AI" in value:
                st.success(f"**{label}:** {value}")
            elif "Demo" in value:
                st.warning(f"**{label}:** {value}")
            else:
                st.info(f"**{label}:** {value}")
        
        # Estimated time
        est_time = num_questions * (3 if use_ai else 1)
        st.caption(f"⏱️ Estimated time: ~{est_time} seconds")
    
    st.divider()
    
    # Generate button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🚀 Generate Questions", type="primary", use_container_width=True):
            with st.spinner("🎯 Generating questions..."):
                try:
                    questions = generator.generate_question_set(
                        track=track,
                        num_questions=num_questions, 
                        difficulty=difficulty,
                        use_ai=use_ai
                    )
                    
                    # Store questions in session state
                    st.session_state.questions = questions
                    st.session_state.generation_info = {
                        'track': track,
                        'difficulty': difficulty,
                        'count': len(questions),
                        'mode': 'AI' if use_ai else 'Demo',
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.success(f"✅ Successfully generated {len(questions)} questions!")
                    
                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")
    
    # Display questions
    if 'questions' in st.session_state and st.session_state.questions:
        st.divider()
        
        # Questions header with info
        col_header1, col_header2 = st.columns([2, 1])
        with col_header1:
            st.header("📝 Generated Questions")
        with col_header2:
            if 'generation_info' in st.session_state:
                info = st.session_state.generation_info
                st.caption(f"Generated: {info['timestamp']}")
                st.caption(f"Mode: {info['mode']} | Track: {info['track'].upper()}")
        
        # Display each question
        for i, q in enumerate(st.session_state.questions, 1):
            st.markdown(f"""
            <div class="question-container">
                <h4>Question {i}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Question metadata
            col_meta1, col_meta2, col_meta3 = st.columns([2, 1, 1])
            with col_meta1:
                st.markdown(f"**{q['text']}**")
            with col_meta2:
                st.badge(q['track'].upper(), type="secondary")
            with col_meta3:
                st.badge(q['difficulty'].upper(), type="secondary")
            
            # Answer options
            for j, option in enumerate(q['options']):
                option_letter = chr(65 + j)  # A, B, C, D
                if option == q['correct_answer']:
                    st.success(f"✅ **{option_letter}) {option}**")
                else:
                    st.write(f"{option_letter}) {option}")
            
            # Explanation in expandable section
            with st.expander("💡 View Explanation"):
                st.info(q['explanation'])
                st.caption(f"Generated by: {q['generated_by']}")
            
            if i < len(st.session_state.questions):
                st.divider()
        
        # Export options
        st.divider()
        st.subheader("📥 Export Options")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            # JSON export
            if st.button("📄 Export as JSON", use_container_width=True):
                questions_json = json.dumps(st.session_state.questions, indent=2, ensure_ascii=False)
                st.download_button(
                    label="⬇️ Download JSON File",
                    data=questions_json,
                    file_name=f"mcq_{track}_{difficulty}_{len(st.session_state.questions)}_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col_export2:
            # Text export
            if st.button("📝 Export as Text", use_container_width=True):
                text_content = f"MCQ Questions - {track.upper()} ({difficulty.title()})\n"
                text_content += f"Generated: {st.session_state.generation_info['timestamp']}\n"
                text_content += "=" * 50 + "\n\n"
                
                for i, q in enumerate(st.session_state.questions, 1):
                    text_content += f"Question {i}:\n{q['text']}\n\n"
                    for j, option in enumerate(q['options']):
                        letter = chr(65 + j)
                        marker = "✓" if option == q['correct_answer'] else " "
                        text_content += f"{letter}) {option} {marker}\n"
                    text_content += f"\nExplanation: {q['explanation']}\n"
                    text_content += "-" * 30 + "\n\n"
                
                st.download_button(
                    label="⬇️ Download Text File",
                    data=text_content,
                    file_name=f"mcq_{track}_{difficulty}_{len(st.session_state.questions)}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()