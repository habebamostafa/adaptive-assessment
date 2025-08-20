"""
Simple MCQ Generator - Streamlined Version
Focuses on reliability and basic functionality
"""

import streamlit as st
import json
import time
from typing import List, Dict
import re

# Try importing Hugging Face Hub
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class MCQGenerator:
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
        
        self.hf_token = None
        self.client = None
        self._setup_hugging_face()
    
    def _setup_hugging_face(self):
        """Setup Hugging Face connection"""
        if not HF_AVAILABLE:
            st.info("📋 Hugging Face not available - using demo mode")
            return
            
        # Try to get token from different sources
        try:
            # From Streamlit secrets
            if hasattr(st, 'secrets'):
                self.hf_token = st.secrets.get("HF_TOKEN") or st.secrets.get("hf_token")
            
            # From environment
            if not self.hf_token:
                import os
                self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            
            if self.hf_token:
                self.client = InferenceClient(token=self.hf_token)
                st.success("✅ AI Mode Ready!")
            else:
                st.info("📋 No HF token found - using demo mode")
                
        except Exception as e:
            st.warning(f"⚠️ HF setup issue: {str(e)[:100]}")
    
    def get_demo_questions(self) -> Dict[str, Dict[str, Dict]]:
        """Comprehensive demo questions database"""
        return {
            "web": {
                "easy": {
                    "question": "What does CSS stand for?",
                    "options": [
                        "Cascading Style Sheets",
                        "Computer Style Sheets",
                        "Creative Style Sheets", 
                        "Colorful Style Sheets"
                    ],
                    "correct_answer": "Cascading Style Sheets",
                    "explanation": "CSS stands for Cascading Style Sheets, used to describe the presentation of HTML documents."
                },
                "medium": {
                    "question": "What is the main advantage of using React hooks?",
                    "options": [
                        "Use state and lifecycle methods in functional components",
                        "Automatically optimize performance",
                        "Replace all class components",
                        "Handle routing automatically"
                    ],
                    "correct_answer": "Use state and lifecycle methods in functional components",
                    "explanation": "React hooks allow functional components to use state and other React features without writing a class."
                },
                "hard": {
                    "question": "What is the purpose of webpack's code splitting?",
                    "options": [
                        "Split code into smaller bundles loaded on demand",
                        "Separate CSS from JavaScript",
                        "Divide development from production code",
                        "Split frontend from backend code"
                    ],
                    "correct_answer": "Split code into smaller bundles loaded on demand",
                    "explanation": "Code splitting allows you to split your code into various bundles which can be loaded on demand or in parallel, improving performance."
                }
            },
            "ai": {
                "easy": {
                    "question": "What type of learning uses labeled training data?",
                    "options": [
                        "Supervised learning",
                        "Unsupervised learning",
                        "Reinforcement learning",
                        "Deep learning"
                    ],
                    "correct_answer": "Supervised learning",
                    "explanation": "Supervised learning uses labeled examples to train models to make predictions on new, unseen data."
                },
                "medium": {
                    "question": "What is the vanishing gradient problem in deep neural networks?",
                    "options": [
                        "Gradients become very small in early layers during backpropagation",
                        "The model fails to learn any patterns",
                        "Training data becomes corrupted",
                        "The loss function stops working"
                    ],
                    "correct_answer": "Gradients become very small in early layers during backpropagation",
                    "explanation": "The vanishing gradient problem occurs when gradients become exponentially smaller as they propagate back through the network, making it difficult to train early layers."
                },
                "hard": {
                    "question": "What is the key innovation of attention mechanisms in neural networks?",
                    "options": [
                        "Allow models to focus on relevant parts of input sequences",
                        "Reduce computational complexity to linear time",
                        "Eliminate the need for training data",
                        "Automatically generate new data samples"
                    ],
                    "correct_answer": "Allow models to focus on relevant parts of input sequences",
                    "explanation": "Attention mechanisms enable models to dynamically focus on different parts of the input sequence when producing each part of the output, improving performance on long sequences."
                }
            },
            "cyber": {
                "easy": {
                    "question": "What is phishing in cybersecurity?",
                    "options": [
                        "Fraudulent attempts to obtain sensitive information",
                        "A type of malware that encrypts files",
                        "A method of network monitoring",
                        "A firewall configuration technique"
                    ],
                    "correct_answer": "Fraudulent attempts to obtain sensitive information",
                    "explanation": "Phishing is a social engineering attack where attackers impersonate trusted entities to steal sensitive information like passwords or credit card numbers."
                },
                "medium": {
                    "question": "What is the difference between authentication and authorization?",
                    "options": [
                        "Authentication verifies identity, authorization controls access",
                        "Authentication controls access, authorization verifies identity",
                        "They are the same concept",
                        "Authentication is for users, authorization is for systems"
                    ],
                    "correct_answer": "Authentication verifies identity, authorization controls access",
                    "explanation": "Authentication confirms who you are (identity verification), while authorization determines what you can access (permission control)."
                },
                "hard": {
                    "question": "What is a side-channel attack in cryptography?",
                    "options": [
                        "Exploiting physical implementation characteristics rather than theoretical weaknesses",
                        "Using multiple encryption algorithms simultaneously",
                        "Attacking through network side connections",
                        "Breaking encryption by guessing keys"
                    ],
                    "correct_answer": "Exploiting physical implementation characteristics rather than theoretical weaknesses",
                    "explanation": "Side-channel attacks exploit information leaked through physical implementation of cryptographic systems, such as timing, power consumption, or electromagnetic emissions."
                }
            },
            "data": {
                "easy": {
                    "question": "What is the purpose of data normalization?",
                    "options": [
                        "Scale features to similar ranges for better model performance",
                        "Remove duplicate records from datasets",
                        "Convert categorical data to numerical",
                        "Compress data to save storage space"
                    ],
                    "correct_answer": "Scale features to similar ranges for better model performance",
                    "explanation": "Data normalization scales numerical features to similar ranges, preventing features with larger scales from dominating the learning process."
                },
                "medium": {
                    "question": "When should you use cross-validation in machine learning?",
                    "options": [
                        "To get reliable estimates of model performance",
                        "To increase the size of your training dataset",
                        "To speed up training time",
                        "To automatically select features"
                    ],
                    "correct_answer": "To get reliable estimates of model performance",
                    "explanation": "Cross-validation helps obtain more reliable estimates of model performance by testing on multiple different train-test splits of the data."
                },
                "hard": {
                    "question": "What is the bias-variance tradeoff in machine learning?",
                    "options": [
                        "Balance between model simplicity and flexibility to minimize total error",
                        "Choice between accuracy and interpretability",
                        "Tradeoff between training speed and prediction speed",
                        "Balance between training data size and model complexity"
                    ],
                    "correct_answer": "Balance between model simplicity and flexibility to minimize total error",
                    "explanation": "The bias-variance tradeoff describes the relationship between model complexity, underfitting (high bias), and overfitting (high variance) to minimize total prediction error."
                }
            }
        }
    
    def get_demo_question(self, track: str, difficulty: str) -> Dict:
        """Get a demo question"""
        demo_db = self.get_demo_questions()
        
        # Try to get specific question
        if track in demo_db and difficulty in demo_db[track]:
            demo = demo_db[track][difficulty]
            return {
                'text': demo['question'],
                'options': demo['options'],
                'correct_answer': demo['correct_answer'],
                'explanation': demo['explanation'],
                'track': track,
                'difficulty': difficulty,
                'generated_by': 'Demo Database'
            }
        
        # Fallback generic question
        return {
            'text': f"What is a key concept in {self.available_tracks.get(track, track)}?",
            'options': [
                "The fundamental principle",
                "An incorrect alternative",
                "Another wrong option",
                "A clearly false choice"
            ],
            'correct_answer': "The fundamental principle",
            'explanation': f"This tests basic knowledge in {track} at {difficulty} level.",
            'track': track,
            'difficulty': difficulty,
            'generated_by': 'Generic Fallback'
        }
    
    def generate_with_ai(self, track: str, difficulty: str) -> Dict:
        """Generate question with AI"""
        if not self.client:
            return self.get_demo_question(track, difficulty)
        
        # Create a focused prompt
        prompt = f"""Create a {difficulty} multiple choice question about {self.available_tracks[track]}.

Return JSON format:
{{
    "question": "your question here",
    "options": ["option A", "option B", "option C", "option D"],
    "correct": 0,
    "explanation": "why this answer is correct"
}}

Make it professional and test practical knowledge."""
        
        try:
            response = self.client.text_generation(
                model="google/flan-t5-base",
                prompt=prompt,
                max_new_tokens=300,
                temperature=0.7
            )
            
            # Parse response
            if isinstance(response, str):
                response_text = response
            elif isinstance(response, dict):
                response_text = response.get("generated_text", "")
            else:
                response_text = str(response)
            
            return self._parse_ai_response(response_text, track, difficulty)
            
        except Exception as e:
            st.warning(f"AI generation failed: {str(e)[:100]}")
            return self.get_demo_question(track, difficulty)
    
    def _parse_ai_response(self, response: str, track: str, difficulty: str) -> Dict:
        """Parse AI response"""
        try:
            # Try to find JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                options = data.get("options", [])
                correct_idx = data.get("correct", 0)
                
                if options and 0 <= correct_idx < len(options):
                    return {
                        'text': data.get("question", f"Question about {track}"),
                        'options': options,
                        'correct_answer': options[correct_idx],
                        'explanation': data.get("explanation", f"Testing {difficulty} {track} knowledge"),
                        'track': track,
                        'difficulty': difficulty,
                        'generated_by': 'FLAN-T5 AI'
                    }
            
            # If parsing fails, use demo
            return self.get_demo_question(track, difficulty)
            
        except:
            return self.get_demo_question(track, difficulty)
    
    def generate_questions(self, track: str, difficulty: str, count: int, use_ai: bool = True) -> List[Dict]:
        """Generate multiple questions"""
        questions = []
        progress_bar = st.progress(0)
        
        for i in range(count):
            progress_bar.progress((i + 1) / count, f"Generating question {i + 1}/{count}")
            
            if use_ai and self.client:
                question = self.generate_with_ai(track, difficulty)
            else:
                question = self.get_demo_question(track, difficulty)
            
            questions.append(question)
            
            # Prevent rate limiting
            if use_ai and i < count - 1:
                time.sleep(0.5)
        
        progress_bar.empty()
        return questions

def main():
    st.set_page_config(
        page_title="MCQ Generator",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 MCQ Generator")
    st.write("Generate technical interview questions")
    
    # Initialize
    if 'generator' not in st.session_state:
        st.session_state.generator = MCQGenerator()
    
    generator = st.session_state.generator
    
    # Configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Track selection
        track = st.selectbox(
            "Technology Track:",
            options=list(generator.available_tracks.keys()),
            format_func=lambda x: f"{x.upper()} - {generator.available_tracks[x].split('(')[0].strip()}"
        )
        
        # Settings
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            difficulty = st.selectbox("Difficulty:", ["easy", "medium", "hard"], index=1)
        with col1_2:
            count = st.number_input("Questions:", min_value=1, max_value=10, value=3)
    
    with col2:
        st.subheader("Settings")
        
        # Show mode
        if generator.client:
            st.success("🤖 AI Available")
            use_ai = st.checkbox("Use AI Generation", value=True)
        else:
            st.info("📋 Demo Mode")
            use_ai = False
        
        st.info(f"Track: {track.upper()}")
        st.info(f"Level: {difficulty}")
        st.info(f"Count: {count}")
    
    # Generate button
    if st.button("Generate Questions", type="primary", use_container_width=True):
        questions = generator.generate_questions(track, difficulty, count, use_ai)
        st.session_state.questions = questions
        st.success(f"Generated {len(questions)} questions!")
    
    # Display questions
    if 'questions' in st.session_state:
        st.divider()
        st.subheader("Generated Questions")
        
        for i, q in enumerate(st.session_state.questions, 1):
            st.write(f"### Question {i}")
            st.write(f"**{q['text']}**")
            
            # Options
            for j, option in enumerate(q['options']):
                if option == q['correct_answer']:
                    st.success(f"✅ {chr(65+j)}) {option}")
                else:
                    st.write(f"{chr(65+j)}) {option}")
            
            # Explanation
            with st.expander("Show Explanation"):
                st.info(q['explanation'])
                st.caption(f"Source: {q['generated_by']}")
            
            if i < len(st.session_state.questions):
                st.divider()
        
        # Export
        if st.button("Export as JSON"):
            json_data = json.dumps(st.session_state.questions, indent=2)
            st.download_button(
                "Download JSON",
                json_data,
                f"mcq_{track}_{difficulty}_{count}.json",
                "application/json"
            )

if __name__ == "__main__":
    main()