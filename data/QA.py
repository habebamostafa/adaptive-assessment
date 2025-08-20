"""
Enhanced MCQ Generator using Google's FLAN-T5 with Hugging Face Token
Fixed and Simplified Version
"""

import streamlit as st
import requests
import json
import random
import time
from typing import List, Dict, Optional
import os

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
        """Setup Hugging Face connection"""
        if not HF_AVAILABLE:
            st.warning("⚠️ Hugging Face Hub not available. Install with: pip install huggingface-hub")
            return
            
        try:
            # Get token from secrets
            if hasattr(st, 'secrets') and hasattr(st.secrets, 'get'):
                self.hf_token = st.secrets.get("hf_token", None)
            
            if self.hf_token:
                self.client = InferenceClient(token=self.hf_token)
                st.success("✅ Hugging Face connected successfully!")
            else:
                st.info("ℹ️ No Hugging Face token found. Using demo mode.")
        except Exception as e:
            st.warning(f"⚠️ Hugging Face setup failed: {e}")
    
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
                    "explanation": "HTML stands for HyperText Markup Language, the standard markup language for web pages."
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
                    "explanation": "Virtual DOM creates a virtual representation of the UI and efficiently updates only changed parts."
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
                    "explanation": "Tree shaking removes dead code by analyzing ES6 module imports/exports during build."
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
                    "explanation": "Machine learning allows computers to improve performance on tasks through experience."
                },
                "medium": {
                    "question": "What distinguishes supervised from unsupervised learning?",
                    "options": [
                        "Supervised uses labeled data, unsupervised finds patterns in unlabeled data",
                        "Supervised is always faster",
                        "Unsupervised requires human oversight",
                        "Supervised only works with images"
                    ],
                    "correct_answer": "Supervised uses labeled data, unsupervised finds patterns in unlabeled data",
                    "explanation": "Supervised learning learns from labeled examples, while unsupervised finds hidden patterns."
                },
                "hard": {
                    "question": "What is the key innovation of Transformer architecture?",
                    "options": [
                        "Self-attention mechanism for parallel sequence processing",
                        "Convolutional layers for text processing",
                        "Recurrent connections for memory",
                        "Reinforcement learning integration"
                    ],
                    "correct_answer": "Self-attention mechanism for parallel sequence processing",
                    "explanation": "Transformers use self-attention to process all sequence positions simultaneously."
                }
            },
            "cyber": {
                "easy": {
                    "question": "What is a firewall's primary function?",
                    "options": [
                        "Control network traffic based on security rules",
                        "Encrypt hard drive data",
                        "Prevent physical device access",
                        "Create data backups automatically"
                    ],
                    "correct_answer": "Control network traffic based on security rules",
                    "explanation": "Firewalls monitor and filter network traffic according to security policies."
                },
                "medium": {
                    "question": "What's the difference between symmetric and asymmetric encryption?",
                    "options": [
                        "Symmetric uses one key, asymmetric uses public/private key pairs",
                        "Symmetric is always more secure",
                        "Asymmetric is only for SSL/TLS",
                        "Symmetric only encrypts text"
                    ],
                    "correct_answer": "Symmetric uses one key, asymmetric uses public/private key pairs",
                    "explanation": "Symmetric encryption uses the same key for encryption/decryption, asymmetric uses paired keys."
                },
                "hard": {
                    "question": "What makes zero-day vulnerabilities particularly dangerous?",
                    "options": [
                        "They're unknown to vendors with no available patches",
                        "They only work on new systems",
                        "They can only be exploited at midnight",
                        "They require no user interaction"
                    ],
                    "correct_answer": "They're unknown to vendors with no available patches",
                    "explanation": "Zero-day vulnerabilities are unknown security flaws with no existing defenses."
                }
            }
        }
        
        # Get the question or return a generic one
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
        """Generate question using AI (FLAN-T5)"""
        if not self.client or not self.hf_token:
            return self.get_demo_question(track, difficulty)
        
        try:
            prompt = self._create_prompt(track, difficulty)
            
            response = self.client.text_generation(
                model="google/flan-t5-large",
                prompt=prompt,          # النص هنا
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                return_full_text=False
            )
            if isinstance(response, list) and "generated_text" in response[0]:
                generated_text = response[0]["generated_text"]
                return self._parse_response(generated_text, track, difficulty)
            else:
                st.warning("⚠️ لم يتم توليد نص صحيح من AI، استخدام الأسئلة التجريبية")
                return self.get_demo_question(track, difficulty)
                    
            return self._parse_response(generated_text,response, track, difficulty)
            
        except Exception as e:
            st.warning(f"AI generation failed: {e}")
            return self.get_demo_question(track, difficulty)
    
    def _create_prompt(self, track: str, difficulty: str) -> str:
        """Create prompt for FLAN-T5"""
        track_desc = self.available_tracks[track]
        
        prompt = f"""Create a {difficulty} level multiple choice question about {track_desc}.

Requirements:
- Professional interview quality
- Test practical knowledge  
- Exactly 4 options (A, B, C, D)
- One correct answer
- Brief explanation

Format as JSON:
{{
    "question": "question text here",
    "options": {{
        "A": "first option",
        "B": "second option", 
        "C": "third option",
        "D": "fourth option"
    }},
    "correct_answer": "A",
    "explanation": "explanation text"
}}

Topic: {track_desc}
Level: {difficulty}
"""
        return prompt
    
    def _parse_response(self, response: str, track: str, difficulty: str) -> Dict:
        """Parse AI response"""
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                options_dict = data.get("options", {})
                options_list = [
                    options_dict.get("A", "Option A"),
                    options_dict.get("B", "Option B"),
                    options_dict.get("C", "Option C"),
                    options_dict.get("D", "Option D")
                ]
                
                correct_key = data.get("correct_answer", "A")
                correct_answer_text = options_dict.get(correct_key, options_list[0])
                
                return {
                    'text': data.get("question", f"What is important in {track}?"),
                    'options': options_list,
                    'correct_answer': correct_answer_text,
                    'explanation': data.get("explanation", f"This tests {difficulty} {track} knowledge."),
                    'track': track,
                    'difficulty': difficulty,
                    'generated_by': 'FLAN-T5 AI'
                }
            else:
                raise ValueError("No JSON found")
                
        except Exception as e:
            return self.get_demo_question(track, difficulty)
    
    def generate_question_set(self, track: str, num_questions: int, difficulty: str, use_ai: bool = True) -> List[Dict]:
        """Generate a set of questions"""
        questions = []
        
        for i in range(num_questions):
            if use_ai and self.client and self.hf_token:
                question = self.generate_with_ai(track, difficulty)
            else:
                question = self.get_demo_question(track, difficulty)
            
            questions.append(question)
            
            # Show progress
            progress = (i + 1) / num_questions
            if 'progress_bar' in st.session_state:
                st.session_state.progress_bar.progress(progress)
            
            time.sleep(0.5)  # Small delay
        
        return questions

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="MCQ Generator",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 MCQ Generator with FLAN-T5")
    st.write("Generate technical interview questions using AI")
    
    # Initialize generator
    if 'generator' not in st.session_state:
        st.session_state.generator = SimpleMCQGenerator()
    
    generator = st.session_state.generator
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Connection status
        if generator.hf_token:
            st.success("✅ AI Mode Active")
        else:
            st.info("📋 Demo Mode Active")
            st.caption("Add hf_token to secrets for AI generation")
        
        # Options
        use_ai = st.checkbox("Use AI Generation", value=bool(generator.hf_token))
        if not generator.hf_token:
            use_ai = False
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Track selection
        track = st.selectbox(
            "Technology Track:",
            options=generator.get_available_tracks(),
            format_func=lambda x: f"{x.upper()} - {generator.available_tracks[x].split('(')[0].strip()}"
        )
        
        # Difficulty and number
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            difficulty = st.selectbox("Difficulty:", ["easy", "medium", "hard"], index=1)
        with col1_2:
            num_questions = st.number_input("Questions:", min_value=1, max_value=10, value=3)
    
    with col2:
        st.subheader("📊 Summary")
        st.info(f"**Track:** {track.upper()}")
        st.info(f"**Level:** {difficulty.title()}")
        st.info(f"**Count:** {num_questions}")
        if use_ai:
            st.success("**Mode:** AI")
        else:
            st.warning("**Mode:** Demo")
    
    # Generate button
    if st.button("🚀 Generate Questions", type="primary", use_container_width=True):
        with st.spinner("Generating questions..."):
            # Create progress bar
            st.session_state.progress_bar = st.progress(0)
            
            questions = generator.generate_question_set(
                track=track,
                num_questions=num_questions, 
                difficulty=difficulty,
                use_ai=use_ai
            )
            
            # Clear progress bar
            st.session_state.progress_bar.empty()
            del st.session_state.progress_bar
            
            # Store questions
            st.session_state.questions = questions
            
            st.success(f"✅ Generated {len(questions)} questions!")
    
    # Display questions
    if 'questions' in st.session_state:
        st.divider()
        st.header("📝 Generated Questions")
        
        for i, q in enumerate(st.session_state.questions, 1):
            with st.container():
                # Question header
                col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
                with col_h1:
                    st.subheader(f"Question {i}")
                with col_h2:
                    st.badge(q['track'].upper())
                with col_h3:
                    st.badge(q['difficulty'].upper())
                
                # Question text
                st.write(f"**{q['text']}**")
                
                # Options
                for j, option in enumerate(q['options']):
                    option_letter = chr(65 + j)
                    if option == q['correct_answer']:
                        st.success(f"✅ **{option_letter}) {option}**")
                    else:
                        st.write(f"{option_letter}) {option}")
                
                # Explanation
                with st.expander("💡 Show Explanation"):
                    st.info(q['explanation'])
                    st.caption(f"Generated by: {q['generated_by']}")
                
                st.divider()
        
        # Export button
        if st.button("📥 Export as JSON"):
            questions_json = json.dumps(st.session_state.questions, indent=2, ensure_ascii=False)
            st.download_button(
                label="⬇️ Download JSON File",
                data=questions_json,
                file_name=f"mcq_{track}_{difficulty}_{len(st.session_state.questions)}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()