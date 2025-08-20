"""
MCQ Generator - Clean Working Version
Fixed all compatibility issues and simplified for reliability
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
        self.setup_status = "initializing"
        self._setup_hugging_face()
    
    def _setup_hugging_face(self):
        """Setup Hugging Face connection"""
        if not HF_AVAILABLE:
            self.setup_status = "hf_not_available"
            return
            
        try:
            # Try to get token from different sources
            self.hf_token = None
            
            # From Streamlit secrets
            if hasattr(st, 'secrets'):
                try:
                    self.hf_token = st.secrets.get("HF_TOKEN") or st.secrets.get("hf_token")
                except Exception:
                    pass
            
            # From environment
            if not self.hf_token:
                import os
                self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            
            if self.hf_token:
                try:
                    self.client = InferenceClient(token=self.hf_token)
                    self.setup_status = "ai_ready"
                except Exception as e:
                    self.setup_status = f"client_error: {str(e)[:50]}"
            else:
                self.setup_status = "no_token"
                
        except Exception as e:
            self.setup_status = f"setup_error: {str(e)[:50]}"
    
    def get_setup_status(self):
        """Get human readable setup status"""
        status_messages = {
            "ai_ready": ("✅", "AI Mode Ready", "success"),
            "no_token": ("📋", "Demo Mode - No HF Token", "info"),
            "hf_not_available": ("⚠️", "Demo Mode - Install huggingface-hub", "warning"),
            "initializing": ("🔄", "Initializing...", "info")
        }
        
        if self.setup_status.startswith(("client_error:", "setup_error:")):
            return ("⚠️", f"Demo Mode - {self.setup_status}", "warning")
        
        return status_messages.get(self.setup_status, ("❓", "Unknown Status", "info"))
    
    def get_comprehensive_demo_questions(self):
        """Comprehensive demo questions for all tracks"""
        return {
            "web": {
                "easy": [
                    {
                        "question": "What does HTML stand for?",
                        "options": ["HyperText Markup Language", "High Tech Modern Language", "Home Tool Markup Language", "Hyperlink Text Management Language"],
                        "correct_answer": "HyperText Markup Language",
                        "explanation": "HTML stands for HyperText Markup Language, the standard markup language for creating web pages."
                    },
                    {
                        "question": "Which CSS property is used to change text color?",
                        "options": ["color", "text-color", "font-color", "text-style"],
                        "correct_answer": "color",
                        "explanation": "The 'color' property in CSS is used to set the color of text content."
                    }
                ],
                "medium": [
                    {
                        "question": "What is the main advantage of using React hooks?",
                        "options": ["Use state in functional components", "Automatic performance optimization", "Built-in routing", "Direct DOM manipulation"],
                        "correct_answer": "Use state in functional components",
                        "explanation": "React hooks allow functional components to use state and other React features without writing class components."
                    },
                    {
                        "question": "What is the purpose of the Virtual DOM in React?",
                        "options": ["Optimize rendering performance", "Store application data", "Handle user authentication", "Manage server connections"],
                        "correct_answer": "Optimize rendering performance",
                        "explanation": "Virtual DOM creates an in-memory representation of the real DOM to optimize updates and improve performance."
                    }
                ],
                "hard": [
                    {
                        "question": "What is webpack's tree shaking feature?",
                        "options": ["Remove unused code from bundles", "Organize file structure", "Optimize images", "Minify CSS files"],
                        "correct_answer": "Remove unused code from bundles",
                        "explanation": "Tree shaking eliminates dead code from JavaScript bundles using static analysis of ES6 imports/exports."
                    }
                ]
            },
            "ai": {
                "easy": [
                    {
                        "question": "What type of learning uses labeled training data?",
                        "options": ["Supervised learning", "Unsupervised learning", "Reinforcement learning", "Transfer learning"],
                        "correct_answer": "Supervised learning",
                        "explanation": "Supervised learning uses labeled examples to train models to make predictions on new data."
                    }
                ],
                "medium": [
                    {
                        "question": "What is overfitting in machine learning?",
                        "options": ["Model performs well on training but poor on test data", "Model trains too slowly", "Model uses too much memory", "Model cannot learn patterns"],
                        "correct_answer": "Model performs well on training but poor on test data",
                        "explanation": "Overfitting occurs when a model learns the training data too specifically and fails to generalize to new data."
                    }
                ],
                "hard": [
                    {
                        "question": "What is the key innovation of Transformer architecture?",
                        "options": ["Self-attention mechanism", "Convolutional layers", "Recurrent connections", "Pooling layers"],
                        "correct_answer": "Self-attention mechanism",
                        "explanation": "Transformers introduced self-attention, allowing models to process sequences in parallel and capture long-range dependencies."
                    }
                ]
            },
            "cyber": {
                "easy": [
                    {
                        "question": "What is the primary function of a firewall?",
                        "options": ["Control network traffic", "Encrypt files", "Scan for viruses", "Create backups"],
                        "correct_answer": "Control network traffic",
                        "explanation": "A firewall monitors and controls incoming and outgoing network traffic based on security rules."
                    }
                ],
                "medium": [
                    {
                        "question": "What's the difference between symmetric and asymmetric encryption?",
                        "options": ["Symmetric uses one key, asymmetric uses key pairs", "Symmetric is faster, asymmetric is slower", "Symmetric is newer, asymmetric is older", "No difference"],
                        "correct_answer": "Symmetric uses one key, asymmetric uses key pairs",
                        "explanation": "Symmetric encryption uses the same key for encryption/decryption, while asymmetric uses public/private key pairs."
                    }
                ],
                "hard": [
                    {
                        "question": "What makes zero-day vulnerabilities dangerous?",
                        "options": ["No patches available", "They spread automatically", "They're always critical", "They affect all systems"],
                        "correct_answer": "No patches available",
                        "explanation": "Zero-day vulnerabilities are unknown to vendors, so no patches or defenses exist when they're discovered by attackers."
                    }
                ]
            },
            "data": {
                "easy": [
                    {
                        "question": "What is the purpose of data visualization?",
                        "options": ["Make data easier to understand", "Reduce file size", "Encrypt information", "Speed up processing"],
                        "correct_answer": "Make data easier to understand",
                        "explanation": "Data visualization transforms data into visual formats to make patterns and insights more accessible."
                    }
                ],
                "medium": [
                    {
                        "question": "When should you use median instead of mean?",
                        "options": ["When data has outliers", "When data is small", "When data is large", "Never"],
                        "correct_answer": "When data has outliers",
                        "explanation": "Median is more robust to outliers and provides better central tendency for skewed distributions."
                    }
                ],
                "hard": [
                    {
                        "question": "What is the curse of dimensionality?",
                        "options": ["Performance degrades with more features", "Visualization becomes impossible", "Storage requirements increase", "Processing time increases linearly"],
                        "correct_answer": "Performance degrades with more features",
                        "explanation": "In high-dimensional spaces, data becomes sparse and distance metrics less meaningful, degrading ML performance."
                    }
                ]
            }
        }
    
    def get_demo_question(self, track: str, difficulty: str) -> Dict:
        """Get a demo question for the track and difficulty"""
        demo_db = self.get_comprehensive_demo_questions()
        
        # Get questions for this track/difficulty
        if track in demo_db and difficulty in demo_db[track]:
            questions = demo_db[track][difficulty]
            if questions:
                # Rotate through questions
                if not hasattr(self, '_question_index'):
                    self._question_index = {}
                
                key = f"{track}_{difficulty}"
                if key not in self._question_index:
                    self._question_index[key] = 0
                
                question_data = questions[self._question_index[key] % len(questions)]
                self._question_index[key] += 1
                
                return {
                    'text': question_data['question'],
                    'options': question_data['options'],
                    'correct_answer': question_data['correct_answer'],
                    'explanation': question_data['explanation'],
                    'track': track,
                    'difficulty': difficulty,
                    'generated_by': 'Demo Database'
                }
        
        # Fallback
        return {
            'text': f"What is an important concept in {self.available_tracks.get(track, track)}?",
            'options': ["The correct fundamental principle", "An incorrect alternative", "Another wrong option", "A clearly false choice"],
            'correct_answer': "The correct fundamental principle",
            'explanation': f"This tests {difficulty} level knowledge of {self.available_tracks.get(track, track)}.",
            'track': track,
            'difficulty': difficulty,
            'generated_by': 'Generic Fallback'
        }
    
    def generate_with_ai(self, track: str, difficulty: str) -> Dict:
        """Generate question using AI"""
        if not self.client:
            return self.get_demo_question(track, difficulty)
        
        prompt = f"""Generate a {difficulty} level multiple choice question about {self.available_tracks[track]}.

Return only JSON:
{{
    "question": "your question text",
    "options": ["option A", "option B", "option C", "option D"], 
    "correct": 0,
    "explanation": "brief explanation"
}}"""
        
        try:
            response = self.client.text_generation(
                model="google/flan-t5-base",
                prompt=prompt,
                max_new_tokens=300,
                temperature=0.7
            )
            
            # Handle different response formats
            response_text = ""
            if isinstance(response, str):
                response_text = response
            elif isinstance(response, dict):
                response_text = response.get("generated_text", str(response))
            else:
                response_text = str(response)
            
            return self._parse_ai_response(response_text, track, difficulty)
            
        except Exception as e:
            st.warning(f"AI generation failed: {str(e)[:100]}...")
            return self.get_demo_question(track, difficulty)
    
    def _parse_ai_response(self, response: str, track: str, difficulty: str) -> Dict:
        """Parse AI response"""
        try:
            # Find JSON in response
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
                        'generated_by': 'AI Generated'
                    }
            
        except Exception:
            pass
        
        # Fallback to demo
        return self.get_demo_question(track, difficulty)
    
    def generate_questions(self, track: str, difficulty: str, count: int, use_ai: bool = True) -> List[Dict]:
        """Generate multiple questions"""
        questions = []
        progress_bar = st.progress(0, text="Starting generation...")
        
        for i in range(count):
            progress_bar.progress((i + 1) / count, text=f"Generating question {i + 1} of {count}...")
            
            if use_ai and self.client:
                question = self.generate_with_ai(track, difficulty)
            else:
                question = self.get_demo_question(track, difficulty)
            
            questions.append(question)
            
            # Small delay to prevent rate limiting
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
    st.markdown("Generate technical interview questions with AI or demo mode")
    
    # Initialize generator
    if 'generator' not in st.session_state:
        with st.spinner("Initializing MCQ Generator..."):
            st.session_state.generator = MCQGenerator()
    
    generator = st.session_state.generator
    
    # Show setup status
    icon, message, status_type = generator.get_setup_status()
    if status_type == "success":
        st.success(f"{icon} {message}")
    elif status_type == "warning":
        st.warning(f"{icon} {message}")
    else:
        st.info(f"{icon} {message}")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📚 Configuration")
        
        # Track selection
        track = st.selectbox(
            "Technology Track:",
            options=list(generator.available_tracks.keys()),
            format_func=lambda x: f"{x.upper()} - {generator.available_tracks[x].split('(')[0].strip()}"
        )
        
        # Settings in columns
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            difficulty = st.selectbox("Difficulty:", ["easy", "medium", "hard"], index=1)
        with col1_2:
            count = st.number_input("Questions:", min_value=1, max_value=10, value=3)
    
    with col2:
        st.subheader("📊 Summary")
        
        # Show current settings
        st.info(f"**Track:** {track.upper()}")
        st.info(f"**Level:** {difficulty.title()}")
        st.info(f"**Count:** {count}")
        
        # AI toggle
        if generator.client:
            use_ai = st.checkbox("🤖 Use AI Generation", value=True)
            if use_ai:
                st.success("**Mode:** AI")
            else:
                st.warning("**Mode:** Demo")
        else:
            use_ai = False
            st.warning("**Mode:** Demo Only")
    
    st.divider()
    
    # Generate button
    if st.button("🚀 Generate Questions", type="primary", use_container_width=True):
        with st.spinner("Generating questions..."):
            try:
                questions = generator.generate_questions(track, difficulty, count, use_ai)
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
        
        # Header
        col_h1, col_h2 = st.columns([2, 1])
        with col_h1:
            st.header("📝 Generated Questions")
        with col_h2:
            if 'generation_info' in st.session_state:
                info = st.session_state.generation_info
                st.caption(f"Generated: {info['timestamp']}")
                st.caption(f"Mode: {info['mode']}")
        
        # Questions
        for i, q in enumerate(st.session_state.questions, 1):
            with st.container():
                st.subheader(f"Question {i}")
                
                # Question text with metadata
                col_q1, col_q2, col_q3 = st.columns([3, 1, 1])
                with col_q1:
                    st.markdown(f"**{q['text']}**")
                with col_q2:
                    st.write(f"*{q['track'].upper()}*")
                with col_q3:
                    st.write(f"*{q['difficulty'].title()}*")
                
                # Answer options
                for j, option in enumerate(q['options']):
                    option_letter = chr(65 + j)  # A, B, C, D
                    if option == q['correct_answer']:
                        st.success(f"✅ **{option_letter}) {option}**")
                    else:
                        st.write(f"{option_letter}) {option}")
                
                # Explanation
                with st.expander("💡 Show Explanation"):
                    st.info(q['explanation'])
                    st.caption(f"Generated by: {q['generated_by']}")
                
                if i < len(st.session_state.questions):
                    st.divider()
        
        # Export section
        st.divider()
        st.subheader("📥 Export Options")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            if st.button("📄 Export as JSON", use_container_width=True):
                json_data = json.dumps(st.session_state.questions, indent=2, ensure_ascii=False)
                st.download_button(
                    "⬇️ Download JSON File",
                    json_data,
                    f"mcq_{track}_{difficulty}_{count}_{int(time.time())}.json",
                    "application/json",
                    use_container_width=True
                )
        
        with col_exp2:
            if st.button("📝 Export as Text", use_container_width=True):
                # Create text format
                text_content = f"MCQ Questions - {track.upper()} ({difficulty.title()})\n"
                text_content += f"Generated: {st.session_state.generation_info['timestamp']}\n"
                text_content += "=" * 50 + "\n\n"
                
                for i, q in enumerate(st.session_state.questions, 1):
                    text_content += f"Question {i}:\n{q['text']}\n\n"
                    for j, option in enumerate(q['options']):
                        letter = chr(65 + j)
                        marker = " ✓" if option == q['correct_answer'] else ""
                        text_content += f"{letter}) {option}{marker}\n"
                    text_content += f"\nExplanation: {q['explanation']}\n"
                    text_content += "-" * 30 + "\n\n"
                
                st.download_button(
                    "⬇️ Download Text File", 
                    text_content,
                    f"mcq_{track}_{difficulty}_{count}.txt",
                    "text/plain",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()