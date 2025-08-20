"""MCQ Generator - Robust API Version
Enhanced error handling and multiple API approaches
"""

import streamlit as st
import json
import time
import requests
from typing import List, Dict
import re

# Try importing Hugging Face Hub
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class RobustMCQGenerator:
    def __init__(self):
        """Initialize the MCQ generator with robust API handling"""
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
        self.api_status = "initializing"
        self._setup_api_access()
    
    def _setup_api_access(self):
        """Setup API access with multiple fallback methods"""
        # Try to get token
        self.hf_token = self._get_hf_token()
        
        if not self.hf_token:
            self.api_status = "no_token"
            return
        
        # Try different API methods
        self._test_api_methods()
    
    def _get_hf_token(self):
        """Get Hugging Face token from multiple sources"""
        token = None
        
        # From Streamlit secrets
        if hasattr(st, 'secrets'):
            try:
                token = st.secrets.get("HF_TOKEN") or st.secrets.get("hf_token")
            except Exception:
                pass
        
        # From environment
        if not token:
            import os
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        return token
    
    def _test_api_methods(self):
        """Test different API access methods"""
        if not HF_AVAILABLE:
            self.api_status = "hf_not_available"
            return
        
        # Method 1: Try InferenceClient with a more reliable model
        if self._test_inference_client():
            self.api_status = "inference_client_ready"
            return
        
        # Method 2: Try direct HTTP requests with a different model
        if self._test_direct_api():
            self.api_status = "direct_api_ready"
            return
        
        self.api_status = "api_failed"
    
    def _test_inference_client(self):
        """Test InferenceClient method with a more reliable model"""
        try:
            self.client = InferenceClient(token=self.hf_token)
            
            # Use a different model that's more likely to work
            test_response = self.client.text_generation(
                model="microsoft/DialoGPT-small",  # Use a different model
                prompt="Hello",
                max_new_tokens=5
            )
            return True
        except Exception as e:
            st.warning(f"InferenceClient test failed: {str(e)[:100]}...")
            return False
    
    def _test_direct_api(self):
        """Test direct API calls with a different model"""
        try:
            # Try a different model endpoint
            url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-small"
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": "Hello",
                "parameters": {
                    "max_new_tokens": 5,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                return True
            else:
                st.warning(f"Direct API test failed: Status {response.status_code}")
                return False
        except Exception as e:
            st.warning(f"Direct API test failed: {str(e)[:100]}...")
            return False
    
    def get_api_status_message(self):
        """Get user-friendly API status message"""
        status_messages = {
            "inference_client_ready": ("✅", "AI Ready - InferenceClient", "success"),
            "direct_api_ready": ("✅", "AI Ready - Direct API", "success"),
            "no_token": ("📋", "Demo Mode - No HF Token Found", "info"),
            "hf_not_available": ("⚠️", "Demo Mode - Install huggingface-hub", "warning"),
            "api_failed": ("⚠️", "Demo Mode - API Connection Failed", "warning"),
            "initializing": ("🔄", "Initializing API Connection...", "info")
        }
        
        return status_messages.get(self.api_status, ("❓", "Unknown Status", "info"))
    
    def get_comprehensive_demo_questions(self):
        """Comprehensive demo question database"""
        return {
            "web": {
                "easy": [
                    {
                        "question": "What does CSS stand for?",
                        "options": ["Cascading Style Sheets", "Computer Style Sheets", "Creative Style Sheets", "Colorful Style Sheets"],
                        "correct_answer": "Cascading Style Sheets",
                        "explanation": "CSS stands for Cascading Style Sheets, used to describe the presentation of HTML documents including layout, colors, and fonts."
                    },
                    {
                        "question": "Which HTML tag is used for the largest heading?",
                        "options": ["<h1>", "<h6>", "<header>", "<heading>"],
                        "correct_answer": "<h1>",
                        "explanation": "The <h1> tag represents the largest/most important heading in HTML, with headings decreasing in size from h1 to h6."
                    },
                    {
                        "question": "What does JavaScript primarily add to web pages?",
                        "options": ["Interactivity and dynamic behavior", "Styling and layout", "Structure and content", "Database connectivity"],
                        "correct_answer": "Interactivity and dynamic behavior",
                        "explanation": "JavaScript is a programming language that adds interactive elements and dynamic functionality to web pages."
                    }
                ],
                "medium": [
                    {
                        "question": "What is the main purpose of React's useState hook?",
                        "options": ["Manage component state in functional components", "Handle HTTP requests", "Style components", "Route between pages"],
                        "correct_answer": "Manage component state in functional components",
                        "explanation": "useState is a React hook that allows functional components to have and update state, eliminating the need for class components in many cases."
                    },
                    {
                        "question": "What is the difference between '==' and '===' in JavaScript?",
                        "options": ["=== checks type and value, == only checks value", "== is faster than ===", "=== is deprecated", "No difference"],
                        "correct_answer": "=== checks type and value, == only checks value",
                        "explanation": "=== performs strict equality checking both type and value, while == performs loose equality with type coercion."
                    },
                    {
                        "question": "What is the purpose of CSS Grid?",
                        "options": ["Create two-dimensional layouts", "Add animations", "Handle responsive images", "Manage fonts"],
                        "correct_answer": "Create two-dimensional layouts",
                        "explanation": "CSS Grid is a layout system that allows you to create complex two-dimensional layouts with rows and columns."
                    }
                ],
                "hard": [
                    {
                        "question": "What is the event loop in JavaScript?",
                        "options": ["Mechanism for handling asynchronous operations", "A type of HTML element", "A CSS animation property", "A React lifecycle method"],
                        "correct_answer": "Mechanism for handling asynchronous operations",
                        "explanation": "The event loop is JavaScript's concurrency model that handles asynchronous callbacks and ensures non-blocking execution."
                    },
                    {
                        "question": "What is webpack's code splitting feature?",
                        "options": ["Divide code into smaller bundles for better performance", "Separate CSS from JavaScript", "Split development and production builds", "Divide frontend from backend"],
                        "correct_answer": "Divide code into smaller bundles for better performance",
                        "explanation": "Code splitting allows webpack to break your code into smaller chunks that can be loaded on demand, improving initial load times."
                    }
                ]
            },
            "ai": {
                "easy": [
                    {
                        "question": "What is artificial intelligence?",
                        "options": ["Computer systems that can perform tasks requiring human intelligence", "Only robots that look like humans", "Software for data storage", "Internet search engines"],
                        "correct_answer": "Computer systems that can perform tasks requiring human intelligence",
                        "explanation": "AI refers to computer systems capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving."
                    },
                    {
                        "question": "What type of learning uses labeled training data?",
                        "options": ["Supervised learning", "Unsupervised learning", "Reinforcement learning", "Deep learning"],
                        "correct_answer": "Supervised learning",
                        "explanation": "Supervised learning uses labeled datasets where the correct output is known, allowing the model to learn the relationship between inputs and outputs."
                    }
                ],
                "medium": [
                    {
                        "question": "What is overfitting in machine learning?",
                        "options": ["Model performs well on training data but poorly on new data", "Model trains too slowly", "Model uses too much memory", "Model cannot learn any patterns"],
                        "correct_answer": "Model performs well on training data but poorly on new data",
                        "explanation": "Overfitting occurs when a model learns the training data too specifically, including noise, making it perform poorly on unseen data."
                    },
                    {
                        "question": "What is the purpose of activation functions in neural networks?",
                        "options": ["Introduce non-linearity to enable complex learning", "Store training data", "Connect to databases", "Display results to users"],
                        "correct_answer": "Introduce non-linearity to enable complex learning",
                        "explanation": "Activation functions add non-linearity to neural networks, allowing them to learn complex patterns and relationships in data."
                    }
                ],
                "hard": [
                    {
                        "question": "What is the vanishing gradient problem?",
                        "options": ["Gradients become extremely small during backpropagation", "Model outputs become zero", "Training data disappears", "Network connections break"],
                        "correct_answer": "Gradients become extremely small during backpropagation",
                        "explanation": "The vanishing gradient problem occurs when gradients become exponentially smaller during backpropagation, making it difficult to train deep networks."
                    }
                ]
            },
            "cyber": {
                "easy": [
                    {
                        "question": "What is malware?",
                        "options": ["Malicious software designed to harm computers", "A type of firewall", "Network monitoring tool", "Data backup system"],
                        "correct_answer": "Malicious software designed to harm computers",
                        "explanation": "Malware (malicious software) includes viruses, trojans, ransomware, and other programs designed to damage, disrupt, or gain unauthorized access to computer systems."
                    },
                    {
                        "question": "What is the primary purpose of encryption?",
                        "options": ["Protect data confidentiality", "Speed up data transfer", "Reduce file sizes", "Organize files"],
                        "correct_answer": "Protect data confidentiality",
                        "explanation": "Encryption converts readable data into an encoded format to protect confidentiality and prevent unauthorized access to sensitive information."
                    }
                ],
                "medium": [
                    {
                        "question": "What is a Man-in-the-Middle (MITM) attack?",
                        "options": ["Intercepting communication between two parties", "Physical theft of computers", "Overloading servers with requests", "Installing malware via email"],
                        "correct_answer": "Intercepting communication between two parties",
                        "explanation": "A MITM attack occurs when an attacker secretly intercepts and potentially alters communication between two parties who believe they are communicating directly."
                    },
                    {
                        "question": "What is two-factor authentication (2FA)?",
                        "options": ["Using two different types of verification", "Having two passwords", "Logging in twice", "Using two different browsers"],
                        "correct_answer": "Using two different types of verification",
                        "explanation": "2FA adds an extra layer of security by requiring two different authentication factors, such as something you know (password) and something you have (phone)."
                    }
                ],
                "hard": [
                    {
                        "question": "What is a zero-day exploit?",
                        "options": ["Attack using previously unknown vulnerabilities", "Attack that happens at midnight", "Attack that takes zero time", "Attack that costs no money"],
                        "correct_answer": "Attack using previously unknown vulnerabilities",
                        "explanation": "A zero-day exploit takes advantage of a security vulnerability that is unknown to security vendors and has no available patch or defense."
                    }
                ]
            },
            "data": {
                "easy": [
                    {
                        "question": "What is data science?",
                        "options": ["Extracting insights and knowledge from data", "Just creating charts and graphs", "Only working with big data", "Programming databases"],
                        "correct_answer": "Extracting insights and knowledge from data",
                        "explanation": "Data science combines statistics, programming, and domain expertise to extract meaningful insights and knowledge from structured and unstructured data."
                    }
                ],
                "medium": [
                    {
                        "question": "What is the purpose of data normalization?",
                        "options": ["Scale features to similar ranges", "Remove duplicate data", "Convert text to numbers", "Compress data files"],
                        "correct_answer": "Scale features to similar ranges",
                        "explanation": "Data normalization scales numerical features to similar ranges, preventing features with larger scales from dominating the analysis or model training."
                    }
                ],
                "hard": [
                    {
                        "question": "What is the bias-variance tradeoff?",
                        "options": ["Balance between model simplicity and complexity", "Choice between speed and accuracy", "Tradeoff between training and testing", "Balance between data size and model size"],
                        "correct_answer": "Balance between model simplicity and complexity",
                        "explanation": "The bias-variance tradeoff describes the relationship between a model's ability to minimize bias (underfitting) and variance (overfitting) to achieve optimal predictive performance."
                    }
                ]
            }
        }
    
    def get_demo_question(self, track: str, difficulty: str) -> Dict:
        """Get a demo question with rotation"""
        demo_db = self.get_comprehensive_demo_questions()
        
        # Initialize question indices
        if not hasattr(self, '_question_indices'):
            self._question_indices = {}
        
        key = f"{track}_{difficulty}"
        
        if track in demo_db and difficulty in demo_db[track]:
            questions = demo_db[track][difficulty]
            if questions:
                # Rotate through questions
                if key not in self._question_indices:
                    self._question_indices[key] = 0
                
                question_data = questions[self._question_indices[key] % len(questions)]
                self._question_indices[key] += 1
                
                return {
                    'text': question_data['question'],
                    'options': question_data['options'],
                    'correct_answer': question_data['correct_answer'],
                    'explanation': question_data['explanation'],
                    'track': track,
                    'difficulty': difficulty,
                    'generated_by': 'Demo Database'
                }
        
        # Fallback generic question
        return {
            'text': f"What is a fundamental concept in {self.available_tracks.get(track, track)}?",
            'options': [
                "The core principle of this technology",
                "An unrelated software tool",
                "A hardware component only",
                "A deprecated technique"
            ],
            'correct_answer': "The core principle of this technology",
            'explanation': f"This question tests fundamental understanding of {self.available_tracks.get(track, track)} concepts at {difficulty} level.",
            'track': track,
            'difficulty': difficulty,
            'generated_by': 'Generic Fallback'
        }
    
    def generate_with_ai(self, track: str, difficulty: str) -> Dict:
        """Generate question using AI with multiple API methods"""
        if self.api_status not in ["inference_client_ready", "direct_api_ready"]:
            return self.get_demo_question(track, difficulty)
        
        # Try inference client method first
        if self.api_status == "inference_client_ready":
            result = self._generate_with_inference_client(track, difficulty)
            if result:
                return result
        
        # Try direct API method
        if self.api_status == "direct_api_ready":
            result = self._generate_with_direct_api(track, difficulty)
            if result:
                return result
        
        # Fallback to demo
        st.warning("AI generation failed, using demo question")
        return self.get_demo_question(track, difficulty)
    
    def _generate_with_inference_client(self, track: str, difficulty: str) -> Dict:
        """Generate using InferenceClient"""
        try:
            prompt = self._create_simple_prompt(track, difficulty)
            
            # Use a different model that's more likely to work
            response = self.client.text_generation(
                model="google/flan-t5-large",  # Use a different model
                prompt=prompt,
                max_new_tokens=200,
                temperature=0.7
            )
            
            response_text = self._extract_response_text(response)
            if response_text:
                return self._parse_ai_response(response_text, track, difficulty)
                
        except Exception as e:
            st.warning(f"InferenceClient generation failed: {str(e)[:100]}...")
        
        return None
    
    def _generate_with_direct_api(self, track: str, difficulty: str) -> Dict:
        """Generate using direct API calls"""
        try:
            prompt = self._create_simple_prompt(track, difficulty)
            
            # Try a different model endpoint
            url = "https://huggingface.co/api/models/google/flan-t5-small?expan"
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                response_text = self._extract_response_text(result)
                if response_text:
                    return self._parse_ai_response(response_text, track, difficulty)
            else:
                st.warning(f"Direct API failed: Status {response.status_code}")
                
        except Exception as e:
            st.warning(f"Direct API generation failed: {str(e)[:100]}...")
        
        return None
    
    def _create_simple_prompt(self, track: str, difficulty: str) -> str:
        """Create a simple, effective prompt"""
        return f"""Create a {difficulty} level multiple choice question about {self.available_tracks[track]}.

Format:
Question: [your question]
A) [option 1]
B) [option 2]  
C) [option 3]
D) [option 4]
Answer: [A/B/C/D]
Explanation: [brief explanation]

Topic: {self.available_tracks[track]}
Level: {difficulty}
"""
    
    def _extract_response_text(self, response) -> str:
        """Extract text from various response formats"""
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            return response.get("generated_text", "")
        elif isinstance(response, list) and len(response) > 0:
            first_item = response[0]
            if isinstance(first_item, dict):
                return first_item.get("generated_text", "")
            elif isinstance(first_item, str):
                return first_item
        return ""
    
    def _parse_ai_response(self, response: str, track: str, difficulty: str) -> Dict:
        """Parse AI response with flexible parsing"""
        try:
            # Try to extract question and options
            lines = response.strip().split('\n')
            question = ""
            options = []
            correct_answer = ""
            explanation = ""
            
            current_section = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.lower().startswith('question:'):
                    question = line[9:].strip()
                    current_section = "question"
                elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                    option_text = line[2:].strip()
                    options.append(option_text)
                elif line.lower().startswith('answer:'):
                    answer_letter = line[7:].strip().upper()
                    if answer_letter in ['A', 'B', 'C', 'D'] and options:
                        idx = ord(answer_letter) - ord('A')
                        if 0 <= idx < len(options):
                            correct_answer = options[idx]
                elif line.lower().startswith('explanation:'):
                    explanation = line[12:].strip()
            
            # Validate parsed data
            if question and len(options) >= 4 and correct_answer:
                return {
                    'text': question,
                    'options': options[:4],  # Take first 4 options
                    'correct_answer': correct_answer,
                    'explanation': explanation or f"This tests {difficulty} level {track} knowledge.",
                    'track': track,
                    'difficulty': difficulty,
                    'generated_by': 'AI Generated'
                }
        
        except Exception:
            pass
        
        # If parsing fails, return None to trigger fallback
        return None
    
    def generate_questions(self, track: str, difficulty: str, count: int, use_ai: bool = True) -> List[Dict]:
        """Generate multiple questions"""
        questions = []
        progress_bar = st.progress(0, text="Starting generation...")
        
        for i in range(count):
            progress_bar.progress((i + 1) / count, text=f"Generating question {i + 1} of {count}...")
            
            if use_ai and self.api_status in ["inference_client_ready", "direct_api_ready"]:
                question = self.generate_with_ai(track, difficulty)
            else:
                question = self.get_demo_question(track, difficulty)
            
            questions.append(question)
            
            # Rate limiting for AI calls
            if use_ai and i < count - 1:
                time.sleep(2)  # Longer delay to avoid rate limits
        
        progress_bar.empty()
        return questions

def main():
    st.set_page_config(
        page_title="MCQ Generator",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Advanced MCQ Generator")
    st.markdown("Generate technical interview questions with robust AI integration")
    
    # Initialize generator
    if 'generator' not in st.session_state:
        with st.spinner("Initializing MCQ Generator..."):
            st.session_state.generator = RobustMCQGenerator()
    
    generator = st.session_state.generator
    
    # Show API status
    icon, message, status_type = generator.get_api_status_message()
    if status_type == "success":
        st.success(f"{icon} {message}")
    elif status_type == "warning":
        st.warning(f"{icon} {message}")
    else:
        st.info(f"{icon} {message}")
    
    # Configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📚 Configuration")
        
        track = st.selectbox(
            "Technology Track:",
            options=list(generator.available_tracks.keys()),
            format_func=lambda x: f"{x.upper()} - {generator.available_tracks[x].split('(')[0].strip()}"
        )
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            difficulty = st.selectbox("Difficulty:", ["easy", "medium", "hard"], index=1)
        with col1_2:
            count = st.number_input("Questions:", min_value=1, max_value=10, value=3)
    
    with col2:
        st.subheader("📊 Summary")
        
        # Show settings
        st.info(f"**Track:** {track.upper()}")
        st.info(f"**Level:** {difficulty.title()}")
        st.info(f"**Count:** {count}")
        
        # AI toggle
        can_use_ai = generator.api_status in ["inference_client_ready", "direct_api_ready"]
        if can_use_ai:
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
                st.success(f"✅ Generated {len(questions)} questions!")
                
            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")
    
    # Display questions
    if 'questions' in st.session_state and st.session_state.questions:
        st.divider()
        st.header("📝 Generated Questions")
        
        for i, q in enumerate(st.session_state.questions, 1):
            st.subheader(f"Question {i}")
            st.write(f"**{q['text']}**")
            
            # Answer options
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
            
            if i < len(st.session_state.questions):
                st.divider()
        
        # Export
        st.divider()
        if st.button("📥 Export as JSON"):
            json_data = json.dumps(st.session_state.questions, indent=2)
            st.download_button(
                "Download JSON",
                json_data,
                f"mcq_{track}_{difficulty}_{count}.json",
                "application/json"
            )

if __name__ == "__main__":
    main()