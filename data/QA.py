"""
Intelligent MCQ Generator using Google's FLAN-T5 with Hugging Face Token integration
"""

import streamlit as st
import requests
from typing import List, Dict, Optional
import json
import random
import time
from huggingface_hub import InferenceClient, login
import os

class FlanT5MCQGenerator:
    def __init__(self, hf_token: str = None):
        """Initialize the MCQ generator with Hugging Face token"""
        self.available_tracks = {
            "web": "Web Development (HTML, CSS, JavaScript, React, etc.)",
            "ai": "Artificial Intelligence (Machine Learning, Deep Learning, NLP, etc.)",
            "cyber": "Cybersecurity (Network Security, Encryption, Ethical Hacking, etc.)",
            "data": "Data Science (Data Analysis, Visualization, Statistics, etc.)",
            "mobile": "Mobile Development (Android, iOS, Flutter, React Native, etc.)",
            "devops": "DevOps (Docker, Kubernetes, CI/CD, Cloud Computing, etc.)"
        }
        
        # Set Hugging Face token
        self.hf_token = st.secrets["token"]
        if hf_token:
            os.environ['token'] = hf_token
            self.client = InferenceClient()
            try:
                login(token=hf_token)
                st.success("✅ تم تسجيل الدخول إلى Hugging Face بنجاح!")
            except Exception as e:
                st.error(f"❌ خطأ في تسجيل الدخول: {e}")
        
        # Initialize Hugging Face Inference Client
        self.client = None
        if hf_token:
            try:
                self.client = InferenceClient()
                st.success("✅ تم تهيئة عميل Hugging Face بنجاح!")
            except Exception as e:
                st.error(f"❌ خطأ في تهيئة العميل: {e}")
    
    def get_available_tracks(self) -> List[str]:
        """Get list of available technology tracks"""
        return list(self.available_tracks.keys())
    
    def generate_question_with_flan_t5(self, track: str, difficulty: str = "medium") -> Dict:
        """
        Generate a question using Google's FLAN-T5 model via Hugging Face
        
        Args:
            track: Technology track
            difficulty: Difficulty level (easy, medium, hard)
        
        Returns:
            Question dictionary
        """
        try:
            # Create a prompt for the FLAN-T5 model
            prompt = self._create_flan_t5_prompt(track, difficulty)
            
            # Generate question using FLAN-T5
            question_data = self._query_flan_t5(prompt)
            
            # Parse the response
            return self._parse_flan_t5_response(question_data, track, difficulty)
            
        except Exception as e:
            st.error(f"Error generating question with FLAN-T5: {e}")
            return self._create_fallback_question(track, difficulty)
    
    def _create_flan_t5_prompt(self, track: str, difficulty: str) -> str:
        """Create a prompt for FLAN-T5 model"""
        track_description = self.available_tracks[track]
        
        prompt = f"""
        Create a {difficulty} difficulty multiple choice question about {track_description}.
        The question should be technical and appropriate for a job interview.
        Provide the question text, four options (A, B, C, D), the correct answer, and a brief explanation.
        
        Format your response as JSON with the following structure:
        {{
            "question": "question text here",
            "options": {{
                "A": "option A text",
                "B": "option B text", 
                "C": "option C text",
                "D": "option D text"
            }},
            "correct_answer": "A",
            "explanation": "brief explanation here"
        }}
        
        Make sure the question is challenging and relevant to {track_description}.
        """
        
        return prompt
    
    def _query_flan_t5(self, prompt: str) -> Dict:
        """Query the FLAN-T5 model using Hugging Face Inference API"""
        if not self.client or not self.hf_token:
            st.warning("Using simulated response - add Hugging Face token for real API calls")
            return self._simulate_flan_t5_response(prompt)
        
        try:
            # Call the Hugging Face API
            response = self.client.text_generation(
                inputs=prompt,
                model="google/flan-t5-large",
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                return_full_text=False
            )
            
            # Try to extract JSON from the response
            try:
                # Find JSON part in the response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    return json.loads(json_str)
                else:
                    st.warning("Could not find JSON in response, using simulated response")
                    return self._simulate_flan_t5_response(prompt)
            except json.JSONDecodeError:
                st.warning("Failed to parse JSON response, using simulated response")
                return self._simulate_flan_t5_response(prompt)
                
        except Exception as e:
            st.error(f"Error calling Hugging Face API: {e}")
            return self._simulate_flan_t5_response(prompt)
    
    def _simulate_flan_t5_response(self, prompt: str) -> Dict:
        """Simulate FLAN-T5 response for demonstration purposes"""
        time.sleep(1)  # Simulate API call delay
        
        # Extract track and difficulty from prompt
        if "easy" in prompt:
            difficulty = "easy"
        elif "hard" in prompt:
            difficulty = "hard"
        else:
            difficulty = "medium"
            
        track = "web"
        if "Artificial Intelligence" in prompt:
            track = "ai"
        elif "Cybersecurity" in prompt:
            track = "cyber"
        elif "Data Science" in prompt:
            track = "data"
        elif "Mobile Development" in prompt:
            track = "mobile"
        elif "DevOps" in prompt:
            track = "devops"
        
        # Simulate different responses based on track and difficulty
        responses = {
            "web": {
                "easy": {
                    "question": "What does CSS stand for in web development?",
                    "options": {
                        "A": "Cascading Style Sheets",
                        "B": "Computer Style System",
                        "C": "Creative Style Solutions",
                        "D": "Coded Styling Syntax"
                    },
                    "correct_answer": "A",
                    "explanation": "CSS stands for Cascading Style Sheets, which is used to style and layout web pages."
                },
                "medium": {
                    "question": "What is the purpose of the Virtual DOM in React?",
                    "options": {
                        "A": "To improve performance by minimizing direct DOM manipulation",
                        "B": "To create 3D visualizations in the browser",
                        "C": "To provide virtual reality capabilities",
                        "D": "To encrypt DOM elements for security"
                    },
                    "correct_answer": "A",
                    "explanation": "The Virtual DOM is a programming concept where a virtual representation of the UI is kept in memory and synced with the real DOM, making React efficient."
                },
                "hard": {
                    "question": "How does Webpack's code splitting feature improve application performance?",
                    "options": {
                        "A": "By allowing lazy loading of code chunks when needed",
                        "B": "By compressing all JavaScript files into a single bundle",
                        "C": "By automatically minifying CSS and HTML files",
                        "D": "By encrypting the source code for security"
                    },
                    "correct_answer": "A",
                    "explanation": "Code splitting allows Webpack to divide code into various bundles that can be loaded on demand or in parallel, improving initial load time."
                }
            },
            "ai": {
                "easy": {
                    "question": "What is the primary goal of machine learning?",
                    "options": {
                        "A": "To enable computers to learn from data without explicit programming",
                        "B": "To create artificial human intelligence",
                        "C": "To replace human decision making entirely",
                        "D": "To build robots that can perform physical tasks"
                    },
                    "correct_answer": "A",
                    "explanation": "Machine learning focuses on developing algorithms that allow computers to learn from and make predictions based on data."
                },
                "medium": {
                    "question": "What is the difference between supervised and unsupervised learning?",
                    "options": {
                        "A": "Supervised learning uses labeled data, unsupervised learning finds patterns in unlabeled data",
                        "B": "Supervised learning is faster than unsupervised learning",
                        "C": "Unsupervised learning requires human supervision",
                        "D": "Supervised learning is for classification, unsupervised for regression"
                    },
                    "correct_answer": "A",
                    "explanation": "Supervised learning uses labeled datasets to train algorithms, while unsupervised learning finds hidden patterns in unlabeled data."
                },
                "hard": {
                    "question": "What is the transformer architecture's key innovation in natural language processing?",
                    "options": {
                        "A": "Self-attention mechanism that weights the importance of different words",
                        "B": "Using convolutional layers for text processing",
                        "C": "Implementing reinforcement learning for text generation",
                        "D": "Combining vision and language models in a single architecture"
                    },
                    "correct_answer": "A",
                    "explanation": "The transformer architecture introduced the self-attention mechanism, which allows the model to focus on different parts of the input sequence when processing each word."
                }
            },
            "cyber": {
                "easy": {
                    "question": "What is the purpose of a firewall in network security?",
                    "options": {
                        "A": "To monitor and control incoming and outgoing network traffic",
                        "B": "To encrypt all data passing through a network",
                        "C": "To prevent physical access to network devices",
                        "D": "To create backups of important data"
                    },
                    "correct_answer": "A",
                    "explanation": "A firewall is a network security device that monitors and filters incoming and outgoing network traffic based on an organization's security policies."
                },
                "medium": {
                    "question": "What is the difference between symmetric and asymmetric encryption?",
                    "options": {
                        "A": "Symmetric uses one key, asymmetric uses a public/private key pair",
                        "B": "Symmetric is faster but less secure than asymmetric",
                        "C": "Asymmetric is used for SSL/TLS, symmetric for hashing",
                        "D": "Symmetric is for encryption, asymmetric for digital signatures only"
                    },
                    "correct_answer": "A",
                    "explanation": "Symmetric encryption uses the same key for encryption and decryption, while asymmetric encryption uses a pair of public and private keys."
                },
                "hard": {
                    "question": "How does a zero-day vulnerability differ from other security vulnerabilities?",
                    "options": {
                        "A": "It is unknown to the software vendor and has no available patch",
                        "B": "It affects systems with zero days of uptime",
                        "C": "It can only be exploited at midnight (zero hour)",
                        "D": "It requires zero user interaction to exploit"
                    },
                    "correct_answer": "A",
                    "explanation": "A zero-day vulnerability is a software vulnerability that is unknown to the vendor or has no patch available, making it particularly dangerous."
                }
            }
        }
        
        # Return appropriate response based on track and difficulty
        if track in responses and difficulty in responses[track]:
            return responses[track][difficulty]
        else:
            # Fallback response
            return {
                "question": f"What is a key concept in {self.available_tracks[track]}?",
                "options": {
                    "A": "Correct answer",
                    "B": "Incorrect option 1",
                    "C": "Incorrect option 2",
                    "D": "Incorrect option 3"
                },
                "correct_answer": "A",
                "explanation": f"This is a {difficulty} level question about {self.available_tracks[track]}."
            }
    
    def _parse_flan_t5_response(self, response_data: Dict, track: str, difficulty: str) -> Dict:
        """Parse the response from FLAN-T5 into our question format"""
        if not response_data:
            return self._create_fallback_question(track, difficulty)
        
        # Convert options from object to list
        options_dict = response_data.get("options", {})
        options_list = [options_dict.get("A", ""), options_dict.get("B", ""), 
                       options_dict.get("C", ""), options_dict.get("D", "")]
        
        # Get correct answer text
        correct_key = response_data.get("correct_answer", "A")
        correct_answer_text = options_dict.get(correct_key, options_list[0])
        
        return {
            'text': response_data.get("question", f"What is a key concept in {self.available_tracks[track]}?"),
            'options': options_list,
            'correct_answer': correct_answer_text,
            'explanation': response_data.get("explanation", f"This is a {difficulty} level question about {self.available_tracks[track]}."),
            'track': track,
            'difficulty': difficulty,
            'generated': True
        }
    
    def _create_fallback_question(self, track: str, difficulty: str) -> Dict:
        """Create a fallback question if generation fails"""
        return {
            'text': f"What is a {difficulty} level concept in {self.available_tracks[track]}?",
            'options': [
                "Correct answer",
                "Incorrect option 1",
                "Incorrect option 2", 
                "Incorrect option 3"
            ],
            'correct_answer': "Correct answer",
            'explanation': f"This is a {difficulty} level question about {self.available_tracks[track]}.",
            'track': track,
            'difficulty': difficulty,
            'generated': True,
            'fallback': True
        }
    
    def generate_question_set(self, track: str, num_questions: int = 5, difficulty: str = "medium") -> List[Dict]:
        """
        Generate a set of questions for a track
        
        Args:
            track: Technology track
            num_questions: Number of questions to generate
            difficulty: Difficulty level (easy, medium, hard)
        
        Returns:
            List of question dictionaries
        """
        questions = []
        for i in range(num_questions):
            question = self.generate_question_with_flan_t5(track, difficulty)
            questions.append(question)
            # Small delay to simulate API call
            time.sleep(0.5)
        
        return questions

# Streamlit interface
def main():
    """Main function to run the Streamlit app"""
    st.title("🤖 FLAN-T5 MCQ Generator with Hugging Face Token")
    st.write("Generate technical interview questions using Google's FLAN-T5 model with your Hugging Face token")
    
    # Hugging Face Token input
    hf_token = st.text_input(
        "Enter your Hugging Face Token:",
        type="password",
        help="Get your token from https://huggingface.co/settings/tokens"
    )
    
    # Initialize generator
    if 'generator' not in st.session_state or st.session_state.get('current_token') != hf_token:
        if hf_token:
            st.session_state.generator = FlanT5MCQGenerator(hf_token)
            st.session_state.current_token = hf_token
        else:
            st.session_state.generator = FlanT5MCQGenerator()
    
    generator = st.session_state.generator
    
    # Track selection
    track = st.selectbox(
        "Select technology track:",
        options=generator.get_available_tracks(),
        format_func=lambda x: f"{x} - {generator.available_tracks[x]}"
    )
    
    # Difficulty selection
    difficulty = st.radio(
        "Select difficulty level:",
        options=["easy", "medium", "hard"],
        horizontal=True
    )
    
    # Number of questions
    num_questions = st.slider("Number of questions to generate:", 1, 10, 3)
    
    # Generate button
    if st.button("Generate Questions with FLAN-T5"):
        if not hf_token:
            st.warning("⚠️ Please enter your Hugging Face token to generate questions")
        else:
            with st.spinner(f"Generating {num_questions} {difficulty} questions for {track} using FLAN-T5..."):
                questions = generator.generate_question_set(track, num_questions, difficulty)
                
                # Display questions
                for i, q in enumerate(questions, 1):
                    st.subheader(f"Question {i}")
                    st.write(q['text'])
                    
                    # Display options
                    for j, option in enumerate(q['options']):
                        st.write(f"{chr(65+j)}) {option}")
                    
                    # Add expander for answer and explanation
                    with st.expander("Show Answer and Explanation"):
                        st.success(f"Correct answer: {q['correct_answer']}")
                        st.info(f"Explanation: {q['explanation']}")
                    
                    st.divider()

if __name__ == "__main__":
    main()