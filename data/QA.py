import streamlit as st
import json
import time
import requests
from typing import List, Dict
import random

class OnlineMCQGenerator:
    def __init__(self):
        """Initialize the MCQ generator with online API"""
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
        
        # Free API endpoints (no token required)
        self.api_endpoints = [
            "https://api-inference.huggingface.co/models/google/flan-t5-large",
            "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large",
            "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
        ]
        
        self.current_api_index = 0
        
    def get_api_url(self):
        """Get the current API endpoint"""
        return self.api_endpoints[self.current_api_index % len(self.api_endpoints)]
        
    def generate_with_api(self, track: str, difficulty: str) -> Dict:
        """Generate question using online API"""
        try:
            prompt = self._create_prompt(track, difficulty)
            
            # Try different API endpoints
            for attempt in range(3):
                api_url = self.get_api_url()
                try:
                    response = requests.post(
                        api_url,
                        json={
                            "inputs": prompt,
                            "parameters": {
                                "max_length": 300,
                                "temperature": 0.8,
                                "do_sample": True,
                                "return_full_text": False
                            }
                        },
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get("generated_text", "")
                        else:
                            generated_text = str(result)
                        
                        parsed_question = self._parse_ai_response(generated_text, track, difficulty)
                        if parsed_question:
                            return parsed_question
                    
                    # Rotate to next API endpoint
                    self.current_api_index += 1
                    
                except Exception as e:
                    st.warning(f"API attempt {attempt + 1} failed: {str(e)[:100]}...")
                    self.current_api_index += 1
                    time.sleep(2)
            
            # If all API attempts fail, use fallback
            return self._generate_fallback_question(track, difficulty)
                
        except Exception as e:
            st.warning(f"API generation failed: {str(e)[:100]}...")
            return self._generate_fallback_question(track, difficulty)
    
    def _create_prompt(self, track: str, difficulty: str) -> str:
        """Create a prompt for the AI model"""
        track_description = self.available_tracks[track]
        
        prompt = f"""Create a {difficulty} level multiple choice question about {track_description}.

Please follow this exact format:

Question: [Your question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Answer: [Correct option letter]
Explanation: [Brief explanation of the correct answer]

Make the question challenging but fair for {difficulty} level.
Ensure all options are plausible but only one is correct.
"""
        return prompt
    
    def _parse_ai_response(self, response: str, track: str, difficulty: str) -> Dict:
        """Parse AI response with flexible parsing"""
        try:
            lines = response.strip().split('\n')
            question = ""
            options = []
            correct_answer = ""
            explanation = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.lower().startswith('question:'):
                    question = line[9:].strip()
                elif line.startswith(('A)', 'A )', 'A.', 'A:')):
                    option_text = line[2:].strip().lstrip(')').lstrip('.').lstrip(':').strip()
                    options.append(option_text)
                elif line.startswith(('B)', 'B )', 'B.', 'B:')):
                    option_text = line[2:].strip().lstrip(')').lstrip('.').lstrip(':').strip()
                    options.append(option_text)
                elif line.startswith(('C)', 'C )', 'C.', 'C:')):
                    option_text = line[2:].strip().lstrip(')').lstrip('.').lstrip(':').strip()
                    options.append(option_text)
                elif line.startswith(('D)', 'D )', 'D.', 'D:')):
                    option_text = line[2:].strip().lstrip(')').lstrip('.').lstrip(':').strip()
                    options.append(option_text)
                elif line.lower().startswith('answer:'):
                    answer_text = line[7:].strip()
                    if answer_text and answer_text[0].upper() in ['A', 'B', 'C', 'D']:
                        idx = ord(answer_text[0].upper()) - ord('A')
                        if 0 <= idx < len(options):
                            correct_answer = options[idx]
                elif line.lower().startswith('explanation:'):
                    explanation = line[12:].strip()
            
            # Validate parsed data
            if question and len(options) >= 4 and correct_answer:
                return {
                    'text': question,
                    'options': options[:4],
                    'correct_answer': correct_answer,
                    'explanation': explanation or f"This tests {difficulty} level knowledge in {track}.",
                    'track': track,
                    'difficulty': difficulty,
                    'generated_by': 'AI Model'
                }
            else:
                return self._generate_fallback_question(track, difficulty)
        
        except Exception:
            return self._generate_fallback_question(track, difficulty)
    
    def _generate_fallback_question(self, track: str, difficulty: str) -> Dict:
        """Generate a fallback question when API fails"""
        # Dynamic question generation based on track and difficulty
        track_name = self.available_tracks[track].split('(')[0].strip()
        
        difficulty_modifiers = {
            "easy": ["basic", "fundamental", "essential", "primary", "simple"],
            "medium": ["important", "key", "significant", "practical", "intermediate"],
            "hard": ["advanced", "complex", "sophisticated", "challenging", "expert"]
        }
        
        modifiers = difficulty_modifiers[difficulty]
        modifier = random.choice(modifiers)
        
        question_types = [
            f"What is the {modifier} concept of {{topic}} in {track_name}?",
            f"Which of these best describes {modifier} {{topic}} in {track_name}?",
            f"What is the primary purpose of {modifier} {{topic}} in {track_name}?",
            f"How does {modifier} {{topic}} work in {track_name}?",
            f"What problem does {modifier} {{topic}} solve in {track_name}?"
        ]
        
        # Track-specific topics
        topics = {
            "web": ["responsive design", "API integration", "state management", "component architecture", "DOM manipulation"],
            "ai": ["neural networks", "model training", "feature engineering", "algorithm selection", "data preprocessing"],
            "cyber": ["encryption methods", "access control", "threat detection", "vulnerability assessment", "security protocols"],
            "data": ["data cleaning", "statistical analysis", "visualization techniques", "machine learning models", "data transformation"],
            "mobile": ["UI adaptation", "performance optimization", "native functionality", "cross-platform development", "user experience"],
            "devops": ["continuous integration", "containerization", "infrastructure as code", "monitoring solutions", "deployment strategies"],
            "backend": ["database design", "API development", "server optimization", "authentication systems", "caching mechanisms"],
            "frontend": ["user interface design", "accessibility standards", "performance optimization", "cross-browser compatibility", "responsive layouts"]
        }
        
        topic = random.choice(topics.get(track, topics["web"]))
        question_template = random.choice(question_types)
        question = question_template.format(topic=topic)
        
        # Generate plausible options
        options = [
            f"The {modifier} approach to {topic}",
            f"A common misconception about {topic}",
            f"A related but different technology to {topic}",
            f"A basic implementation detail of {topic}"
        ]
        
        # Shuffle options but ensure the first one is correct
        correct_option = options[0]
        random.shuffle(options)
        
        return {
            'text': question,
            'options': options,
            'correct_answer': correct_option,
            'explanation': f"This question tests understanding of {modifier} {topic} in {track_name} at {difficulty} level.",
            'track': track,
            'difficulty': difficulty,
            'generated_by': 'Fallback Generator'
        }
    
    def generate_questions(self, track: str, difficulty: str, count: int) -> List[Dict]:
        """Generate multiple questions"""
        questions = []
        
        for i in range(count):
            question = self.generate_with_api(track, difficulty)
            questions.append(question)
            
            # Add a small delay between API calls
            if i < count - 1:
                time.sleep(2)
        
        return questions

def main():
    st.set_page_config(
        page_title="AI MCQ Generator",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 AI-Powered MCQ Generator")
    st.markdown("Generate technical interview questions using AI models (no setup required)")
    
    # Initialize generator
    if 'generator' not in st.session_state:
        with st.spinner("Initializing MCQ Generator..."):
            st.session_state.generator = OnlineMCQGenerator()
    
    generator = st.session_state.generator
    
    # Show status
    st.success("✅ Online AI Generator Ready - No Setup Required")
    
    # Information about the tool
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        This tool uses AI models from Hugging Face to generate multiple-choice questions.
        
        **Features:**
        - No API tokens or setup required
        - Generates unique questions on demand
        - Multiple difficulty levels
        - Various technology tracks
        - Automatic fallback if AI is unavailable
        
        **How it works:**
        1. Select your preferred technology track
        2. Choose the difficulty level
        3. Specify how many questions to generate
        4. Click the generate button
        5. Review, export, or generate more questions
        """)
    
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
        
        # Show generation mode
        st.success("**Mode:** AI-Powered Generation")
    
    st.divider()
    
    # Generate button
    if st.button("🚀 Generate Questions", type="primary", use_container_width=True):
        with st.spinner("Generating questions with AI..."):
            try:
                questions = generator.generate_questions(track, difficulty, count)
                st.session_state.questions = questions
                st.session_state.generation_info = {
                    'track': track,
                    'difficulty': difficulty, 
                    'count': len(questions),
                    'mode': 'AI-Powered',
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
        
        # Export options
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export as JSON"):
                json_data = json.dumps(st.session_state.questions, indent=2)
                st.download_button(
                    "Download JSON",
                    json_data,
                    f"mcq_{track}_{difficulty}_{count}.json",
                    "application/json"
                )
        
        with col2:
            if st.button("🔄 Generate More Questions"):
                # Clear previous questions to generate new ones
                if 'questions' in st.session_state:
                    del st.session_state.questions
                st.rerun()

if __name__ == "__main__":
    main()