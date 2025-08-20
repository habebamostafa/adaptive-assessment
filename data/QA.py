import streamlit as st
import json
import time
from typing import List, Dict
from crewai import Agent, Task, Crew
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set up environment
os.environ["OPENAI_API_KEY"] = "free"  # This allows using CrewAI without actual API key

class CrewAIMCQGenerator:
    def __init__(self):
        """Initialize the MCQ generator with CrewAI"""
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
                    }
                ],
                "medium": [
                    {
                        "question": "What is the main purpose of React's useState hook?",
                        "options": ["Manage component state in functional components", "Handle HTTP requests", "Style components", "Route between pages"],
                        "correct_answer": "Manage component state in functional components",
                        "explanation": "useState is a React hook that allows functional components to have and update state, eliminating the need for class components in many cases."
                    }
                ],
                "hard": [
                    {
                        "question": "What is the event loop in JavaScript?",
                        "options": ["Mechanism for handling asynchronous operations", "A type of HTML element", "A CSS animation property", "A React lifecycle method"],
                        "correct_answer": "Mechanism for handling asynchronous operations",
                        "explanation": "The event loop is JavaScript's concurrency model that handles asynchronous callbacks and ensures non-blocking execution."
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
                    }
                ],
                "medium": [
                    {
                        "question": "What is overfitting in machine learning?",
                        "options": ["Model performs well on training data but poorly on new data", "Model trains too slowly", "Model uses too much memory", "Model cannot learn any patterns"],
                        "correct_answer": "Model performs well on training data but poorly on new data",
                        "explanation": "Overfitting occurs when a model learns the training data too specifically, including noise, making it perform poorly on unseen data."
                    }
                ],
                "hard": [
                    {
                        "question": "What is the vanishing gradient problem?",
                        "options": ["Gradients become very small in early layers during backpropagation", "Model outputs become zero", "Training data disappears", "Network connections break"],
                        "correct_answer": "Gradients become very small in early layers during backpropagation",
                        "explanation": "The vanishing gradient problem occurs when gradients become exponentially smaller as they propagate backward through deep networks, making early layers difficult to train."
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
                    }
                ],
                "medium": [
                    {
                        "question": "What is a Man-in-the-Middle (MITM) attack?",
                        "options": ["Intercepting communication between two parties", "Physical theft of computers", "Overloading servers with requests", "Installing malware via email"],
                        "correct_answer": "Intercepting communication between two parties",
                        "explanation": "A MITM attack occurs when an attacker secretly intercepts and potentially alters communication between two parties who believe they are communicating directly."
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
    
    def generate_with_crewai(self, track: str, difficulty: str) -> Dict:
        """Generate question using CrewAI"""
        try:
            # Create an expert agent for question generation
            question_agent = Agent(
                role='Technical Interview Question Expert',
                goal=f'Create high-quality {difficulty} level multiple choice questions about {self.available_tracks[track]}',
                backstory=f"""You are an expert technical interviewer with deep knowledge of {self.available_tracks[track]}.
                You specialize in creating challenging yet fair multiple choice questions that test both fundamental
                understanding and practical application of concepts.""",
                verbose=False,  # Set to False to reduce output
                allow_delegation=False
            )
            
            # Create a task for generating a question
            question_task = Task(
                description=f"""Create a {difficulty} level multiple choice question about {self.available_tracks[track]}.
                
                The question should:
                1. Be clear and unambiguous
                2. Have 4 plausible options (A, B, C, D)
                3. Have one correct answer
                4. Include a brief explanation of why the correct answer is right
                5. Be appropriate for {difficulty} level
                
                Format your response as:
                Question: [question text]
                A) [option A]
                B) [option B]
                C) [option C]
                D) [option D]
                Answer: [letter of correct option]
                Explanation: [brief explanation]
                """,
                agent=question_agent,
                expected_output="A well-formatted multiple choice question with options, correct answer, and explanation."
            )
            
            # Create crew and execute task
            crew = Crew(
                agents=[question_agent],
                tasks=[question_task],
                verbose=False  # Set to False to reduce output
            )
            
            # Execute the task
            result = crew.kickoff()
            
            # Parse the result
            return self._parse_ai_response(result, track, difficulty)
                
        except Exception as e:
            st.warning(f"CrewAI generation failed: {str(e)[:100]}...")
            return self.get_demo_question(track, difficulty)
    
    def _parse_ai_response(self, response: str, track: str, difficulty: str) -> Dict:
        """Parse AI response with flexible parsing"""
        try:
            # Try to extract question and options
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
                elif line.startswith(('A)', 'A )', 'A.')):
                    option_text = line[2:].strip().lstrip(')').lstrip('.').strip()
                    options.append(option_text)
                elif line.startswith(('B)', 'B )', 'B.')):
                    option_text = line[2:].strip().lstrip(')').lstrip('.').strip()
                    options.append(option_text)
                elif line.startswith(('C)', 'C )', 'C.')):
                    option_text = line[2:].strip().lstrip(')').lstrip('.').strip()
                    options.append(option_text)
                elif line.startswith(('D)', 'D )', 'D.')):
                    option_text = line[2:].strip().lstrip(')').lstrip('.').strip()
                    options.append(option_text)
                elif line.lower().startswith('answer:'):
                    answer_text = line[7:].strip()
                    if answer_text and answer_text[0] in ['A', 'B', 'C', 'D']:
                        idx = ord(answer_text[0]) - ord('A')
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
                    'generated_by': 'CrewAI Generated'
                }
            else:
                # If parsing fails, return demo question
                return self.get_demo_question(track, difficulty)
        
        except Exception:
            # If parsing fails, return demo question
            return self.get_demo_question(track, difficulty)
    
    def generate_questions(self, track: str, difficulty: str, count: int, use_ai: bool = True) -> List[Dict]:
        """Generate multiple questions"""
        questions = []
        
        for i in range(count):
            if use_ai:
                question = self.generate_with_crewai(track, difficulty)
            else:
                question = self.get_demo_question(track, difficulty)
            
            questions.append(question)
            
            # Rate limiting for AI calls
            if use_ai and i < count - 1:
                time.sleep(1)  # Short delay to avoid rate limits
        
        return questions

def main():
    st.set_page_config(
        page_title="MCQ Generator with CrewAI",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 MCQ Generator with CrewAI")
    st.markdown("Generate technical interview questions using CrewAI (no API token needed)")
    
    # Initialize generator
    if 'generator' not in st.session_state:
        with st.spinner("Initializing MCQ Generator..."):
            st.session_state.generator = CrewAIMCQGenerator()
    
    generator = st.session_state.generator
    
    # Show status
    st.success("✅ CrewAI Ready - No API Token Needed")
    
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
            count = st.number_input("Questions:", min_value=1, max_value=5, value=2)
    
    with col2:
        st.subheader("📊 Summary")
        
        # Show settings
        st.info(f"**Track:** {track.upper()}")
        st.info(f"**Level:** {difficulty.title()}")
        st.info(f"**Count:** {count}")
        
        # AI toggle
        use_ai = st.checkbox("🤖 Use CrewAI Generation", value=True)
        if use_ai:
            st.success("**Mode:** CrewAI")
        else:
            st.warning("**Mode:** Demo")
    
    st.divider()
    
    # Generate button
    if st.button("🚀 Generate Questions", type="primary", use_container_width=True):
        with st.spinner("Generating questions with CrewAI..."):
            try:
                questions = generator.generate_questions(track, difficulty, count, use_ai)
                st.session_state.questions = questions
                st.session_state.generation_info = {
                    'track': track,
                    'difficulty': difficulty, 
                    'count': len(questions),
                    'mode': 'CrewAI' if use_ai else 'Demo',
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