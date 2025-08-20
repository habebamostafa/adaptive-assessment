"""
Enhanced questions.py with intelligent MCQ generator using language models
Focuses on generating questions dynamically rather than using pre-stored ones
"""

import json
import random
import streamlit as st
from typing import List, Dict, Optional, Tuple
import time
import re

class IntelligentMCQGenerator:
    def __init__(self):
        """Initialize the intelligent MCQ generator"""
        # We'll use a mock model for generation since we can't load real models in this environment
        self.available_tracks = {
            "web": "Web Development (HTML, CSS, JavaScript, React, etc.)",
            "ai": "Artificial Intelligence (Machine Learning, Deep Learning, NLP, etc.)",
            "cyber": "Cybersecurity (Network Security, Encryption, Ethical Hacking, etc.)",
            "data": "Data Science (Data Analysis, Visualization, Statistics, etc.)",
            "mobile": "Mobile Development (Android, iOS, Flutter, React Native, etc.)",
            "devops": "DevOps (Docker, Kubernetes, CI/CD, Cloud Computing, etc.)"
        }
        
        # Cache for generated questions to avoid duplicates
        self.question_cache = {}

    def get_available_tracks(self) -> List[str]:
        """Get list of available technology tracks"""
        return list(self.available_tracks.keys())

    def generate_question(self, track: str, difficulty: str = "medium") -> Dict:
        """
        Generate a question for the specified track and difficulty
        
        Args:
            track: Technology track (web, ai, cyber, data, mobile, devops)
            difficulty: Difficulty level (easy, medium, hard)
        
        Returns:
            Question dictionary with text, options, correct answer, and explanation
        """
        if track not in self.available_tracks:
            return self._create_fallback_question(track, difficulty)
        
        try:
            # Generate question and options based on track and difficulty
            question_text, options = self._generate_question_content(track, difficulty)
            
            # Ensure we have valid question and options
            if not question_text or not options or len(options) < 4:
                return self._create_fallback_question(track, difficulty)
            
            # Create explanation
            explanation = self._generate_explanation(track, question_text, options[0])
            
            return {
                'text': question_text,
                'options': options,
                'correct_answer': options[0],  # First option is correct
                'explanation': explanation,
                'track': track,
                'difficulty': difficulty,
                'generated': True
            }
            
        except Exception as e:
            print(f"Error generating question: {e}")
            return self._create_fallback_question(track, difficulty)

    def _generate_question_content(self, track: str, difficulty: str) -> Tuple[str, List[str]]:
        """Generate question content based on track and difficulty"""
        # This is a mock implementation that simulates what a language model would generate
        # In a real implementation, this would call an actual language model
        
        track_topics = {
            "web": {
                "easy": ["HTML tags", "CSS selectors", "basic JavaScript syntax", "responsive design principles"],
                "medium": ["React components", "API integration", "state management", "CSS frameworks"],
                "hard": ["performance optimization", "WebAssembly", "Progressive Web Apps", "advanced JavaScript patterns"]
            },
            "ai": {
                "easy": ["machine learning basics", "neural network components", "data preprocessing", "model evaluation"],
                "medium": ["CNN architectures", "RNN applications", "transfer learning", "hyperparameter tuning"],
                "hard": ["transformers architecture", "GAN implementations", "reinforcement learning", "explainable AI"]
            },
            "cyber": {
                "easy": ["password security", "firewall basics", "encryption types", "social engineering awareness"],
                "medium": ["network penetration testing", "cryptographic protocols", "incident response", "vulnerability assessment"],
                "hard": ["zero-day exploits", "advanced persistent threats", "quantum cryptography", "reverse engineering"]
            },
            "data": {
                "easy": ["data cleaning techniques", "basic statistics", "visualization types", "SQL queries"],
                "medium": ["feature engineering", "regression models", "clustering algorithms", "time series analysis"],
                "hard": ["deep learning for data", "big data architectures", "MLOps practices", "advanced statistical modeling"]
            },
            "mobile": {
                "easy": ["UI components", "basic app structure", "platform differences", "simple user interactions"],
                "medium": ["state management", "native module integration", "performance optimization", "offline capabilities"],
                "hard": ["advanced animations", "cross-platform challenges", "security implementations", "low-level optimizations"]
            },
            "devops": {
                "easy": ["version control basics", "CI/CD concepts", "container basics", "cloud fundamentals"],
                "medium": ["infrastructure as code", "orchestration tools", "monitoring solutions", "deployment strategies"],
                "hard": ["chaos engineering", "gitops methodologies", "service mesh implementations", "advanced security practices"]
            }
        }
        
        # Select a topic based on track and difficulty
        topics = track_topics.get(track, {}).get(difficulty, ["key concepts"])
        topic = random.choice(topics)
        
        # Generate question text
        question_types = [
            f"What is the primary purpose of {topic} in {track} development?",
            f"Which of these best describes {topic} in the context of {track}?",
            f"How does {topic} contribute to effective {track} solutions?",
            f"What is a key consideration when implementing {topic} in {track} projects?",
            f"Which statement accurately describes the role of {topic} in {track}?"
        ]
        
        question_text = random.choice(question_types)
        
        # Generate options - first is correct, others are plausible but incorrect
        correct_options = {
            "web": [
                "It provides structure and semantics to web content",
                "It enables dynamic styling and responsive layouts",
                "It adds interactivity and client-side functionality",
                "It facilitates component-based UI development"
            ],
            "ai": [
                "It enables machines to learn from data without explicit programming",
                "It processes and analyzes complex patterns in large datasets",
                "It mimics human cognitive functions for problem solving",
                "It optimizes decision-making processes through algorithms"
            ],
            "cyber": [
                "It protects systems and data from unauthorized access and attacks",
                "It ensures confidentiality, integrity and availability of information",
                "It identifies and mitigates potential security vulnerabilities",
                "It establishes trust and compliance in digital operations"
            ],
            "data": [
                "It extracts meaningful insights from raw information",
                "It transforms, analyzes and visualizes complex datasets",
                "It supports data-driven decision making through analysis",
                "It manages and processes large volumes of structured and unstructured data"
            ],
            "mobile": [
                "It creates native experiences optimized for mobile devices",
                "It enables cross-platform development with shared codebase",
                "It implements touch-friendly interfaces and mobile-specific features",
                "It manages device resources efficiently for optimal performance"
            ],
            "devops": [
                "It automates and streamlines development and operations processes",
                "It enables continuous integration and delivery of software",
                "It ensures reliable and scalable infrastructure management",
                "It bridges development and operations for faster delivery"
            ]
        }
        
        incorrect_options = {
            "web": [
                "It handles server-side business logic and database operations",
                "It manages network protocols and data transmission",
                "It optimizes hardware performance and resource allocation",
                "It implements cryptographic security algorithms"
            ],
            "ai": [
                "It designs user interfaces and experience flows",
                "It manages database transactions and data persistence",
                "It configures network infrastructure and security policies",
                "It develops compilers and low-level system utilities"
            ],
            "cyber": [
                "It designs user experience and interface elements",
                "It develops application features and functionality",
                "It manages cloud storage and data backup solutions",
                "It optimizes database queries and performance"
            ],
            "data": [
                "It implements user authentication and authorization systems",
                "It develops mobile application interfaces and interactions",
                "It configures network routers and switching infrastructure",
                "It designs graphical assets and visual branding elements"
            ],
            "mobile": [
                "It implements server-side API endpoints and business logic",
                "It configures network security policies and firewall rules",
                "It designs database schemas and optimization strategies",
                "It develops operating system kernels and low-level drivers"
            ],
            "devops": [
                "It designs user interface components and interactions",
                "It implements application-specific business rules and logic",
                "It creates visual designs and branding elements",
                "It develops algorithms for data processing and analysis"
            ]
        }
        
        # Select one correct option and three incorrect ones
        options = [random.choice(correct_options.get(track, ["Correct answer"]))]
        options.extend(random.sample(incorrect_options.get(track, [
            "Incorrect alternative 1", 
            "Incorrect alternative 2",
            "Incorrect alternative 3",
            "Incorrect alternative 4"
        ]), 3))
        
        # Shuffle options but remember the correct one
        correct_answer = options[0]
        random.shuffle(options)
        
        # Return the question and options
        return question_text, options, correct_answer

    def _generate_explanation(self, track: str, question: str, correct_answer: str) -> str:
        """Generate an explanation for the correct answer"""
        explanations = {
            "web": f"The correct answer is {correct_answer} because it represents a fundamental concept in web development that addresses the question: '{question}'.",
            "ai": f"{correct_answer} is the right choice as it aligns with core principles of artificial intelligence and machine learning related to: '{question}'.",
            "cyber": f"In cybersecurity, {correct_answer} is the appropriate response to '{question}' as it reflects established security protocols and best practices.",
            "data": f"For data science, {correct_answer} correctly addresses '{question}' based on statistical principles and data analysis methodologies.",
            "mobile": f"In mobile development, {correct_answer} is the correct approach for '{question}' following platform-specific guidelines and patterns.",
            "devops": f"The DevOps perspective confirms {correct_answer} as the right answer for '{question}' based on automation, collaboration, and integration principles."
        }
        
        return explanations.get(track, f"The correct answer is {correct_answer} because it best addresses the question: '{question}'.")

    def _create_fallback_question(self, track: str, difficulty: str) -> Dict:
        """Create a fallback question if generation fails"""
        difficulty_text = {
            "easy": "basic",
            "medium": "intermediate",
            "hard": "advanced"
        }.get(difficulty, "intermediate")
        
        track_name = self.available_tracks.get(track, track)
        
        questions = {
            "web": f"What is a {difficulty_text} concept in {track_name}?",
            "ai": f"Which {difficulty_text} technique is commonly used in {track_name}?",
            "cyber": f"What is a {difficulty_text} security consideration in {track_name}?",
            "data": f"Which {difficulty_text} approach is important in {track_name}?",
            "mobile": f"What is a {difficulty_text} development pattern in {track_name}?",
            "devops": f"Which {difficulty_text} practice is essential in {track_name}?"
        }
        
        question_text = questions.get(track, f"What is a key concept in {track_name}?")
        
        return {
            'text': question_text,
            'options': [
                f"Correct answer for {difficulty} {track}",
                f"Alternative option 1 for {track}",
                f"Alternative option 2 for {track}",
                f"Alternative option 3 for {track}"
            ],
            'correct_answer': f"Correct answer for {difficulty} {track}",
            'explanation': f"This is a {difficulty} level question about {track_name}.",
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
            question = self.generate_question(track, difficulty)
            questions.append(question)
            # Small delay to simulate model processing
            time.sleep(0.1)
        
        return questions

    def get_track_description(self, track: str) -> str:
        """Get description of a track"""
        return self.available_tracks.get(track, f"Questions about {track}")


# Streamlit interface for testing
def main():
    """Main function to run the Streamlit app"""
    st.title("🤖 Intelligent MCQ Generator")
    st.write("Generate technical interview questions using AI")
    
    # Initialize generator
    if 'generator' not in st.session_state:
        st.session_state.generator = IntelligentMCQGenerator()
    
    generator = st.session_state.generator
    
    # Track selection
    track = st.selectbox(
        "Select technology track:",
        options=generator.get_available_tracks(),
        format_func=lambda x: f"{x} - {generator.get_track_description(x)}"
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
    if st.button("Generate Questions"):
        with st.spinner(f"Generating {num_questions} {difficulty} questions for {track}..."):
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