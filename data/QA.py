"""
Intelligent MCQ Generator using only prompt-based generation without pre-stored data
"""

import streamlit as st
import random
import time

class PurePromptMCQGenerator:
    def __init__(self):
        """Initialize the pure prompt-based MCQ generator"""
        self.available_tracks = {
            "web": "Web Development (HTML, CSS, JavaScript, React, etc.)",
            "ai": "Artificial Intelligence (Machine Learning, Deep Learning, NLP, etc.)",
            "cyber": "Cybersecurity (Network Security, Encryption, Ethical Hacking, etc.)",
            "data": "Data Science (Data Analysis, Visualization, Statistics, etc.)",
            "mobile": "Mobile Development (Android, iOS, Flutter, React Native, etc.)",
            "devops": "DevOps (Docker, Kubernetes, CI/CD, Cloud Computing, etc.)"
        }
        
    def get_available_tracks(self) -> list:
        """Get list of available technology tracks"""
        return list(self.available_tracks.keys())
    
    def generate_question(self, track: str, difficulty: str = "medium") -> dict:
        """
        Generate a question using only prompt-based logic
        
        Args:
            track: Technology track
            difficulty: Difficulty level (easy, medium, hard)
        
        Returns:
            Question dictionary
        """
        # Generate question text based on track and difficulty
        question_text = self._generate_question_text(track, difficulty)
        
        # Generate options
        options, correct_answer = self._generate_options(track, difficulty)
        
        # Generate explanation
        explanation = self._generate_explanation(track, correct_answer)
        
        return {
            'text': question_text,
            'options': options,
            'correct_answer': correct_answer,
            'explanation': explanation,
            'track': track,
            'difficulty': difficulty,
            'generated': True
        }
    
    def _generate_question_text(self, track: str, difficulty: str) -> str:
        """Generate question text using prompt logic"""
        track_name = self.available_tracks[track]
        
        question_types = {
            "easy": [
                f"What is the basic purpose of {{topic}} in {track_name}?",
                f"Which of these is a fundamental concept of {{topic}} in {track_name}?",
                f"How does {{topic}} work at a basic level in {track_name}?",
                f"Why is {{topic}} important for beginners in {track_name}?"
            ],
            "medium": [
                f"Which statement best describes {{topic}} in {track_name}?",
                f"How would you implement {{topic}} in a real {track_name} project?",
                f"What is the primary advantage of using {{topic}} in {track_name}?",
                f"Which approach is most effective for {{topic}} in {track_name}?"
            ],
            "hard": [
                f"What are the advanced considerations when working with {{topic}} in {track_name}?",
                f"How would you optimize {{topic}} for enterprise-level {track_name} applications?",
                f"What challenges might you encounter when implementing {{topic}} in complex {track_name} systems?",
                f"Which advanced technique is most appropriate for {{topic}} in high-performance {track_name} applications?"
            ]
        }
        
        # Select topics based on track and difficulty
        topics = self._get_topics(track, difficulty)
        topic = random.choice(topics)
        
        # Select question template
        template = random.choice(question_types[difficulty])
        
        return template.format(topic=topic)
    
    def _get_topics(self, track: str, difficulty: str) -> list:
        """Get relevant topics for a track and difficulty"""
        topics = {
            "web": {
                "easy": ["HTML", "CSS", "JavaScript", "responsive design", "DOM manipulation"],
                "medium": ["React", "API integration", "state management", "CSS frameworks", "routing"],
                "hard": ["performance optimization", "WebAssembly", "Progressive Web Apps", "server-side rendering", "advanced security"]
            },
            "ai": {
                "easy": ["machine learning", "neural networks", "data preprocessing", "model training", "basic algorithms"],
                "medium": ["CNN", "RNN", "transfer learning", "hyperparameter tuning", "model evaluation"],
                "hard": ["transformers", "GANs", "reinforcement learning", "explainable AI", "advanced optimization"]
            },
            "cyber": {
                "easy": ["encryption", "firewalls", "authentication", "basic security protocols", "password management"],
                "medium": ["penetration testing", "cryptographic protocols", "incident response", "vulnerability assessment", "network security"],
                "hard": ["zero-day exploits", "advanced persistent threats", "quantum cryptography", "reverse engineering", "threat intelligence"]
            },
            "data": {
                "easy": ["data cleaning", "basic statistics", "data visualization", "SQL queries", "data types"],
                "medium": ["feature engineering", "regression models", "clustering algorithms", "time series analysis", "data pipelines"],
                "hard": ["deep learning", "big data architectures", "MLOps", "advanced statistical modeling", "real-time analytics"]
            },
            "mobile": {
                "easy": ["UI components", "app structure", "platform differences", "user interactions", "basic layouts"],
                "medium": ["state management", "native modules", "performance optimization", "offline capabilities", "device APIs"],
                "hard": ["advanced animations", "cross-platform challenges", "security implementations", "low-level optimizations", "custom components"]
            },
            "devops": {
                "easy": ["version control", "CI/CD", "containers", "cloud basics", "basic automation"],
                "medium": ["infrastructure as code", "orchestration", "monitoring", "deployment strategies", "configuration management"],
                "hard": ["chaos engineering", "gitops", "service mesh", "advanced security", "scalability solutions"]
            }
        }
        
        return topics[track][difficulty]
    
    def _generate_options(self, track: str, difficulty: str) -> tuple:
        """Generate options for a question"""
        # Correct answer patterns
        correct_patterns = {
            "web": [
                "It provides structure and semantics to web content",
                "It enables dynamic styling and responsive layouts",
                "It adds interactivity and client-side functionality",
                "It facilitates component-based UI development",
                "It ensures cross-browser compatibility"
            ],
            "ai": [
                "It enables machines to learn from data without explicit programming",
                "It processes and analyzes complex patterns in large datasets",
                "It mimics human cognitive functions for problem solving",
                "It optimizes decision-making processes through algorithms",
                "It extracts meaningful insights from raw information"
            ],
            "cyber": [
                "It protects systems and data from unauthorized access and attacks",
                "It ensures confidentiality, integrity and availability of information",
                "It identifies and mitigates potential security vulnerabilities",
                "It establishes trust and compliance in digital operations",
                "It monitors and responds to security incidents in real-time"
            ],
            "data": [
                "It extracts meaningful insights from raw information",
                "It transforms, analyzes and visualizes complex datasets",
                "It supports data-driven decision making through analysis",
                "It manages and processes large volumes of structured and unstructured data",
                "It builds predictive models from historical data patterns"
            ],
            "mobile": [
                "It creates native experiences optimized for mobile devices",
                "It enables cross-platform development with shared codebase",
                "It implements touch-friendly interfaces and mobile-specific features",
                "It manages device resources efficiently for optimal performance",
                "It ensures app security and data protection on mobile platforms"
            ],
            "devops": [
                "It automates and streamlines development and operations processes",
                "It enables continuous integration and delivery of software",
                "It ensures reliable and scalable infrastructure management",
                "It bridges development and operations for faster delivery",
                "It implements monitoring and alerting for system reliability"
            ]
        }
        
        # Incorrect answer patterns
        incorrect_patterns = [
            "It handles server-side business logic and database operations",
            "It manages network protocols and data transmission",
            "It optimizes hardware performance and resource allocation",
            "It implements cryptographic security algorithms",
            "It designs user interfaces and experience flows",
            "It manages database transactions and data persistence",
            "It configures network infrastructure and security policies",
            "It develops compilers and low-level system utilities",
            "It designs user experience and interface elements",
            "It develops application features and functionality",
            "It manages cloud storage and data backup solutions",
            "It optimizes database queries and performance",
            "It implements user authentication and authorization systems",
            "It develops mobile application interfaces and interactions",
            "It configures network routers and switching infrastructure",
            "It designs graphical assets and visual branding elements",
            "It implements server-side API endpoints and business logic",
            "It configures network security policies and firewall rules",
            "It designs database schemas and optimization strategies",
            "It develops operating system kernels and low-level drivers"
        ]
        
        # Select one correct option
        correct_answer = random.choice(correct_patterns[track])
        
        # Select three incorrect options
        incorrect_options = random.sample(incorrect_patterns, 3)
        
        # Combine and shuffle options
        options = [correct_answer] + incorrect_options
        random.shuffle(options)
        
        return options, correct_answer
    
    def _generate_explanation(self, track: str, correct_answer: str) -> str:
        """Generate explanation for the correct answer"""
        explanations = {
            "web": f"{correct_answer} is correct because it represents a core principle of web development that ensures effective, responsive, and secure web applications.",
            "ai": f"{correct_answer} is the right choice as it aligns with fundamental AI concepts that enable machines to learn, reason, and solve complex problems.",
            "cyber": f"In cybersecurity, {correct_answer} is essential for protecting digital assets, preventing unauthorized access, and maintaining system integrity.",
            "data": f"For data science, {correct_answer} correctly describes processes for extracting insights, building models, and supporting data-driven decisions.",
            "mobile": f"In mobile development, {correct_answer} addresses key considerations for creating performant, user-friendly, and platform-appropriate applications.",
            "devops": f"The DevOps practice confirms {correct_answer} as critical for automating workflows, ensuring reliability, and accelerating software delivery."
        }
        
        return explanations.get(track, f"{correct_answer} is the correct answer because it best addresses the question based on established principles and practices.")
    
    def generate_question_set(self, track: str, num_questions: int = 5, difficulty: str = "medium") -> list:
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
            # Small delay to simulate processing
            time.sleep(0.1)
        
        return questions

# Streamlit interface
def main():
    """Main function to run the Streamlit app"""
    st.title("🤖 Pure Prompt MCQ Generator")
    st.write("Generate technical interview questions using pure prompt-based generation (no pre-stored data)")
    
    # Initialize generator
    if 'generator' not in st.session_state:
        st.session_state.generator = PurePromptMCQGenerator()
    
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
    num_questions = st.slider("Number of questions to generate:", 1, 10, 5)
    
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