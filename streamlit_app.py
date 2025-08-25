# app.py - Dynamic AI-Powered Adaptive Assessment Platform
import streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import random
import requests

# Configure Streamlit page
st.set_page_config(
    page_title="AI Adaptive Assessment Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class User:
    username: str
    password_hash: str
    email: str
    created_at: datetime
    role: str = "student"
    profile_data: Dict = None

@dataclass
class AssessmentSession:
    session_id: str
    username: str
    track: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    final_score: float = 0.0
    questions_answered: int = 0
    ability_level: float = 0.5
    performance_data: Dict = None
    learning_progress: Dict = None

class SimpleDatabase:
    """Simple in-memory database for users and sessions"""
    
    def __init__(self):
        self.users = {}
        self.sessions = {}
        self.load_sample_data()
    
    def load_sample_data(self):
        """Load some sample users for testing"""
        sample_users = [
            {"username": "admin", "password": "admin123", "email": "admin@test.com", "role": "admin"},
            {"username": "student1", "password": "pass123", "email": "student1@test.com", "role": "student"},
            {"username": "teacher", "password": "teach123", "email": "teacher@test.com", "role": "teacher"}
        ]
        
        for user_data in sample_users:
            self.create_user(
                user_data["username"],
                user_data["password"], 
                user_data["email"],
                user_data["role"]
            )
    
    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, password: str, email: str, role: str = "student") -> bool:
        if username in self.users:
            return False
        
        self.users[username] = User(
            username=username,
            password_hash=self.hash_password(password),
            email=email,
            created_at=datetime.now(),
            role=role,
            profile_data={}
        )
        return True
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        if username not in self.users:
            return None
        
        user = self.users[username]
        if user.password_hash == self.hash_password(password):
            return user
        return None
    
    def get_user_sessions(self, username: str) -> List[AssessmentSession]:
        return [session for session in self.sessions.values() if session.username == username]
    
    def save_session(self, session: AssessmentSession):
        self.sessions[session.session_id] = session
    
    def get_all_users(self) -> List[User]:
        return list(self.users.values())
    
    def get_all_sessions(self) -> List[AssessmentSession]:
        return list(self.sessions.values())

class DynamicAIQuestionGenerator:
    """Generate questions dynamically using detailed AI prompts"""
    
    def __init__(self):
        self.question_history = {}  # Track what's been asked to avoid repetition
        self.learning_paths = {}    # Store dynamic learning paths per user
        
    def generate_learning_path(self, track: str, user_level: float = 0.5) -> Dict:
        """Generate a dynamic learning path based on track and user level"""
        
        # Determine current level
        if user_level < 0.3:
            current_level = "beginner"
        elif user_level < 0.7:
            current_level = "intermediate"  
        else:
            current_level = "advanced"
            
        prompt = self._create_learning_path_prompt(track, current_level)
        
        # In a real implementation, this would call an AI API
        # For demo, we'll simulate with structured responses
        return self._simulate_ai_learning_path(track, current_level)
    
    def _create_learning_path_prompt(self, track: str, level: str) -> str:
        """Create detailed prompt for learning path generation"""
        
        prompts = {
            "ai": f"""
            Create a progressive learning path for {level} level Artificial Intelligence and Machine Learning.
            
            Current learner level: {level}
            
            Requirements:
            1. List 6-8 specific topics in logical learning order
            2. For each topic, provide 3-4 key concepts that should be mastered
            3. Include prerequisite knowledge for each topic
            4. Suggest practical projects or exercises
            5. Ensure smooth progression from basics to more complex concepts
            
            Focus areas for {level} level:
            - Beginner: Python basics, basic ML concepts, statistics fundamentals
            - Intermediate: ML algorithms, data preprocessing, model evaluation
            - Advanced: Deep learning, specialized domains (NLP/CV), MLOps
            
            Format the response as a structured learning pathway.
            """,
            
            "web": f"""
            Create a progressive learning path for {level} level Web Development.
            
            Current learner level: {level}
            
            Requirements:
            1. List 6-8 specific topics in logical learning order
            2. For each topic, provide 3-4 key concepts that should be mastered  
            3. Include prerequisite knowledge for each topic
            4. Suggest practical projects or exercises
            5. Ensure smooth progression from basics to more complex concepts
            
            Focus areas for {level} level:
            - Beginner: HTML/CSS basics, JavaScript fundamentals, DOM manipulation
            - Intermediate: Frameworks (React/Vue), APIs, responsive design
            - Advanced: Performance optimization, advanced patterns, full-stack architecture
            
            Format the response as a structured learning pathway.
            """,
            
            "cyber": f"""
            Create a progressive learning path for {level} level Cybersecurity.
            
            Current learner level: {level}
            
            Requirements:
            1. List 6-8 specific topics in logical learning order
            2. For each topic, provide 3-4 key concepts that should be mastered
            3. Include prerequisite knowledge for each topic  
            4. Suggest practical projects or exercises
            5. Ensure smooth progression from basics to more complex concepts
            
            Focus areas for {level} level:
            - Beginner: Network basics, common threats, basic security tools
            - Intermediate: Penetration testing, security protocols, incident response
            - Advanced: Advanced persistent threats, security architecture, compliance
            
            Format the response as a structured learning pathway.
            """
        }
        
        return prompts.get(track, prompts["ai"])
    
    def _simulate_ai_learning_path(self, track: str, level: str) -> Dict:
        """Simulate AI response for learning path (replace with real AI API)"""
        
        paths = {
            "ai": {
                "beginner": {
                    "topics": [
                        {
                            "name": "Python Programming Fundamentals",
                            "concepts": ["Variables and data types", "Control structures", "Functions", "Error handling"],
                            "description": "Master Python basics essential for AI development"
                        },
                        {
                            "name": "Mathematics for Machine Learning", 
                            "concepts": ["Linear algebra basics", "Statistics fundamentals", "Probability theory", "Calculus basics"],
                            "description": "Build mathematical foundation for understanding ML algorithms"
                        },
                        {
                            "name": "Introduction to Data Science",
                            "concepts": ["Data collection methods", "Data cleaning techniques", "Basic visualization", "Pandas and NumPy"],
                            "description": "Learn to work with data effectively"
                        },
                        {
                            "name": "Machine Learning Concepts",
                            "concepts": ["Supervised vs unsupervised learning", "Training and testing", "Overfitting and underfitting", "Model evaluation"],
                            "description": "Understand fundamental ML principles"
                        },
                        {
                            "name": "First ML Algorithms",
                            "concepts": ["Linear regression", "Logistic regression", "Decision trees", "K-means clustering"],
                            "description": "Implement basic machine learning algorithms"
                        }
                    ]
                },
                "intermediate": {
                    "topics": [
                        {
                            "name": "Advanced Python for ML",
                            "concepts": ["Object-oriented programming", "Libraries ecosystem", "Performance optimization", "Code organization"],
                            "description": "Advanced Python skills for ML projects"
                        },
                        {
                            "name": "Feature Engineering",
                            "concepts": ["Feature selection techniques", "Dimensionality reduction", "Feature scaling", "Handling categorical data"],
                            "description": "Master the art of preparing data for ML models"
                        },
                        {
                            "name": "Model Selection and Evaluation",
                            "concepts": ["Cross-validation", "Hyperparameter tuning", "Performance metrics", "Model comparison"],
                            "description": "Learn to evaluate and compare ML models effectively"
                        },
                        {
                            "name": "Ensemble Methods", 
                            "concepts": ["Random Forest", "Gradient Boosting", "Bagging", "Stacking"],
                            "description": "Combine multiple models for better performance"
                        },
                        {
                            "name": "Introduction to Neural Networks",
                            "concepts": ["Perceptrons", "Backpropagation", "Activation functions", "Network architecture"],
                            "description": "Understand the basics of artificial neural networks"
                        }
                    ]
                },
                "advanced": {
                    "topics": [
                        {
                            "name": "Deep Learning Architectures",
                            "concepts": ["Convolutional Neural Networks", "Recurrent Neural Networks", "Transformers", "Attention mechanisms"],
                            "description": "Master advanced neural network architectures"
                        },
                        {
                            "name": "Natural Language Processing",
                            "concepts": ["Text preprocessing", "Word embeddings", "Language models", "Named entity recognition"],
                            "description": "Apply AI to text and language understanding"
                        },
                        {
                            "name": "Computer Vision",
                            "concepts": ["Image preprocessing", "Object detection", "Image segmentation", "Transfer learning"],
                            "description": "Develop AI systems that can see and understand images"
                        },
                        {
                            "name": "MLOps and Production",
                            "concepts": ["Model deployment", "Monitoring", "Version control", "Continuous integration"],
                            "description": "Deploy and maintain ML models in production"
                        },
                        {
                            "name": "AI Ethics and Responsible AI",
                            "concepts": ["Bias detection", "Fairness metrics", "Interpretability", "Privacy preservation"],
                            "description": "Build ethical and responsible AI systems"
                        }
                    ]
                }
            },
            "web": {
                "beginner": {
                    "topics": [
                        {
                            "name": "HTML Structure and Semantics",
                            "concepts": ["Document structure", "Semantic elements", "Forms and inputs", "Accessibility basics"],
                            "description": "Build well-structured and accessible web content"
                        },
                        {
                            "name": "CSS Styling and Layout",
                            "concepts": ["Selectors and properties", "Box model", "Flexbox", "CSS Grid"],
                            "description": "Style and layout web pages effectively"
                        },
                        {
                            "name": "JavaScript Fundamentals",
                            "concepts": ["Variables and functions", "DOM manipulation", "Event handling", "Arrays and objects"],
                            "description": "Add interactivity to web pages"
                        },
                        {
                            "name": "Responsive Web Design",
                            "concepts": ["Media queries", "Mobile-first approach", "Flexible images", "Viewport meta tag"],
                            "description": "Create websites that work on all devices"
                        },
                        {
                            "name": "Web Development Tools",
                            "concepts": ["Browser DevTools", "Version control (Git)", "Code editors", "Basic debugging"],
                            "description": "Use essential tools for web development"
                        }
                    ]
                },
                "intermediate": {
                    "topics": [
                        {
                            "name": "Modern JavaScript (ES6+)",
                            "concepts": ["Arrow functions", "Destructuring", "Promises and async/await", "Modules"],
                            "description": "Master modern JavaScript features"
                        },
                        {
                            "name": "Frontend Frameworks",
                            "concepts": ["React fundamentals", "Component architecture", "State management", "React Router"],
                            "description": "Build complex user interfaces with React"
                        },
                        {
                            "name": "API Integration",
                            "concepts": ["RESTful APIs", "Fetch API", "Error handling", "Authentication"],
                            "description": "Connect frontend applications to backend services"
                        },
                        {
                            "name": "Build Tools and Workflow",
                            "concepts": ["Webpack", "npm/yarn", "Babel", "Development servers"],
                            "description": "Optimize development workflow and build processes"
                        },
                        {
                            "name": "Testing and Quality Assurance",
                            "concepts": ["Unit testing", "Integration testing", "Code linting", "Debugging techniques"],
                            "description": "Ensure code quality and reliability"
                        }
                    ]
                },
                "advanced": {
                    "topics": [
                        {
                            "name": "Advanced React Patterns",
                            "concepts": ["Higher-order components", "Render props", "Custom hooks", "Context API"],
                            "description": "Implement advanced React patterns and architectures"
                        },
                        {
                            "name": "Performance Optimization",
                            "concepts": ["Code splitting", "Lazy loading", "Memoization", "Bundle analysis"],
                            "description": "Optimize web applications for maximum performance"
                        },
                        {
                            "name": "Full-Stack Architecture",
                            "concepts": ["Server-side rendering", "Static site generation", "Microservices", "Database integration"],
                            "description": "Design and implement full-stack web applications"
                        },
                        {
                            "name": "Web Security",
                            "concepts": ["HTTPS", "Content Security Policy", "XSS prevention", "Authentication patterns"],
                            "description": "Secure web applications against common threats"
                        },
                        {
                            "name": "Progressive Web Apps",
                            "concepts": ["Service workers", "Web app manifest", "Offline functionality", "Push notifications"],
                            "description": "Build web apps that feel like native applications"
                        }
                    ]
                }
            },
            "cyber": {
                "beginner": {
                    "topics": [
                        {
                            "name": "Cybersecurity Fundamentals",
                            "concepts": ["CIA triad", "Risk assessment", "Threat landscape", "Security frameworks"],
                            "description": "Understand basic cybersecurity principles and concepts"
                        },
                        {
                            "name": "Network Security Basics",
                            "concepts": ["Network protocols", "Firewalls", "VPNs", "Network monitoring"],
                            "description": "Learn how to secure network infrastructure"
                        },
                        {
                            "name": "Common Security Threats",
                            "concepts": ["Malware types", "Social engineering", "Phishing", "Password attacks"],
                            "description": "Identify and understand common cybersecurity threats"
                        },
                        {
                            "name": "Basic Security Tools",
                            "concepts": ["Antivirus software", "Network scanners", "Vulnerability scanners", "Log analysis"],
                            "description": "Use essential cybersecurity tools and software"
                        },
                        {
                            "name": "Security Policies and Procedures",
                            "concepts": ["Access control", "Incident response", "Security awareness", "Compliance basics"],
                            "description": "Develop and implement basic security policies"
                        }
                    ]
                },
                "intermediate": {
                    "topics": [
                        {
                            "name": "Penetration Testing",
                            "concepts": ["Ethical hacking", "Vulnerability assessment", "Exploitation techniques", "Reporting"],
                            "description": "Learn to test systems for security vulnerabilities"
                        },
                        {
                            "name": "Cryptography Applications",
                            "concepts": ["Encryption algorithms", "Digital signatures", "PKI", "Hash functions"],
                            "description": "Apply cryptographic techniques for data protection"
                        },
                        {
                            "name": "Incident Response",
                            "concepts": ["Incident handling", "Digital forensics", "Recovery procedures", "Lessons learned"],
                            "description": "Respond effectively to security incidents"
                        },
                        {
                            "name": "Security Architecture",
                            "concepts": ["Defense in depth", "Zero trust", "Security controls", "Risk management"],
                            "description": "Design secure system architectures"
                        },
                        {
                            "name": "Advanced Threat Detection",
                            "concepts": ["SIEM systems", "Threat intelligence", "Behavioral analysis", "Machine learning in security"],
                            "description": "Detect and analyze advanced security threats"
                        }
                    ]
                }
            }
        }
        
        return paths.get(track, {}).get(level, {"topics": []})
    
    def generate_contextual_question(self, track: str, user_level: float, 
                                   performance_history: Dict, topic_focus: str = None) -> Dict:
        """Generate a contextual question based on user progress and performance"""
        
        # Determine difficulty based on recent performance
        difficulty = self._calculate_adaptive_difficulty(user_level, performance_history)
        
        # Get learning path for context
        learning_path = self.generate_learning_path(track, user_level)
        
        # Create detailed prompt for question generation
        prompt = self._create_question_prompt(track, difficulty, learning_path, 
                                            performance_history, topic_focus)
        
        # Generate question (simulate AI response)
        return self._simulate_ai_question_generation(track, difficulty, topic_focus, learning_path)
    
    def _calculate_adaptive_difficulty(self, user_level: float, performance_history: Dict) -> str:
        """Calculate appropriate difficulty based on user performance"""
        
        recent_scores = performance_history.get('recent_scores', [])
        
        if not recent_scores:
            # No history, use user level
            if user_level < 0.3:
                return "beginner"
            elif user_level < 0.7:
                return "intermediate"
            else:
                return "advanced"
        
        # Adjust based on recent performance
        avg_recent = sum(recent_scores[-3:]) / len(recent_scores[-3:])
        
        if avg_recent > 0.8:
            # Performing well, increase difficulty
            if user_level < 0.6:
                return "intermediate"
            else:
                return "advanced"
        elif avg_recent < 0.4:
            # Struggling, decrease difficulty
            if user_level > 0.4:
                return "beginner"
            else:
                return "beginner"
        else:
            # Moderate performance, maintain level
            if user_level < 0.3:
                return "beginner"
            elif user_level < 0.7:
                return "intermediate"
            else:
                return "advanced"
    
    def _create_question_prompt(self, track: str, difficulty: str, learning_path: Dict,
                               performance_history: Dict, topic_focus: str = None) -> str:
        """Create detailed prompt for AI question generation"""
        
        # Get current topic from learning path
        current_topic = topic_focus or self._get_current_topic(learning_path, performance_history)
        
        prompt = f"""
        Generate a {difficulty} level multiple-choice question for {track} development.
        
        Context:
        - Student level: {difficulty}
        - Current learning focus: {current_topic}
        - Track: {track}
        
        Learning Path Context:
        {json.dumps(learning_path, indent=2)}
        
        Question Requirements:
        1. Focus specifically on: {current_topic}
        2. Difficulty level: {difficulty}
        3. Include 4 plausible answer choices
        4. Provide detailed explanation for the correct answer
        5. Make the question practical and scenario-based
        6. Ensure it tests understanding, not just memorization
        7. Include real-world application context
        
        Question Style Guidelines:
        - Beginner: Basic concepts, definitions, simple applications
        - Intermediate: Implementation details, best practices, problem-solving
        - Advanced: Architecture decisions, optimization, complex scenarios
        
        Format the response as:
        {{
            "question": "Question text",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "Detailed explanation",
            "topic": "{current_topic}",
            "difficulty": "{difficulty}",
            "practical_context": "Real-world scenario this applies to"
        }}
        """
        
        return prompt
    
    def _get_current_topic(self, learning_path: Dict, performance_history: Dict) -> str:
        """Determine current topic based on learning progress"""
        
        topics = learning_path.get('topics', [])
        if not topics:
            return "General concepts"
        
        # Simple logic: progress through topics based on performance
        progress_index = min(len(topics) - 1, 
                           performance_history.get('topics_mastered', 0))
        
        return topics[progress_index]['name']
    
    def _simulate_ai_question_generation(self, track: str, difficulty: str, 
                                       topic_focus: str, learning_path: Dict) -> Dict:
        """Simulate AI question generation (replace with real AI API)"""
        
        # This is where you would call your actual AI API
        # For demo purposes, we'll generate contextual questions
        
        questions_bank = {
            "ai": {
                "beginner": {
                    "Python Programming Fundamentals": [
                        {
                            "question": "You want to store a list of student grades in Python. Which data type would be most appropriate?",
                            "options": ["list", "string", "integer", "boolean"],
                            "correct_answer": "list",
                            "explanation": "A list is the most appropriate data type for storing multiple values like student grades because it can hold multiple items, is mutable (can be changed), and supports various operations like adding, removing, and accessing elements by index.",
                            "practical_context": "Managing student data in an educational system"
                        },
                        {
                            "question": "What will be the output of this Python code?\n\n```python\nfor i in range(3):\n    print(i)\n```",
                            "options": ["0 1 2", "1 2 3", "0 1 2 3", "1 2"],
                            "correct_answer": "0 1 2",
                            "explanation": "The range(3) function generates numbers from 0 to 2 (3 is excluded). The for loop iterates through these values and prints each one on a new line: 0, then 1, then 2.",
                            "practical_context": "Iterating through data collections in data processing tasks"
                        }
                    ],
                    "Mathematics for Machine Learning": [
                        {
                            "question": "In machine learning, what does a high variance in your model typically indicate?",
                            "options": ["Overfitting to training data", "Underfitting to training data", "Perfect model performance", "Need for more features"],
                            "correct_answer": "Overfitting to training data",
                            "explanation": "High variance indicates that the model is too complex and has learned the noise in the training data rather than the underlying pattern. This leads to overfitting, where the model performs well on training data but poorly on new, unseen data.",
                            "practical_context": "Evaluating model performance in real ML projects"
                        }
                    ]
                },
                "intermediate": {
                    "Feature Engineering": [
                        {
                            "question": "You have a dataset with a 'salary' column ranging from $30,000 to $200,000. Before training a neural network, what preprocessing step would be most beneficial?",
                            "options": ["One-hot encoding", "Feature scaling/normalization", "Creating polynomial features", "Removing the column"],
                            "correct_answer": "Feature scaling/normalization",
                            "explanation": "Feature scaling (like MinMaxScaler or StandardScaler) is crucial for neural networks because they use gradient descent optimization. Large value ranges can cause gradient issues and slow convergence. Normalizing ensures all features contribute equally to the learning process.",
                            "practical_context": "Preparing salary prediction models for HR systems"
                        }
                    ],
                    "Model Selection and Evaluation": [
                        {
                            "question": "You're building a fraud detection system where missing a fraudulent transaction is much worse than flagging a legitimate transaction as fraud. Which metric should you prioritize?",
                            "options": ["Accuracy", "Precision", "Recall", "F1-score"],
                            "correct_answer": "Recall",
                            "explanation": "Recall measures how well the model identifies all positive cases (fraudulent transactions). In fraud detection, it's critical to catch all fraud cases even if it means some false alarms. High recall ensures you don't miss fraudulent transactions, which could be very costly.",
                            "practical_context": "Building fraud detection systems for financial institutions"
                        }
                    ]
                }
            },
            "web": {
                "beginner": {
                    "HTML Structure and Semantics": [
                        {
                            "question": "Which HTML5 semantic element should you use to wrap the main navigation menu of a website?",
                            "options": ["<div>", "<menu>", "<nav>", "<header>"],
                            "correct_answer": "<nav>",
                            "explanation": "The <nav> element is specifically designed for major navigation blocks. It provides semantic meaning that helps screen readers, search engines, and other tools understand the structure of your page. While <div> would work visually, <nav> is the semantic choice.",
                            "practical_context": "Building accessible and SEO-friendly website navigation"
                        }
                    ],
                    "JavaScript Fundamentals": [
                        {
                            "question": "What's the difference between '==' and '===' in JavaScript?",
                            "options": ["No difference", "'==' checks type, '===' doesn't", "'===' is stricter and checks both value and type", "'==' is newer syntax"],
                            "correct_answer": "'===' is stricter and checks both value and type",
                            "explanation": "'===' performs strict equality comparison, checking both value and data type without type conversion. '==' performs loose equality, converting types if needed. For example: '5' == 5 is true, but '5' === 5 is false because one is a string and one is a number.",
                            "practical_context": "Preventing bugs in form validation and data comparison"
                        }
                    ]
                },
                "intermediate": {
                    "Frontend Frameworks": [
                        {
                            "question": "In React, when should you use useEffect with an empty dependency array []?",
                            "options": ["To run effect on every render", "To run effect only once after initial render", "To prevent the effect from running", "To run effect when props change"],
                            "correct_answer": "To run effect only once after initial render",
                            "explanation": "An empty dependency array tells React to run the effect only once after the initial render, similar to componentDidMount in class components. This is useful for one-time setup like API calls, event listeners, or timers that should only initialize once.",
                            "practical_context": "Setting up API calls and subscriptions in React components"
                        }
                    ]
                }
            },
            "cyber": {
                "beginner": {
                    "Cybersecurity Fundamentals": [
                        {
                            "question": "Which of the following best describes the 'CIA triad' in cybersecurity?",
                            "options": ["Central Intelligence Agency framework", "Confidentiality, Integrity, Availability", "Criminal Investigation Approach", "Cyber Intelligence Analysis"],
                            "correct_answer": "Confidentiality, Integrity, Availability",
                            "explanation": "The CIA triad is a fundamental security model consisting of Confidentiality (protecting data from unauthorized access), Integrity (ensuring data accuracy and preventing unauthorized modification), and Availability (ensuring systems and data are accessible when needed).",
                            "practical_context": "Designing security policies for organizational data protection"
                        }
                    ]
                }
            }
        }
        
        # Get appropriate question bank
        track_questions = questions_bank.get(track, {})
        difficulty_questions = track_questions.get(difficulty, {})
        topic_questions = difficulty_questions.get(topic_focus, [])
        
        if topic_questions:
            question_data = random.choice(topic_questions)
            return {
                **question_data,
                "topic": topic_focus,
                "difficulty": difficulty,
                "generated": True,
                "ai_generated": True
            }
        
        # Fallback generic question
        return self._generate_fallback_question(track, difficulty, topic_focus)
    
    def _generate_fallback_question(self, track: str, difficulty: str, topic: str) -> Dict:
        """Generate a fallback question when specific content isn't available"""
        
        fallback_questions = {
            "ai": {
                "question": f"In the context of {topic}, what is a key consideration for {difficulty}-level practitioners?",
                "options": [
                    "Understanding the fundamental concepts",
                    "Applying best practices from the field",
                    "Optimizing for real-world scenarios", 
                    "Following industry standards"
                ],
                "correct_answer": "Understanding the fundamental concepts",
                "explanation": f"For {difficulty}-level {track} development, having a solid understanding of fundamental concepts in {topic} is essential for building more advanced skills.",
                "topic": topic,
                "difficulty": difficulty
            }
        }
        
        return fallback_questions.get(track, fallback_questions["ai"])

class AdaptiveAssessmentEngine:
    """Main assessment engine that coordinates learning and questioning"""
    
    def __init__(self, track: str):
        self.track = track
        self.question_generator = DynamicAIQuestionGenerator()
        self.user_ability = 0.5
        self.questions_asked = 0
        self.correct_answers = 0
        self.performance_history = {
            'recent_scores': [],
            'topics_covered': [],
            'topics_mastered': 0,
            'difficulty_progression': []
        }
        self.learning_progress = {}
        
    def get_next_question(self) -> Optional[Dict]:
        """Get the next adaptive question based on current progress"""
        
        # Update learning path based on current ability
        learning_path = self.question_generator.generate_learning_path(self.track, self.user_ability)
        
        # Generate contextual question
        question = self.question_generator.generate_contextual_question(
            self.track, 
            self.user_ability, 
            self.performance_history
        )
        
        if question:
            self.questions_asked += 1
            return question
        
        return None
    
    def submit_answer(self, question: Dict, selected_answer: str) -> Tuple[bool, float]:
        """Submit answer and update user ability"""
        
        is_correct = question['correct_answer'] == selected_answer
        
        if is_correct:
            self.correct_answers += 1
            score = 1.0
        else:
            score = 0.0
        
        # Update performance history
        self.performance_history['recent_scores'].append(score)
        
        # Keep only last 10 scores for adaptive calculation
        if len(self.performance_history['recent_scores']) > 10:
            self.performance_history['recent_scores'] = self.performance_history['recent_scores'][-10:]
        
        # Update user ability using simple adaptive algorithm
        self._update_ability(is_correct, question.get('difficulty', 'intermediate'))
        
        # Track topic coverage
        topic = question.get('topic', 'General')
        if topic not in self.performance_history['topics_covered']:
            self.performance_history['topics_covered'].append(topic)
        
        return is_correct, score
    
    def _update_ability(self, is_correct: bool, difficulty: str):
        """Update user ability based on performance"""
        
        difficulty_values = {'beginner': 0.3, 'intermediate': 0.5, 'advanced': 0.8}
        question_difficulty = difficulty_values.get(difficulty, 0.5)
        
        if is_correct:
            # Increase ability if answered correctly
            adjustment = 0.1 * (question_difficulty - self.user_ability + 0.1)
            self.user_ability = min(1.0, self.user_ability + adjustment)
        else:
            # Decrease ability if answered incorrectly
            adjustment = 0.1 * (self.user_ability - question_difficulty + 0.1)
            self.user_ability = max(0.0, self.user_ability - adjustment)
    
    def get_assessment_summary(self) -> Dict:
        """Get comprehensive assessment summary"""
        
        final_score = self.correct_answers / self.questions_asked if self.questions_asked > 0 else 0
        
        return {
            'final_score': final_score,
            'total_questions': self.questions_asked,
            'correct_answers': self.correct_answers,
            'user_ability': self.user_ability,
            'performance_history': self.performance_history,
            'recommended_level': self._get_recommended_level(),
            'learning_suggestions': self._get_learning_suggestions()
        }
    
    def _get_recommended_level(self) -> str:
        """Get recommended difficulty level for future sessions"""
        if self.user_ability < 0.4:
            return "beginner"
        elif self.user_ability < 0.7:
            return "intermediate"
        else:
            return "advanced"
    
    def _get_learning_suggestions(self) -> List[str]:
        """Generate personalized learning suggestions"""
        
        suggestions = []
        final_score = self.correct_answers / self.questions_asked if self.questions_asked > 0 else 0
        
        if final_score >= 0.8:
            suggestions.append("Excellent performance! Consider advancing to the next difficulty level.")
            suggestions.append(f"Explore advanced {self.track} topics and real-world projects.")
        elif final_score >= 0.6:
            suggestions.append("Good progress! Focus on strengthening your understanding of key concepts.")
            suggestions.append("Practice more problems in areas where you struggled.")
        else:
            suggestions.append("Focus on fundamental concepts before moving to advanced topics.")
            suggestions.append(f"Review basic {self.track} principles and practice regularly.")
        
        # Add track-specific suggestions
        track_suggestions = {
            "ai": [
                "Practice implementing algorithms from scratch",
                "Work on real datasets to gain practical experience",
                "Join Kaggle competitions to test your skills"
            ],
            "web": [
                "Build personal projects to apply your knowledge",
                "Learn about web accessibility and performance optimization",
                "Contribute to open-source projects"
            ],
            "cyber": [
                "Set up a home lab for hands-on practice",
                "Stay updated with latest security threats and trends",
                "Practice with cybersecurity CTF challenges"
            ]
        }
        
        suggestions.extend(track_suggestions.get(self.track, []))
        return suggestions[:4]  # Return top 4 suggestions

# Initialize global components
if 'database' not in st.session_state:
    st.session_state.database = SimpleDatabase()

def authenticate_user():
    """Handle user authentication"""
    st.title("🎓 AI-Powered Adaptive Assessment Platform")
    
    st.markdown("""
    Welcome to our intelligent assessment platform! This system uses AI to:
    - Generate personalized questions based on your learning progress
    - Adapt difficulty in real-time based on your performance
    - Create dynamic learning paths tailored to your needs
    - Provide detailed feedback and learning recommendations
    """)
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.header("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                user = st.session_state.database.authenticate_user(username, password)
                if user:
                    st.session_state.user = user
                    st.success(f"Welcome back, {user.username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        st.header("Register New Account")
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_email = st.text_input("Email Address")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            role = st.selectbox("Role", ["student", "teacher"])
            register_button = st.form_submit_button("Register")
            
            if register_button:
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif st.session_state.database.create_user(new_username, new_password, new_email, role):
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("Username already exists")

def student_dashboard():
    """Student dashboard interface"""
    st.title(f"Welcome, {st.session_state.user.username}! 🎓")
    
    # Show AI capabilities
    with st.expander("🤖 AI-Powered Features", expanded=False):
        st.markdown("""
        **Dynamic Question Generation**: Questions are generated in real-time based on your learning progress
        
        **Adaptive Learning Paths**: The system creates personalized learning journeys for each track
        
        **Intelligent Difficulty Adjustment**: Questions adapt to your skill level automatically
        
        **Contextual Learning**: Each question is designed to build upon your previous knowledge
        """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Take Assessment", "My Progress", "Learning Path", "Practice Questions", "Profile"]
    )
    
    if page == "Take Assessment":
        take_assessment_page()
    elif page == "My Progress":
        progress_page()
    elif page == "Learning Path":
        learning_path_page()
    elif page == "Practice Questions":
        practice_page()
    elif page == "Profile":
        profile_page()

def take_assessment_page():
    """Main assessment taking interface"""
    st.header("📝 AI-Powered Assessment")
    
    st.markdown("""
    Our AI system will generate questions specifically tailored to your current skill level and learning progress.
    The questions will adapt in real-time based on your performance!
    """)
    
    # Track selection
    available_tracks = ["ai", "web", "cyber", "data", "mobile", "devops"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_track = st.selectbox(
            "Select Technology Track",
            available_tracks,
            format_func=lambda x: {
                "web": "🌐 Web Development",
                "ai": "🤖 Artificial Intelligence", 
                "cyber": "🔒 Cybersecurity",
                "data": "📊 Data Science",
                "mobile": "📱 Mobile Development",
                "devops": "⚙️ DevOps"
            }.get(x, x.title())
        )
    
    with col2:
        initial_level = st.selectbox(
            "Your Current Level (Initial Assessment)",
            ["beginner", "intermediate", "advanced"],
            format_func=lambda x: {
                "beginner": "🟢 Beginner - New to the field",
                "intermediate": "🟡 Intermediate - Some experience", 
                "advanced": "🔴 Advanced - Experienced practitioner"
            }.get(x, x.title())
        )
    
    # Assessment configuration
    st.subheader("Assessment Settings")
    col3, col4 = st.columns(2)
    
    with col3:
        max_questions = st.slider("Maximum Questions", 5, 20, 10)
        adaptive_mode = st.checkbox("Enable Adaptive Difficulty", True, 
                                   help="AI will adjust question difficulty based on your performance")
    
    with col4:
        time_limit = st.selectbox("Time Limit (minutes)", [None, 15, 30, 45, 60])
        detailed_feedback = st.checkbox("Show detailed explanations", True)
    
    # Show AI features
    st.info("🤖 **AI Features Enabled**: Dynamic question generation, adaptive difficulty, personalized learning path")
    
    if st.button("🚀 Start AI Assessment", type="primary"):
        start_assessment(selected_track, initial_level, max_questions, adaptive_mode, time_limit, detailed_feedback)

def start_assessment(track, initial_level, max_questions, adaptive_mode, time_limit, detailed_feedback):
    """Initialize and run assessment"""
    
    # Initialize assessment engine
    engine = AdaptiveAssessmentEngine(track)
    
    # Set initial ability based on user selection
    level_mapping = {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8}
    engine.user_ability = level_mapping.get(initial_level, 0.5)
    
    # Create session
    session_id = f"session_{int(time.time())}"
    session = AssessmentSession(
        session_id=session_id,
        username=st.session_state.user.username,
        track=track,
        started_at=datetime.now(),
        performance_data={},
        learning_progress={}
    )
    
    # Store in session state
    st.session_state.assessment_engine = engine
    st.session_state.current_session = session
    st.session_state.max_questions = max_questions
    st.session_state.adaptive_mode = adaptive_mode
    st.session_state.detailed_feedback = detailed_feedback
    st.session_state.assessment_started = True
    st.session_state.current_question = None
    
    if time_limit:
        st.session_state.end_time = datetime.now() + timedelta(minutes=time_limit)
    
    st.rerun()

def run_assessment():
    """Run the active assessment"""
    engine = st.session_state.assessment_engine
    session = st.session_state.current_session
    
    # Check time limit
    if 'end_time' in st.session_state:
        remaining_time = st.session_state.end_time - datetime.now()
        if remaining_time.total_seconds() <= 0:
            complete_assessment()
            return
        
        # Show countdown
        minutes, seconds = divmod(int(remaining_time.total_seconds()), 60)
        st.sidebar.metric("⏰ Time Remaining", f"{minutes:02d}:{seconds:02d}")
    
    # Show AI status
    st.sidebar.markdown("🤖 **AI Status**: Active")
    st.sidebar.metric("Current Ability Level", f"{engine.user_ability:.0%}")
    
    # Show progress
    progress = engine.questions_asked / st.session_state.max_questions
    st.progress(progress)
    st.caption(f"Question {engine.questions_asked + 1} of {st.session_state.max_questions}")
    
    # Get current question
    if st.session_state.current_question is None:
        with st.spinner("🤖 AI is generating your next question..."):
            current_question = engine.get_next_question()
            if current_question is None:
                st.error("Unable to generate more questions. Completing assessment.")
                complete_assessment()
                return
            st.session_state.current_question = current_question
    
    question = st.session_state.current_question
    
    # Display question with AI context
    st.subheader(f"🎯 Question {engine.questions_asked + 1}")
    
    # Show learning context
    with st.expander("🧠 Learning Context", expanded=False):
        st.write(f"**Topic**: {question.get('topic', 'General')}")
        st.write(f"**Difficulty**: {question.get('difficulty', 'intermediate').title()}")
        if 'practical_context' in question:
            st.write(f"**Real-world Application**: {question['practical_context']}")
    
    st.write(question['text'])
    
    # Show options
    with st.form("question_form"):
        selected_answer = st.radio("Choose your answer:", question['options'])
        submit_answer = st.form_submit_button("Submit Answer")
        
        if submit_answer:
            process_answer(question, selected_answer)

def process_answer(question, selected_answer):
    """Process the submitted answer"""
    engine = st.session_state.assessment_engine
    
    # Submit answer and get feedback
    is_correct, score = engine.submit_answer(question, selected_answer)
    
    # Show immediate feedback
    if is_correct:
        st.success("✅ Correct! Well done!")
    else:
        st.error(f"❌ Incorrect. The correct answer was: **{question['correct_answer']}**")
    
    # Show detailed explanation if enabled
    if st.session_state.detailed_feedback and 'explanation' in question:
        st.info(f"💡 **Explanation**: {question['explanation']}")
    
    # Show AI adaptation info
    with st.expander("🤖 AI Adaptation", expanded=False):
        st.write(f"**Your ability level**: {engine.user_ability:.0%}")
        recent_performance = engine.performance_history['recent_scores'][-3:]
        if recent_performance:
            avg_recent = sum(recent_performance) / len(recent_performance)
            st.write(f"**Recent performance**: {avg_recent:.0%}")
        
        next_difficulty = engine._get_recommended_level()
        st.write(f"**Next question difficulty**: {next_difficulty}")
    
    # Clear current question
    st.session_state.current_question = None
    
    # Check if assessment should end
    if engine.questions_asked >= st.session_state.max_questions:
        st.button("📊 View Results", on_click=complete_assessment)
    else:
        st.button("➡️ Next Question", on_click=lambda: st.rerun())

def complete_assessment():
    """Complete the assessment and show results"""
    engine = st.session_state.assessment_engine
    session = st.session_state.current_session
    
    # Finalize session
    session.completed_at = datetime.now()
    summary = engine.get_assessment_summary()
    session.final_score = summary['final_score']
    session.questions_answered = engine.questions_asked
    session.ability_level = engine.user_ability
    session.performance_data = summary
    
    # Save session to database
    st.session_state.database.save_session(session)
    
    # Clear assessment state
    st.session_state.assessment_started = False
    
    # Show results
    show_assessment_results(session)

def show_assessment_results(session: AssessmentSession):
    """Display comprehensive assessment results"""
    st.title("🎉 AI Assessment Complete!")
    
    perf_data = session.performance_data
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Score", f"{perf_data['final_score']:.0%}")
    
    with col2:
        st.metric("Questions Answered", perf_data['total_questions'])
    
    with col3:
        st.metric("Final Ability Level", f"{session.ability_level:.0%}")
    
    with col4:
        st.metric("Recommended Level", perf_data['recommended_level'].title())
    
    # AI-generated insights
    st.subheader("🤖 AI Insights & Recommendations")
    
    suggestions = perf_data.get('learning_suggestions', [])
    for i, suggestion in enumerate(suggestions, 1):
        st.write(f"{i}. {suggestion}")
    
    # Performance progression
    if 'recent_scores' in perf_data['performance_history']:
        scores = perf_data['performance_history']['recent_scores']
        if len(scores) > 2:
            st.subheader("📈 Your Learning Journey")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(scores) + 1)),
                y=scores,
                mode='lines+markers',
                name='Performance',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title="Performance Throughout Assessment",
                xaxis_title="Question Number",
                yaxis_title="Score (1=Correct, 0=Incorrect)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Topic coverage
    topics_covered = perf_data['performance_history'].get('topics_covered', [])
    if topics_covered:
        st.subheader("🎯 Topics Explored")
        for topic in topics_covered:
            st.write(f"• {topic}")
    
    # Next steps
    st.subheader("🚀 What's Next?")
    
    final_score = perf_data['final_score']
    if final_score >= 0.8:
        st.success("🌟 Outstanding performance! You're ready for advanced challenges.")
        st.markdown("**Suggested actions:**")
        st.write("- Explore advanced topics in your chosen track")
        st.write("- Take on real-world projects")
        st.write("- Consider mentoring others or teaching")
    elif final_score >= 0.6:
        st.info("👍 Good progress! Keep building on your foundation.")
        st.markdown("**Suggested actions:**")
        st.write("- Practice more in areas where you struggled")
        st.write("- Review intermediate concepts")
        st.write("- Work on practical projects")
    else:
        st.warning("📚 Focus on strengthening fundamentals first.")
        st.markdown("**Suggested actions:**")
        st.write("- Review basic concepts thoroughly")
        st.write("- Practice with beginner-level exercises")
        st.write("- Consider taking a structured course")
    
    if st.button("🔄 Take Another Assessment"):
        st.rerun()

def learning_path_page():
    """Show AI-generated learning path for the user"""
    st.header("🗺️ Your AI-Generated Learning Path")
    
    st.markdown("""
    Based on your performance and preferences, our AI has created a personalized learning path for you.
    This path adapts as you progress and learn!
    """)
    
    # Track selection for learning path
    track = st.selectbox(
        "Select track to view learning path:",
        ["ai", "web", "cyber", "data", "mobile", "devops"],
        format_func=lambda x: {
            "web": "🌐 Web Development",
            "ai": "🤖 Artificial Intelligence",
            "cyber": "🔒 Cybersecurity", 
            "data": "📊 Data Science",
            "mobile": "📱 Mobile Development",
            "devops": "⚙️ DevOps"
        }.get(x, x.title())
    )
    
    # Get user's current level from their session history
    user_sessions = st.session_state.database.get_user_sessions(st.session_state.user.username)
    track_sessions = [s for s in user_sessions if s.track == track and s.completed_at]
    
    if track_sessions:
        latest_session = max(track_sessions, key=lambda x: x.completed_at)
        user_level = latest_session.ability_level
        st.info(f"📊 Based on your latest {track} assessment, your ability level is: **{user_level:.0%}**")
    else:
        user_level = 0.5  # Default
        st.info(f"🆕 This will be your first {track} assessment. Starting with intermediate level.")
    
    # Generate learning path
    generator = DynamicAIQuestionGenerator()
    
    with st.spinner("🤖 AI is generating your personalized learning path..."):
        learning_path = generator.generate_learning_path(track, user_level)
    
    if learning_path and 'topics' in learning_path:
        st.subheader(f"📚 Your {track.upper()} Learning Journey")
        
        for i, topic in enumerate(learning_path['topics'], 1):
            with st.expander(f"📖 {i}. {topic['name']}", expanded=(i == 1)):
                st.write(f"**Description**: {topic['description']}")
                
                st.write("**Key Concepts to Master:**")
                for concept in topic['concepts']:
                    st.write(f"• {concept}")
                
                # Show progress if user has covered this topic
                if i <= len(track_sessions) + 1:
                    if i <= len(track_sessions):
                        st.success("✅ Completed")
                    else:
                        st.info("📍 Current Focus Area")
                else:
                    st.write("🔒 Unlocks after previous topics")
        
        # Show learning statistics
        st.subheader("📊 Learning Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            completed_topics = len(track_sessions)
            total_topics = len(learning_path['topics'])
            st.metric("Topics Completed", f"{completed_topics}/{total_topics}")
        
        with col2:
            if track_sessions:
                avg_score = sum(s.final_score for s in track_sessions) / len(track_sessions)
                st.metric("Average Score", f"{avg_score:.0%}")
            else:
                st.metric("Average Score", "N/A")
        
        with col3:
            progress_percentage = (completed_topics / total_topics) * 100 if total_topics > 0 else 0
            st.metric("Overall Progress", f"{progress_percentage:.0f}%")
        
        # Progress visualization
        if track_sessions:
            st.subheader("📈 Your Progress Over Time")
            
            session_data = [{
                'Date': s.completed_at.strftime('%Y-%m-%d'),
                'Score': s.final_score,
                'Ability Level': s.ability_level
            } for s in sorted(track_sessions, key=lambda x: x.completed_at)]
            
            df = pd.DataFrame(session_data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Date'], 
                y=df['Score'],
                mode='lines+markers',
                name='Assessment Score',
                line=dict(color='#2E86AB')
            ))
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Ability Level'], 
                mode='lines+markers',
                name='Ability Level',
                line=dict(color='#A23B72')
            ))
            fig.update_layout(
                title=f"Your {track.upper()} Progress Journey",
                xaxis_title="Date",
                yaxis_title="Performance", 
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Unable to generate learning path. Please try again.")

def progress_page():
    """Show comprehensive user progress"""
    st.header("📈 My Learning Progress")
    
    sessions = st.session_state.database.get_user_sessions(st.session_state.user.username)
    
    if not sessions:
        st.info("🆕 No assessments taken yet. Start your AI learning journey!")
        if st.button("🚀 Take Your First Assessment"):
            st.session_state.current_page = "Take Assessment"
            st.rerun()
        return
    
    # Overall statistics
    completed_sessions = [s for s in sessions if s.completed_at]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Assessments", len(sessions))
    
    with col2:
        if completed_sessions:
            avg_score = np.mean([s.final_score for s in completed_sessions])
            st.metric("Average Score", f"{avg_score:.0%}")
        else:
            st.metric("Average Score", "N/A")
    
    with col3:
        total_questions = sum(s.questions_answered for s in sessions)
        st.metric("Questions Answered", total_questions)
    
    with col4:
        unique_tracks = len(set(s.track for s in sessions))
        st.metric("Tracks Explored", unique_tracks)
    
    # Track-wise performance
    if completed_sessions:
        st.subheader("🎯 Performance by Track")
        
        track_data = {}
        for session in completed_sessions:
            track = session.track
            if track not in track_data:
                track_data[track] = []
            track_data[track].append(session.final_score)
        
        track_summary = []
        for track, scores in track_data.items():
            track_summary.append({
                'Track': track.upper(),
                'Assessments': len(scores),
                'Average Score': f"{np.mean(scores):.0%}",
                'Best Score': f"{max(scores):.0%}",
                'Latest Score': f"{scores[-1]:.0%}"
            })
        
        df = pd.DataFrame(track_summary)
        st.dataframe(df, use_container_width=True)
        
        # Performance trends
        st.subheader("📊 Score Trends")
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (track, scores) in enumerate(track_data.items()):
            track_sessions = [s for s in completed_sessions if s.track == track]
            track_sessions.sort(key=lambda x: x.completed_at)
            
            dates = [s.completed_at.strftime('%Y-%m-%d') for s in track_sessions]
            scores = [s.final_score for s in track_sessions]
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=scores,
                mode='lines+markers',
                name=track.upper(),
                line=dict(color=colors[i % len(colors)]),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Assessment Score Progression",
            xaxis_title="Date",
            yaxis_title="Score",
            height=400,
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity
        st.subheader("🕒 Recent Activity")
        
        recent_sessions = sorted(sessions, key=lambda x: x.started_at, reverse=True)[:5]
        
        for session in recent_sessions:
            status = "✅ Completed" if session.completed_at else "⏳ In Progress"
            score_text = f"Score: {session.final_score:.0%}" if session.completed_at else "Ongoing"
            
            with st.container():
                st.markdown(f"""
                **{session.track.upper()} Assessment** - {status}
                - Started: {session.started_at.strftime('%Y-%m-%d %H:%M')}
                - {score_text}
                - Questions: {session.questions_answered}
                """)

def practice_page():
    """Enhanced practice page with AI question generation"""
    st.header("🏋️ AI-Powered Practice")
    
    st.markdown("""
    Practice with AI-generated questions that adapt to your skill level.
    Each question is created specifically for your learning needs!
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        practice_track = st.selectbox(
            "Select Track for Practice",
            ["ai", "web", "cyber", "data", "mobile", "devops"],
            format_func=lambda x: {
                "web": "🌐 Web Development",
                "ai": "🤖 Artificial Intelligence",
                "cyber": "🔒 Cybersecurity",
                "data": "📊 Data Science", 
                "mobile": "📱 Mobile Development",
                "devops": "⚙️ DevOps"
            }.get(x, x.title())
        )
    
    with col2:
        practice_level = st.selectbox(
            "Target Difficulty Level",
            ["beginner", "intermediate", "advanced"],
            format_func=lambda x: {
                "beginner": "🟢 Beginner",
                "intermediate": "🟡 Intermediate", 
                "advanced": "🔴 Advanced"
            }.get(x, x.title())
        )
    
    # Practice mode selection
    practice_mode = st.selectbox(
        "Practice Mode",
        ["Single Question", "Quick Practice (5 questions)", "Extended Practice (10 questions)", "Topic Focus"]
    )
    
    # Topic focus option
    if practice_mode == "Topic Focus":
        generator = DynamicAIQuestionGenerator()
        learning_path = generator.generate_learning_path(practice_track, 0.5)
        
        if learning_path and 'topics' in learning_path:
            topic_names = [topic['name'] for topic in learning_path['topics']]
            selected_topic = st.selectbox("Focus on specific topic:", topic_names)
        else:
            selected_topic = None
    else:
        selected_topic = None
    
    # AI generation settings
    st.subheader("🤖 AI Generation Settings")
    col3, col4 = st.columns(2)
    
    with col3:
        use_contextual = st.checkbox("Use contextual learning", True, 
                                   help="AI will consider your learning history")
        show_hints = st.checkbox("Show hints for wrong answers", True)
    
    with col4:
        explain_difficulty = st.checkbox("Explain difficulty adaptation", True)
        track_progress = st.checkbox("Track practice progress", True)
    
    if st.button("🎯 Start AI Practice Session", type="primary"):
        start_practice_session(practice_track, practice_level, practice_mode, 
                             selected_topic, use_contextual, show_hints, 
                             explain_difficulty, track_progress)

def start_practice_session(track, level, mode, topic_focus, use_contextual, 
                          show_hints, explain_difficulty, track_progress):
    """Start an enhanced AI practice session"""
    
    question_counts = {
        "Single Question": 1,
        "Quick Practice (5 questions)": 5,
        "Extended Practice (10 questions)": 10,
        "Topic Focus": 8
    }
    
    num_questions = question_counts[mode]
    
    # Initialize practice session with AI engine
    practice_engine = AdaptiveAssessmentEngine(track)
    
    # Set level based on user selection
    level_mapping = {"beginner": 0.3, "intermediate": 0.5, "advanced": 0.8}
    practice_engine.user_ability = level_mapping.get(level, 0.5)
    
    # Store practice session data
    st.session_state.practice_active = True
    st.session_state.practice_engine = practice_engine
    st.session_state.practice_track = track
    st.session_state.practice_level = level
    st.session_state.practice_topic_focus = topic_focus
    st.session_state.practice_settings = {
        'use_contextual': use_contextual,
        'show_hints': show_hints,
        'explain_difficulty': explain_difficulty,
        'track_progress': track_progress
    }
    st.session_state.practice_questions = []
    st.session_state.practice_answers = []
    st.session_state.practice_current_q = 0
    st.session_state.practice_total = num_questions
    st.session_state.practice_start_time = datetime.now()
    
    st.rerun()

def run_practice_session():
    """Run the enhanced AI practice session"""
    
    if st.session_state.practice_current_q >= st.session_state.practice_total:
        show_practice_results()
        return
    
    engine = st.session_state.practice_engine
    settings = st.session_state.practice_settings
    
    # Progress indicator
    progress = st.session_state.practice_current_q / st.session_state.practice_total
    st.progress(progress)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Question {st.session_state.practice_current_q + 1} of {st.session_state.practice_total}")
    with col2:
        if settings['track_progress']:
            correct_so_far = sum(1 for a in st.session_state.practice_answers if a['correct'])
            st.caption(f"Correct: {correct_so_far}/{len(st.session_state.practice_answers)}")
    
    # AI status in sidebar
    st.sidebar.markdown("🤖 **AI Practice Mode**")
    st.sidebar.metric("Current Ability", f"{engine.user_ability:.0%}")
    
    if settings['explain_difficulty']:
        current_difficulty = engine._get_recommended_level()
        st.sidebar.write(f"**Next difficulty**: {current_difficulty}")
    
    # Generate or get question
    if len(st.session_state.practice_questions) <= st.session_state.practice_current_q:
        
        with st.spinner("🤖 AI is crafting your next question..."):
            # Use topic focus if specified
            if st.session_state.practice_topic_focus:
                # Override the generator to focus on specific topic
                engine.question_generator.performance_history = engine.performance_history
                question = engine.question_generator.generate_contextual_question(
                    st.session_state.practice_track,
                    engine.user_ability,
                    engine.performance_history,
                    st.session_state.practice_topic_focus
                )
            else:
                question = engine.get_next_question()
            
            if question:
                st.session_state.practice_questions.append(question)
            else:
                st.error("Unable to generate more questions")
                return
    
    current_question = st.session_state.practice_questions[st.session_state.practice_current_q]
    
    # Display question with enhanced context
    st.subheader(f"🎯 Practice Question {st.session_state.practice_current_q + 1}")
    
    # Show AI generation context
    with st.expander("🧠 Question Context", expanded=False):
        st.write(f"**Topic**: {current_question.get('topic', 'General')}")
        st.write(f"**Difficulty**: {current_question.get('difficulty', 'intermediate').title()}")
        st.write(f"**Generated by**: AI based on your current ability level")
        
        if 'practical_context' in current_question:
            st.write(f"**Real-world scenario**: {current_question['practical_context']}")
        
        if settings['use_contextual']:
            st.write("**Contextual Learning**: ✅ Enabled")
    
    st.write(current_question['text'])
    
    # Answer form
    with st.form(f"practice_form_{st.session_state.practice_current_q}"):
        selected_answer = st.radio("Select your answer:", current_question['options'])
        
        col1, col2 = st.columns([1, 3])
        with col1:
            submit_button = st.form_submit_button("Submit Answer")
        with col2:
            if settings['show_hints'] and st.form_submit_button("💡 Need a hint?"):
                show_practice_hint(current_question)
        
        if submit_button:
            process_practice_answer(current_question, selected_answer)

def show_practice_hint(question):
    """Show AI-generated hint for practice question"""
    
    # Generate contextual hint based on question
    hints = {
        "beginner": "Think about the basic concepts and fundamental principles.",
        "intermediate": "Consider the practical applications and best practices.",
        "advanced": "Focus on optimization, architecture, and complex scenarios."
    }
    
    difficulty = question.get('difficulty', 'intermediate')
    basic_hint = hints.get(difficulty, "Consider the key concepts involved.")
    
    # More specific hints based on track
    track_hints = {
        "ai": {
            "beginner": "Think about basic ML concepts like supervised vs unsupervised learning.",
            "intermediate": "Consider data preprocessing, model evaluation, or algorithm selection.",
            "advanced": "Focus on neural networks, optimization, or production deployment."
        },
        "web": {
            "beginner": "Think about HTML structure, CSS styling, or basic JavaScript.",
            "intermediate": "Consider frameworks, APIs, or responsive design principles.",
            "advanced": "Focus on performance, security, or advanced architectural patterns."
        },
        "cyber": {
            "beginner": "Think about basic security concepts like confidentiality, integrity, availability.",
            "intermediate": "Consider specific tools, protocols, or incident response procedures.",
            "advanced": "Focus on advanced threats, architecture design, or compliance requirements."
        }
    }
    
    topic = question.get('topic', '')
    track = st.session_state.practice_track
    
    specific_hint = track_hints.get(track, {}).get(difficulty, basic_hint)
    
    st.info(f"💡 **Hint**: {specific_hint}")
    
    # Additional hint based on topic keywords
    topic_lower = topic.lower()
    if 'python' in topic_lower or 'programming' in topic_lower:
        st.info("💭 **Extra hint**: Think about Python syntax, data types, or programming logic.")
    elif 'data' in topic_lower:
        st.info("💭 **Extra hint**: Consider data structures, processing methods, or analysis techniques.")
    elif 'security' in topic_lower or 'cyber' in topic_lower:
        st.info("💭 **Extra hint**: Think about threats, protection methods, or security tools.")

def process_practice_answer(question, selected_answer):
    """Process practice answer with enhanced feedback"""
    
    engine = st.session_state.practice_engine
    settings = st.session_state.practice_settings
    
    # Submit to engine for ability adjustment
    is_correct, score = engine.submit_answer(question, selected_answer)
    
    # Record detailed answer data
    answer_data = {
        'question': question,
        'selected': selected_answer,
        'correct': is_correct,
        'timestamp': datetime.now(),
        'ability_before': engine.user_ability,
        'difficulty': question.get('difficulty', 'intermediate')
    }
    
    st.session_state.practice_answers.append(answer_data)
    
    # Show enhanced feedback
    if is_correct:
        st.success("✅ Excellent! You got it right!")
        
        if settings['explain_difficulty'] and engine.user_ability > answer_data['ability_before']:
            st.info("🎯 **AI Adaptation**: Your ability level increased! Next question may be more challenging.")
    else:
        st.error(f"❌ Not quite right. The correct answer was: **{question['correct_answer']}**")
        
        if settings['explain_difficulty'] and engine.user_ability < answer_data['ability_before']:
            st.info("🎯 **AI Adaptation**: Question was challenging. Next question will be adjusted to your level.")
        
        if settings['show_hints']:
            st.warning("💡 **Learning Tip**: Review the explanation below and try to understand why this answer is correct.")
    
    # Show detailed explanation
    if 'explanation' in question:
        st.info(f"📚 **Explanation**: {question['explanation']}")
    
    # Show practical context
    if 'practical_context' in question:
        st.info(f"🌍 **Real-world Application**: {question['practical_context']}")
    
    # Move to next question
    st.session_state.practice_current_q += 1
    
    # Show progress update
    if settings['track_progress']:
        correct_count = sum(1 for a in st.session_state.practice_answers if a['correct'])
        total_answered = len(st.session_state.practice_answers)
        current_score = correct_count / total_answered if total_answered > 0 else 0
        
        with st.expander("📊 Progress Update", expanded=False):
            st.write(f"**Current Score**: {current_score:.0%} ({correct_count}/{total_answered})")
            st.write(f"**Ability Level**: {engine.user_ability:.0%}")
            
            if total_answered >= 3:
                recent_performance = [a['correct'] for a in st.session_state.practice_answers[-3:]]
                recent_score = sum(recent_performance) / len(recent_performance)
                st.write(f"**Recent Performance**: {recent_score:.0%} (last 3 questions)")
    
    # Continue or finish
    if st.session_state.practice_current_q < st.session_state.practice_total:
        st.button("➡️ Next Question", on_click=lambda: st.rerun())
    else:
        st.button("📊 View Practice Results", on_click=lambda: st.rerun())

def show_practice_results():
    """Show comprehensive practice session results"""
    
    st.title("🎉 Practice Session Complete!")
    
    engine = st.session_state.practice_engine
    answers = st.session_state.practice_answers
    settings = st.session_state.practice_settings
    
    # Calculate metrics
    correct_count = sum(1 for a in answers if a['correct'])
    total_count = len(answers)
    final_score = correct_count / total_count if total_count > 0 else 0
    
    practice_duration = datetime.now() - st.session_state.practice_start_time
    avg_time_per_question = practice_duration.total_seconds() / total_count if total_count > 0 else 0
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Score", f"{final_score:.0%}")
    with col2:
        st.metric("Questions Answered", f"{correct_count}/{total_count}")
    with col3:
        st.metric("Ability Growth", f"{engine.user_ability:.0%}")
    with col4:
        st.metric("Avg Time/Question", f"{avg_time_per_question:.0f}s")
    
    # AI Performance Analysis
    st.subheader("🤖 AI Performance Analysis")
    
    ability_start = answers[0]['ability_before'] if answers else 0.5
    ability_end = engine.user_ability
    ability_change = ability_end - ability_start
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Ability Level Progression:**")
        st.write(f"• Starting level: {ability_start:.0%}")
        st.write(f"• Final level: {ability_end:.0%}")
        
        if ability_change > 0.05:
            st.success(f"📈 Improved by {ability_change:.0%}!")
        elif ability_change < -0.05:
            st.info(f"📊 Adjusted down by {abs(ability_change):.0%} (normal adaptation)")
        else:
            st.info("📊 Stable performance - good consistency!")
    
    with col2:
        difficulty_dist = {}
        for answer in answers:
            diff = answer['difficulty']
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
        
        st.write("**Questions by Difficulty:**")
        for diff, count in difficulty_dist.items():
            st.write(f"• {diff.title()}: {count}")
    
    # Performance visualization
    if len(answers) > 2:
        st.subheader("📈 Learning Journey Visualization")
        
        question_numbers = list(range(1, len(answers) + 1))
        performance = [1 if a['correct'] else 0 for a in answers]
        ability_progression = [a['ability_before'] for a in answers] + [engine.user_ability]
        
        fig = go.Figure()
        
        # Performance line
        fig.add_trace(go.Scatter(
            x=question_numbers,
            y=performance,
            mode='lines+markers',
            name='Question Performance',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=10)
        ))
        
        # Ability progression
        fig.add_trace(go.Scatter(
            x=list(range(1, len(ability_progression) + 1)),
            y=ability_progression,
            mode='lines+markers',
            name='Ability Level',
            line=dict(color='#A23B72', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Practice Session Analysis",
            xaxis_title="Question Number",
            yaxis_title="Correct (1) / Incorrect (0)",
            yaxis2=dict(
                title="Ability Level",
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed question review
    st.subheader("📝 Question Review")
    
    for i, answer in enumerate(answers, 1):
        status = "✅" if answer['correct'] else "❌"
        with st.expander(f"Question {i} {status} - {answer['question']['topic']}", expanded=False):
            st.write(f"**Question**: {answer['question']['text']}")
            st.write(f"**Your Answer**: {answer['selected']}")
            st.write(f"**Correct Answer**: {answer['question']['correct_answer']}")
            st.write(f"**Difficulty**: {answer['difficulty'].title()}")
            
            if 'explanation' in answer['question']:
                st.write(f"**Explanation**: {answer['question']['explanation']}")
    
    # Personalized recommendations
    st.subheader("💡 AI-Generated Recommendations")
    
    if final_score >= 0.8:
        st.success("🌟 Excellent practice session! You're mastering this topic.")
        recommendations = [
            f"You're ready for more advanced {st.session_state.practice_track} challenges",
            "Consider taking a full assessment to measure your progress",
            "Try practicing with 'advanced' difficulty level",
            "Explore specialized topics within your track"
        ]
    elif final_score >= 0.6:
        st.info("👍 Good practice session! Keep building your knowledge.")
        recommendations = [
            "Focus on areas where you had incorrect answers",
            "Practice similar topics with mixed difficulty levels",
            "Review explanations for questions you missed",
            "Try extended practice sessions for more experience"
        ]
    else:
        st.warning("📚 Keep practicing! Focus on fundamentals.")
        recommendations = [
            "Review basic concepts in your chosen track",
            "Practice with 'beginner' level questions first",
            "Take time to understand explanations thoroughly",
            "Consider shorter, frequent practice sessions"
        ]
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Practice Same Topic"):
            st.rerun()
    
    with col2:
        if st.button("📝 Take Full Assessment"):
            st.session_state.practice_active = False
            st.session_state.current_page = "Take Assessment"
            st.rerun()
    
    with col3:
        if st.button("🏠 Back to Dashboard"):
            st.session_state.practice_active = False
            st.rerun()

def profile_page():
    """Enhanced user profile page with AI insights"""
    st.header("👤 My Profile & AI Insights")
    
    user = st.session_state.user
    sessions = st.session_state.database.get_user_sessions(user.username)
    
    # Basic profile info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Profile Information")
        st.text_input("Username", user.username, disabled=True)
        st.text_input("Email", user.email, disabled=True)
        st.text_input("Role", user.role.title(), disabled=True)
        st.text_input("Member Since", user.created_at.strftime('%Y-%m-%d'), disabled=True)
    
    with col2:
        st.subheader("🎯 Learning Statistics")
        if sessions:
            completed = [s for s in sessions if s.completed_at]
            st.metric("Total Assessments", len(sessions))
            st.metric("Completed Assessments", len(completed))
            if completed:
                avg_score = sum(s.final_score for s in completed) / len(completed)
                st.metric("Average Score", f"{avg_score:.0%}")
                
                latest_ability = max(completed, key=lambda x: x.completed_at).ability_level
                st.metric("Current Ability Level", f"{latest_ability:.0%}")
        else:
            st.info("No assessment data yet")
    
    # AI-generated learning profile
    if sessions:
        st.subheader("🤖 AI Learning Profile")
        
        # Generate insights based on user performance
        completed_sessions = [s for s in sessions if s.completed_at]
        
        if completed_sessions:
            # Track strengths and areas for improvement
            track_performance = {}
            for session in completed_sessions:
                track = session.track
                if track not in track_performance:
                    track_performance[track] = []
                track_performance[track].append(session.final_score)
            
            # Find strongest and weakest tracks
            track_averages = {track: sum(scores)/len(scores) 
                            for track, scores in track_performance.items()}
            
            if track_averages:
                strongest_track = max(track_averages, key=track_averages.get)
                weakest_track = min(track_averages, key=track_averages.get)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"💪 **Strongest Area**: {strongest_track.upper()}")
                    st.write(f"Average score: {track_averages[strongest_track]:.0%}")
                    
                with col2:
                    st.info(f"📚 **Growth Opportunity**: {weakest_track.upper()}")
                    st.write(f"Average score: {track_averages[weakest_track]:.0%}")
            
            # Learning pattern analysis
            st.subheader("📊 Learning Pattern Analysis")
            
            # Performance trend
            recent_sessions = sorted(completed_sessions, key=lambda x: x.completed_at)[-5:]
            if len(recent_sessions) >= 3:
                recent_scores = [s.final_score for s in recent_sessions]
                trend_slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
                
                if trend_slope > 0.1:
                    st.success("📈 **Trend**: Improving steadily!")
                elif trend_slope < -0.1:
                    st.warning("📉 **Trend**: Consider reviewing fundamentals")
                else:
                    st.info("📊 **Trend**: Consistent performance")
            
            # AI recommendations
            st.subheader("🎯 Personalized AI Recommendations")
            
            total_assessments = len(completed_sessions)
            avg_score = sum(s.final_score for s in completed_sessions) / len(completed_sessions)
            
            recommendations = []
            
            if avg_score >= 0.8:
                recommendations.extend([
                    "🌟 You're performing excellently! Consider mentoring others",
                    "🚀 Try advanced-level assessments to challenge yourself",
                    "💼 Look into real-world projects in your strongest areas"
                ])
            elif avg_score >= 0.6:
                recommendations.extend([
                    "👍 Good progress! Focus on consistency",
                    "📖 Review topics where you scored below 60%",
                    "🎯 Take regular practice sessions to maintain skills"
                ])
            else:
                recommendations.extend([
                    "📚 Focus on building strong fundamentals",
                    "🔄 Use the practice mode frequently",
                    "⏰ Consider shorter, more frequent study sessions"
                ])
            
            # Track-specific recommendations
            if track_performance:
                most_practiced = max(track_performance, key=lambda x: len(track_performance[x]))
                if len(track_performance[most_practiced]) >= 3:
                    recommendations.append(f"🎯 You're most active in {most_practiced.upper()} - consider specializing further")
                
                if len(track_performance) == 1:
                    recommendations.append("🌍 Try exploring other technology tracks to broaden your knowledge")
            
            for rec in recommendations[:5]:
                st.write(rec)
    
    # Learning preferences
    st.subheader("🎯 Learning Preferences")
    
    with st.form("preferences_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            preferred_tracks = st.multiselect(
                "Preferred Technology Tracks",
                ["ai", "web", "cyber", "data", "mobile", "devops"],
                default=user.profile_data.get('preferred_tracks', [])
            )
            
            difficulty_preference = st.selectbox(
                "Preferred Starting Difficulty",
                ["beginner", "intermediate", "advanced"],
                index=["beginner", "intermediate", "advanced"].index(
                    user.profile_data.get('difficulty_preference', 'intermediate')
                )
            )
        
        with col2:
            ai_features = st.multiselect(
                "Preferred AI Features",
                ["Adaptive Difficulty", "Contextual Questions", "Detailed Explanations", "Learning Path Generation"],
                default=user.profile_data.get('ai_features', ["Adaptive Difficulty", "Detailed Explanations"])
            )
            
            practice_frequency = st.selectbox(
                "Preferred Practice Frequency",
                ["Daily", "Few times a week", "Weekly", "Monthly"],
                index=["Daily", "Few times a week", "Weekly", "Monthly"].index(
                    user.profile_data.get('practice_frequency', 'Few times a week')
                )
            )
        
        notification_preferences = st.multiselect(
            "Notification Preferences",
            ["Practice Reminders", "New Features", "Progress Updates", "Learning Tips"],
            default=user.profile_data.get('notifications', [])
        )
        
        if st.form_submit_button("💾 Save Preferences"):
            if not user.profile_data:
                user.profile_data = {}
            
            user.profile_data.update({
                'preferred_tracks': preferred_tracks,
                'difficulty_preference': difficulty_preference,
                'ai_features': ai_features,
                'practice_frequency': practice_frequency,
                'notifications': notification_preferences
            })
            
            st.success("✅ Preferences saved! The AI will use these to personalize your experience.")

def teacher_dashboard():
    """Enhanced teacher dashboard"""
    st.title(f"Teacher Dashboard - {st.session_state.user.username} 👨‍🏫")
    
    st.sidebar.title("Teacher Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Student Overview", "AI Analytics", "Question Management", "Learning Insights"]
    )
    
    if page == "Student Overview":
        teacher_student_overview()
    elif page == "AI Analytics":
        teacher_ai_analytics()
    elif page == "Question Management":
        teacher_question_management()
    elif page == "Learning Insights":
        teacher_learning_insights()

def teacher_student_overview():
    """Enhanced student overview with AI insights"""
    st.header("📊 Student Overview & AI Insights")
    
    all_users = st.session_state.database.get_all_users()
    students = [user for user in all_users if user.role == 'student']
    
    if not students:
        st.info("No students registered yet.")
        return
    
    # Student performance analytics
    student_analytics = []
    
    for student in students:
        sessions = st.session_state.database.get_user_sessions(student.username)
        completed_sessions = [s for s in sessions if s.completed_at]
        
        if completed_sessions:
            avg_score = np.mean([s.final_score for s in completed_sessions])
            ability_growth = max([s.ability_level for s in completed_sessions]) - min([s.ability_level for s in completed_sessions])
            tracks_explored = len(set(s.track for s in sessions))
            
            # Calculate learning velocity (improvement over time)
            if len(completed_sessions) >= 2:
                sorted_sessions = sorted(completed_sessions, key=lambda x: x.completed_at)
                first_score = sorted_sessions[0].final_score
                last_score = sorted_sessions[-1].final_score
                learning_velocity = last_score - first_score
            else:
                learning_velocity = 0
        else:
            avg_score = 0
            ability_growth = 0
            tracks_explored = 0
            learning_velocity = 0
        
        student_analytics.append({
            'Student': student.username,
            'Email': student.email,
            'Total Assessments': len(sessions),
            'Completed': len(completed_sessions),
            'Avg Score': f"{avg_score:.0%}",
            'Ability Growth': f"{ability_growth:.0%}",
            'Tracks': tracks_explored,
            'Learning Velocity': f"{learning_velocity:+.0%}",
            'Last Active': max([s.started_at for s in sessions], default=student.created_at).strftime('%Y-%m-%d'),
            'Status': '🟢 Active' if sessions and sessions[-1].started_at > datetime.now() - timedelta(days=7) else '🟡 Inactive'
        })
    
    # Display analytics table
    df = pd.DataFrame(student_analytics)
    st.dataframe(df, use_container_width=True)
    
    # Class performance insights
    st.subheader("🎯 Class Performance Insights")
    
    completed_sessions_all = []
    for student in students:
        sessions = st.session_state.database.get_user_sessions(student.username)
        completed_sessions_all.extend([s for s in sessions if s.completed_at])
    
    if completed_sessions_all:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            class_avg = np.mean([s.final_score for s in completed_sessions_all])
            st.metric("Class Average Score", f"{class_avg:.0%}")
        
        with col2:
            active_students = sum(1 for s in student_analytics if s['Status'] == '🟢 Active')
            st.metric("Active Students", f"{active_students}/{len(students)}")
        
        with col3:
            total_questions = sum(s.questions_answered for s in completed_sessions_all)
            st.metric("Total Questions Answered", total_questions)
        
        # Performance distribution
        scores = [s.final_score for s in completed_sessions_all]
        fig = px.histogram(x=scores, nbins=10, title="Class Score Distribution")
        fig.update_xaxis(title="Score")
        fig.update_yaxis(title="Number of Assessments")
        st.plotly_chart(fig, use_container_width=True)
