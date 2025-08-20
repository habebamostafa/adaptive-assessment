import streamlit as st
import json
import time
import random
import numpy as np
from typing import List, Dict, Tuple

class AdaptiveAssessmentEnv:
    def __init__(self, questions: Dict, track: str):
        self.questions = questions
        self.track = track
        self.current_level = "medium"  # Start with medium difficulty
        self.student_ability = 0.5  # Initial ability estimate (0 to 1)
        self.question_history = []
        self.correct_count = 0
        self.total_questions = 0
        
    def get_question(self, difficulty: str) -> Dict:
        """Get a random question of specified difficulty"""
        if difficulty in self.questions.get(self.track, {}):
            available_questions = self.questions[self.track][difficulty]
            if available_questions:
                return random.choice(available_questions)
        return None
    
    def submit_answer(self, question: Dict, answer: str) -> Tuple[float, bool]:
        """Submit answer and return reward and done status"""
        is_correct = answer == question['correct_answer']
        self.question_history.append({
            'question': question['text'],
            'level': self.current_level,
            'is_correct': is_correct,
            'correct_answer': question['correct_answer'],
            'student_answer': answer
        })
        
        # Update student ability estimate
        self.total_questions += 1
        if is_correct:
            self.correct_count += 1
        self.student_ability = self.correct_count / self.total_questions
        
        # Calculate reward
        reward = 1.0 if is_correct else -0.5
        
        # Check if assessment should end
        done = self.total_questions >= 10  # End after 10 questions
        
        return reward, done
    
    def adjust_difficulty(self, action: str):
        """Adjust difficulty based on agent's action"""
        difficulty_levels = ["easy", "medium", "hard"]
        current_index = difficulty_levels.index(self.current_level)
        
        if action == "increase" and current_index < len(difficulty_levels) - 1:
            self.current_level = difficulty_levels[current_index + 1]
        elif action == "decrease" and current_index > 0:
            self.current_level = difficulty_levels[current_index - 1]
        # If "maintain", keep current level

class RLAssessmentAgent:
    def __init__(self, env: AdaptiveAssessmentEnv):
        self.env = env
        self.q_table = {}  # Simple Q-table: (state, action) -> value
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3
        
    def choose_action(self, state: str) -> str:
        """Choose action using epsilon-greedy strategy"""
        if random.random() < self.exploration_rate:
            return random.choice(["increase", "decrease", "maintain"])
        
        # Get Q-values for current state
        actions = ["increase", "decrease", "maintain"]
        q_values = [self.q_table.get((state, action), 0) for action in actions]
        
        # Choose action with highest Q-value
        return actions[np.argmax(q_values)]
    
    def update_q_table(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-table using Q-learning"""
        current_q = self.q_table.get((state, action), 0)
        
        # Get max Q-value for next state
        next_actions = ["increase", "decrease", "maintain"]
        next_q_values = [self.q_table.get((next_state, a), 0) for a in next_actions]
        max_next_q = max(next_q_values) if next_q_values else 0
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

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
        
    def _generate_fallback_question(self, track: str, difficulty: str) -> Dict:
        """Generate a fallback question"""
        track_name = self.available_tracks[track].split('(')[0].strip()
        
        difficulty_modifiers = {
            "easy": ["basic", "fundamental", "essential"],
            "medium": ["important", "key", "practical"],
            "hard": ["advanced", "complex", "sophisticated"]
        }
        
        modifiers = difficulty_modifiers[difficulty]
        modifier = random.choice(modifiers)
        
        topics = {
            "web": ["responsive design", "API integration", "state management"],
            "ai": ["neural networks", "model training", "feature engineering"],
            "cyber": ["encryption methods", "access control", "threat detection"],
            "data": ["data cleaning", "statistical analysis", "visualization techniques"],
            "mobile": ["UI adaptation", "performance optimization", "native functionality"],
            "devops": ["continuous integration", "containerization", "infrastructure as code"],
            "backend": ["database design", "API development", "server optimization"],
            "frontend": ["user interface design", "accessibility standards", "performance optimization"]
        }
        
        topic = random.choice(topics.get(track, topics["web"]))
        
        question_types = [
            f"What is the {modifier} concept of {topic} in {track_name}?",
            f"Which of these best describes {modifier} {topic} in {track_name}?",
            f"What is the primary purpose of {modifier} {topic} in {track_name}?"
        ]
        
        question = random.choice(question_types)
        
        options = [
            f"The {modifier} approach to {topic}",
            f"A common misconception about {topic}",
            f"A related but different technology to {topic}",
            f"A basic implementation detail of {topic}"
        ]
        
        correct_option = options[0]
        random.shuffle(options)
        
        return {
            'text': question,
            'options': options,
            'correct_answer': correct_option,
            'explanation': f"This question tests understanding of {modifier} {topic} in {track_name}.",
            'track': track,
            'difficulty': difficulty
        }
    
    def generate_questions(self, track: str, difficulty: str, count: int) -> List[Dict]:
        """Generate multiple questions"""
        questions = []
        for _ in range(count):
            questions.append(self._generate_fallback_question(track, difficulty))
        return questions

def run_adaptive_assessment():
    st.set_page_config(
        page_title="Adaptive Assessment System",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Adaptive Assessment System")
    st.markdown("Intelligent adaptive testing with AI-powered question generation")
    
    # Initialize session state
    if 'assessment_started' not in st.session_state:
        st.session_state.assessment_started = False
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'env' not in st.session_state:
        st.session_state.env = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'question_generator' not in st.session_state:
        st.session_state.question_generator = OnlineMCQGenerator()
    
    # Configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📚 Assessment Configuration")
        
        track = st.selectbox(
            "Technology Track:",
            options=list(st.session_state.question_generator.available_tracks.keys()),
            format_func=lambda x: f"{x.upper()} - {st.session_state.question_generator.available_tracks[x].split('(')[0].strip()}"
        )
    
    with col2:
        st.subheader("📊 Assessment Info")
        st.info("This adaptive assessment will:")
        st.info("• Adjust difficulty based on your performance")
        st.info("• Generate personalized questions")
        st.info("• Provide detailed results analysis")
    
    # Start assessment button
    if not st.session_state.assessment_started:
        if st.button("🚀 Start Adaptive Assessment", type="primary", use_container_width=True):
            # Generate initial questions
            questions_data = {
                track: {
                    "easy": st.session_state.question_generator.generate_questions(track, "easy", 20),
                    "medium": st.session_state.question_generator.generate_questions(track, "medium", 20),
                    "hard": st.session_state.question_generator.generate_questions(track, "hard", 20)
                }
            }
            
            # Initialize environment and agent
            st.session_state.env = AdaptiveAssessmentEnv(questions_data, track)
            st.session_state.agent = RLAssessmentAgent(st.session_state.env)
            st.session_state.assessment_started = True
            st.session_state.current_question = st.session_state.env.get_question(st.session_state.env.current_level)
            st.rerun()
    
    # Assessment in progress
    if st.session_state.assessment_started and st.session_state.env:
        env = st.session_state.env
        agent = st.session_state.agent
        
        # Display progress
        progress_col1, progress_col2, progress_col3 = st.columns(3)
        with progress_col1:
            st.metric("Questions", f"{env.total_questions}/10")
        with progress_col2:
            st.metric("Current Level", env.current_level.upper())
        with progress_col3:
            st.metric("Ability Score", f"{env.student_ability:.2f}")
        
        # Display current question
        if st.session_state.current_question:
            question = st.session_state.current_question
            st.subheader(f"Question {env.total_questions + 1} ({env.current_level.upper()} Level)")
            st.write(f"**{question['text']}**")
            
            # Answer options
            selected_answer = st.radio(
                "Select your answer:",
                question['options'],
                key=f"question_{env.total_questions}"
            )
            
            # Submit button
            if st.button("Submit Answer", type="primary"):
                # Process answer
                reward, done = env.submit_answer(question, selected_answer)
                
                # Update agent
                next_state = env.current_level
                agent.update_q_table(env.current_level, agent.choose_action(env.current_level), reward, next_state)
                
                # Adjust difficulty
                action = agent.choose_action(env.current_level)
                agent.adjust_difficulty(action)
                
                # Get next question or end assessment
                if done:
                    st.session_state.assessment_started = False
                else:
                    st.session_state.current_question = env.get_question(env.current_level)
                
                st.rerun()
        
        # Show question history if available
        if env.question_history:
            with st.expander("📋 View Question History"):
                for i, q in enumerate(env.question_history, 1):
                    status = "✅" if q['is_correct'] else "❌"
                    st.write(f"{i}. {status} Level {q['level']}: {q['question'][:50]}...")
    
    # Assessment completed
    if not st.session_state.assessment_started and st.session_state.env and st.session_state.env.question_history:
        env = st.session_state.env
        
        st.success("🎉 Assessment Complete!")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", env.total_questions)
        with col2:
            correct = sum(1 for q in env.question_history if q['is_correct'])
            st.metric("Correct Answers", f"{correct}/{env.total_questions}")
        with col3:
            st.metric("Final Ability Score", f"{env.student_ability:.2f}")
        
        # Detailed results
        st.subheader("📊 Detailed Results")
        for i, q in enumerate(env.question_history, 1):
            with st.expander(f"Question {i} ({q['level']} level)"):
                st.write(f"**Question:** {q['question']}")
                st.write(f"**Your answer:** {q['student_answer']}")
                st.write(f"**Correct answer:** {q['correct_answer']}")
                st.write(f"**Result:** {'✅ Correct' if q['is_correct'] else '❌ Incorrect'}")
        
        # Restart button
        if st.button("🔄 Start New Assessment", use_container_width=True):
            for key in ['assessment_started', 'current_question', 'env', 'agent']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    run_adaptive_assessment()