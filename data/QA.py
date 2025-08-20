"""
Enhanced MCQ Generator using Google's FLAN-T5 with Hugging Face Token and CrewAI Agents
"""

import streamlit as st
import requests
from typing import List, Dict, Optional
import json
import random
import time
from huggingface_hub import InferenceClient
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain.llms import HuggingFaceHub
from langchain.schema import BaseLanguageModel

# Custom tool for question generation
class QuestionGenerationTool(BaseTool):
    name: str = "question_generator"
    description: str = "Generates technical interview questions using FLAN-T5"
    
    def _run(self, topic: str, difficulty: str) -> str:
        """Generate a question using the tool"""
        # This would connect to your FLAN-T5 model
        return f"Generated question about {topic} at {difficulty} level"

class EnhancedFlanT5MCQGenerator:
    def __init__(self):
        """Initialize the MCQ generator with Hugging Face token and CrewAI agents"""
        self.available_tracks = {
            "web": "Web Development (HTML, CSS, JavaScript, React, Vue, Angular, etc.)",
            "ai": "Artificial Intelligence (Machine Learning, Deep Learning, NLP, Computer Vision, etc.)",
            "cyber": "Cybersecurity (Network Security, Encryption, Ethical Hacking, Penetration Testing, etc.)",
            "data": "Data Science (Data Analysis, Visualization, Statistics, Big Data, etc.)",
            "mobile": "Mobile Development (Android, iOS, Flutter, React Native, Kotlin, Swift, etc.)",
            "devops": "DevOps (Docker, Kubernetes, CI/CD, Cloud Computing, AWS, Azure, etc.)",
            "backend": "Backend Development (APIs, Databases, Microservices, System Design, etc.)",
            "frontend": "Frontend Development (UI/UX, Responsive Design, Performance, Accessibility, etc.)"
        }
        
        # Initialize Hugging Face client
        self.hf_token = None
        self.client = None
        self._setup_hugging_face()
        
        # Initialize CrewAI agents
        self.agents = self._create_crewai_agents()
        
    def _setup_hugging_face(self):
        """Setup Hugging Face connection"""
        try:
            # Get token from Streamlit secrets
            if "hf_token" in st.secrets:
                self.hf_token = st.secrets["hf_token"]
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.hf_token
                
                # Initialize the client
                self.client = InferenceClient(
                    model="google/flan-t5-large",
                    token=self.hf_token
                )
                st.success("✅ Hugging Face connection established successfully!")
            else:
                st.warning("⚠️ Hugging Face token not found in secrets. Using demo mode.")
        except Exception as e:
            st.error(f"❌ Error setting up Hugging Face: {e}")
            
    def _create_crewai_agents(self) -> Dict[str, Agent]:
        """Create specialized CrewAI agents for different tasks"""
        
        # Question Designer Agent
        question_designer = Agent(
            role='Technical Question Designer',
            goal='Design comprehensive and challenging technical interview questions',
            backstory="""You are an expert technical interviewer with 10+ years of experience 
            in evaluating candidates across different technology domains. You excel at creating 
            questions that test both theoretical knowledge and practical application.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Content Reviewer Agent
        content_reviewer = Agent(
            role='Content Quality Reviewer',
            goal='Review and improve the quality of generated questions',
            backstory="""You are a meticulous content reviewer specializing in technical education. 
            Your expertise lies in ensuring questions are accurate, well-structured, and 
            appropriately challenging for the target difficulty level.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Difficulty Assessor Agent
        difficulty_assessor = Agent(
            role='Difficulty Level Assessor',
            goal='Assess and calibrate the difficulty level of technical questions',
            backstory="""You are a learning assessment specialist who understands how to 
            properly calibrate question difficulty based on skill levels from beginner 
            to expert in various technology domains.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Answer Validator Agent
        answer_validator = Agent(
            role='Answer Validation Specialist',
            goal='Validate correct answers and create comprehensive explanations',
            backstory="""You are a technical expert with deep knowledge across multiple 
            technology domains. You ensure that correct answers are accurate and 
            explanations are clear and educational.""",
            verbose=True,
            allow_delegation=False
        )
        
        return {
            'designer': question_designer,
            'reviewer': content_reviewer,
            'assessor': difficulty_assessor,
            'validator': answer_validator
        }
    
    def get_available_tracks(self) -> List[str]:
        """Get list of available technology tracks"""
        return list(self.available_tracks.keys())
    
    def generate_question_with_crew(self, track: str, difficulty: str = "medium") -> Dict:
        """
        Generate a question using CrewAI agents and FLAN-T5
        
        Args:
            track: Technology track
            difficulty: Difficulty level (easy, medium, hard)
        
        Returns:
            Question dictionary
        """
        try:
            # Create tasks for the crew
            tasks = self._create_crew_tasks(track, difficulty)
            
            # Create and run the crew
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            # Execute the crew workflow
            with st.spinner("🤖 CrewAI agents are collaborating to generate your question..."):
                result = crew.kickoff()
            
            # Parse and return the result
            return self._parse_crew_result(result, track, difficulty)
            
        except Exception as e:
            st.error(f"Error with CrewAI generation: {e}")
            return self._generate_with_flan_t5_fallback(track, difficulty)
    
    def _create_crew_tasks(self, track: str, difficulty: str) -> List[Task]:
        """Create tasks for the CrewAI agents"""
        
        # Task 1: Design the question
        design_task = Task(
            description=f"""
            Design a {difficulty} level multiple choice question for {self.available_tracks[track]}.
            The question should be:
            1. Technically accurate and relevant to job interviews
            2. Have exactly 4 options (A, B, C, D)
            3. Test practical knowledge, not just memorization
            4. Be appropriate for the {difficulty} difficulty level
            
            Include the question text, four distinct options, and identify the correct answer.
            """,
            agent=self.agents['designer'],
            expected_output="A well-structured multiple choice question with 4 options"
        )
        
        # Task 2: Review and improve
        review_task = Task(
            description=f"""
            Review the generated question for:
            1. Technical accuracy
            2. Clarity of language
            3. Appropriateness of difficulty level
            4. Quality of distractors (incorrect options)
            
            Make improvements if necessary and ensure the question meets high standards.
            """,
            agent=self.agents['reviewer'],
            expected_output="Reviewed and improved question with quality feedback"
        )
        
        # Task 3: Validate answers and create explanation
        validate_task = Task(
            description=f"""
            Validate the correct answer and create a comprehensive explanation that:
            1. Confirms why the correct answer is right
            2. Explains why other options are incorrect
            3. Provides additional context or learning points
            4. Uses clear, educational language
            
            Format the final output as JSON with: question, options, correct_answer, explanation
            """,
            agent=self.agents['validator'],
            expected_output="Final validated question with comprehensive explanation in JSON format"
        )
        
        return [design_task, review_task, validate_task]
    
    def _parse_crew_result(self, result: str, track: str, difficulty: str) -> Dict:
        """Parse the result from CrewAI agents"""
        try:
            # Try to extract JSON from the result
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                data = json.loads(json_str)
                
                # Convert to our expected format
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
                    'text': data.get("question", f"What is a key concept in {self.available_tracks[track]}?"),
                    'options': options_list,
                    'correct_answer': correct_answer_text,
                    'explanation': data.get("explanation", f"This is a {difficulty} level question about {self.available_tracks[track]}."),
                    'track': track,
                    'difficulty': difficulty,
                    'generated_by': 'CrewAI + FLAN-T5',
                    'crew_generated': True
                }
            else:
                raise ValueError("No valid JSON found in crew result")
                
        except Exception as e:
            st.warning(f"Error parsing crew result: {e}. Using fallback generation.")
            return self._generate_with_flan_t5_fallback(track, difficulty)
    
    def _generate_with_flan_t5_fallback(self, track: str, difficulty: str) -> Dict:
        """Fallback to direct FLAN-T5 generation"""
        try:
            prompt = self._create_flan_t5_prompt(track, difficulty)
            
            if self.client and self.hf_token:
                response = self.client.text_generation(
                    inputs=prompt,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True
                )
                
                # Parse response
                return self._parse_flan_t5_response(response, track, difficulty)
            else:
                return self._create_demo_question(track, difficulty)
                
        except Exception as e:
            st.error(f"Error with FLAN-T5 fallback: {e}")
            return self._create_demo_question(track, difficulty)
    
    def _create_flan_t5_prompt(self, track: str, difficulty: str) -> str:
        """Create a detailed prompt for FLAN-T5 model"""
        track_description = self.available_tracks[track]
        
        prompt = f"""
        Create a {difficulty} difficulty multiple choice question about {track_description}.
        
        Requirements:
        - Technical and interview-appropriate
        - Test practical knowledge
        - Have exactly 4 options (A, B, C, D)
        - Include brief explanation
        
        Format as JSON:
        {{
            "question": "question text",
            "options": {{
                "A": "option A",
                "B": "option B", 
                "C": "option C",
                "D": "option D"
            }},
            "correct_answer": "A",
            "explanation": "explanation text"
        }}
        
        Topic: {track_description}
        Difficulty: {difficulty}
        """
        
        return prompt
    
    def _parse_flan_t5_response(self, response: str, track: str, difficulty: str) -> Dict:
        """Parse FLAN-T5 response"""
        try:
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
                    'text': data.get("question", f"What is a key concept in {self.available_tracks[track]}?"),
                    'options': options_list,
                    'correct_answer': correct_answer_text,
                    'explanation': data.get("explanation", f"This is a {difficulty} level question about {self.available_tracks[track]}."),
                    'track': track,
                    'difficulty': difficulty,
                    'generated_by': 'FLAN-T5',
                    'direct_flan_t5': True
                }
            else:
                raise ValueError("No valid JSON in response")
                
        except Exception as e:
            return self._create_demo_question(track, difficulty)
    
    def _create_demo_question(self, track: str, difficulty: str) -> Dict:
        """Create demo questions for different tracks and difficulties"""
        demo_questions = {
            "web": {
                "easy": {
                    "question": "What does HTML stand for in web development?",
                    "options": ["HyperText Markup Language", "High Tech Modern Language", "Home Tool Markup Language", "Hyperlink Text Management Language"],
                    "correct_answer": "HyperText Markup Language",
                    "explanation": "HTML stands for HyperText Markup Language, which is the standard markup language for creating web pages."
                },
                "medium": {
                    "question": "What is the main benefit of using React's Virtual DOM?",
                    "options": ["Faster DOM manipulation through efficient diffing", "Direct database connectivity", "Automatic CSS styling", "Built-in security features"],
                    "correct_answer": "Faster DOM manipulation through efficient diffing",
                    "explanation": "React's Virtual DOM improves performance by creating a virtual representation of the UI and efficiently updating only the parts that have changed."
                },
                "hard": {
                    "question": "How does webpack's tree shaking optimization work?",
                    "options": ["Eliminates dead code by analyzing ES6 module imports", "Compresses images automatically", "Minifies CSS files only", "Removes unused HTML elements"],
                    "correct_answer": "Eliminates dead code by analyzing ES6 module imports",
                    "explanation": "Tree shaking is a dead-code elimination technique that removes unused exports from ES6 modules during the build process."
                }
            },
            "ai": {
                "easy": {
                    "question": "What is the primary purpose of machine learning?",
                    "options": ["Enable computers to learn from data without explicit programming", "Create human-like robots", "Replace all human workers", "Process images only"],
                    "correct_answer": "Enable computers to learn from data without explicit programming",
                    "explanation": "Machine learning allows computers to automatically improve their performance on a task through experience and data."
                },
                "medium": {
                    "question": "What distinguishes supervised learning from unsupervised learning?",
                    "options": ["Supervised uses labeled data, unsupervised finds patterns in unlabeled data", "Supervised is faster than unsupervised", "Unsupervised requires human oversight", "Supervised only works with images"],
                    "correct_answer": "Supervised uses labeled data, unsupervised finds patterns in unlabeled data",
                    "explanation": "Supervised learning trains on labeled examples to make predictions, while unsupervised learning discovers hidden patterns in data without labels."
                },
                "hard": {
                    "question": "What is the key innovation of the Transformer architecture in deep learning?",
                    "options": ["Self-attention mechanism for parallel processing of sequences", "Use of convolutional layers for text", "Recurrent connections for memory", "Reinforcement learning integration"],
                    "correct_answer": "Self-attention mechanism for parallel processing of sequences",
                    "explanation": "Transformers introduced self-attention, allowing models to process all positions in a sequence simultaneously and focus on relevant parts."
                }
            }
        }
        
        # Get demo question or create fallback
        if track in demo_questions and difficulty in demo_questions[track]:
            demo = demo_questions[track][difficulty]
            return {
                'text': demo['question'],
                'options': demo['options'],
                'correct_answer': demo['correct_answer'],
                'explanation': demo['explanation'],
                'track': track,
                'difficulty': difficulty,
                'generated_by': 'Demo Mode',
                'demo_mode': True
            }
        else:
            return {
                'text': f"What is an important concept in {self.available_tracks[track]}?",
                'options': [
                    "Correct answer option",
                    "Incorrect option 1",
                    "Incorrect option 2",
                    "Incorrect option 3"
                ],
                'correct_answer': "Correct answer option",
                'explanation': f"This is a {difficulty} level question about {self.available_tracks[track]}.",
                'track': track,
                'difficulty': difficulty,
                'generated_by': 'Fallback',
                'fallback': True
            }
    
    def generate_question_set_with_crew(self, track: str, num_questions: int = 5, difficulty: str = "medium", use_crew: bool = True) -> List[Dict]:
        """
        Generate a set of questions using CrewAI agents
        
        Args:
            track: Technology track
            num_questions: Number of questions to generate
            difficulty: Difficulty level
            use_crew: Whether to use CrewAI agents
        
        Returns:
            List of question dictionaries
        """
        questions = []
        progress_bar = st.progress(0)
        
        for i in range(num_questions):
            if use_crew:
                question = self.generate_question_with_crew(track, difficulty)
            else:
                question = self._generate_with_flan_t5_fallback(track, difficulty)
            
            questions.append(question)
            progress_bar.progress((i + 1) / num_questions)
            time.sleep(0.5)  # Small delay between generations
        
        return questions

# Streamlit interface
def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(
        page_title="Enhanced MCQ Generator",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Enhanced MCQ Generator")
    st.write("Generate technical interview questions using Google's FLAN-T5 and CrewAI agents")
    
    # Initialize session state
    if 'generator' not in st.session_state:
        st.session_state.generator = EnhancedFlanT5MCQGenerator()
    
    generator = st.session_state.generator
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model status
        st.subheader("🔗 Connection Status")
        if generator.hf_token:
            st.success("✅ Hugging Face Connected")
        else:
            st.warning("⚠️ Demo Mode Active")
            st.info("Add your Hugging Face token to secrets.toml for full functionality")
        
        # Generation method
        st.subheader("🎯 Generation Method")
        use_crewai = st.checkbox("Use CrewAI Agents", value=True, 
                                help="Use collaborative AI agents for better question quality")
        
        if use_crewai:
            st.success("🤖 CrewAI agents will collaborate")
        else:
            st.info("🔧 Direct FLAN-T5 generation")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Track selection
        track = st.selectbox(
            "🎯 Select Technology Track:",
            options=generator.get_available_tracks(),
            format_func=lambda x: f"{x.upper()} - {generator.available_tracks[x]}"
        )
        
        # Difficulty and quantity
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            difficulty = st.radio(
                "📊 Difficulty Level:",
                options=["easy", "medium", "hard"],
                horizontal=True,
                help="Easy: Basic concepts, Medium: Practical application, Hard: Advanced topics"
            )
        
        with col1_2:
            num_questions = st.slider(
                "🔢 Number of Questions:",
                min_value=1, max_value=10, value=3,
                help="Number of questions to generate"
            )
    
    with col2:
        st.subheader("📋 Quick Info")
        st.info(f"**Track**: {track.upper()}")
        st.info(f"**Difficulty**: {difficulty.capitalize()}")
        st.info(f"**Questions**: {num_questions}")
        if use_crewai:
            st.success("**Mode**: CrewAI Enhanced")
        else:
            st.warning("**Mode**: Direct Generation")
    
    # Generate button
    st.divider()
    if st.button("🚀 Generate Questions", use_container_width=True, type="primary"):
        with st.spinner(f"🔄 Generating {num_questions} {difficulty} questions for {track}..."):
            questions = generator.generate_question_set_with_crew(
                track=track,
                num_questions=num_questions,
                difficulty=difficulty,
                use_crew=use_crewai
            )
            
            # Store in session state
            st.session_state.questions = questions
            
            # Display success message
            st.success(f"✅ Successfully generated {len(questions)} questions!")
    
    # Display questions
    if 'questions' in st.session_state and st.session_state.questions:
        st.divider()
        st.header("📝 Generated Questions")
        
        for i, q in enumerate(st.session_state.questions, 1):
            with st.container():
                st.subheader(f"Question {i}")
                
                # Question metadata
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                with col_meta1:
                    st.badge(f"Track: {q['track'].upper()}")
                with col_meta2:
                    st.badge(f"Difficulty: {q['difficulty'].upper()}")
                with col_meta3:
                    st.badge(f"Generated by: {q['generated_by']}")
                
                # Question text
                st.write("**Question:**")
                st.write(q['text'])
                
                # Options
                st.write("**Options:**")
                for j, option in enumerate(q['options']):
                    if option.strip():  # Only show non-empty options
                        st.write(f"{chr(65+j)}) {option}")
                
                # Answer and explanation in expander
                with st.expander("💡 Show Answer and Explanation"):
                    st.success(f"**Correct Answer:** {q['correct_answer']}")
                    st.info(f"**Explanation:** {q['explanation']}")
                
                st.divider()
        
        # Export option
        if st.button("📥 Export Questions as JSON"):
            questions_json = json.dumps(st.session_state.questions, indent=2)
            st.download_button(
                label="⬇️ Download JSON",
                data=questions_json,
                file_name=f"mcq_{track}_{difficulty}_{len(st.session_state.questions)}q.json",
                mime="application/json"
            )

if __name__ == "__main__":
    # Initialize session state
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    
    main()