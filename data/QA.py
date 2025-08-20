"""
Enhanced questions.py with intelligent MCQ generator using language models
Focuses on generating questions dynamically rather than using pre-stored ones
"""

import json
import random
import streamlit as st
from typing import List, Dict, Optional, Tuple
import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re

class IntelligentMCQGenerator:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize the intelligent MCQ generator with a language model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Initialize the tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            
            # Set padding token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"✅ Model {model_name} loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            self.tokenizer = None
        
        # Define available tracks and their descriptions
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
            
        if not self.model or not self.tokenizer:
            return self._create_fallback_question(track, difficulty)
        
        try:
            # Create prompt for the model
            prompt = self._create_prompt(track, difficulty)
            
            # Generate question and options
            question_text, options = self._generate_with_model(prompt)
            
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

    def _create_prompt(self, track: str, difficulty: str) -> str:
        """Create a prompt for the language model"""
        track_description = self.available_tracks[track]
        
        prompts = {
            "easy": f"Create an easy multiple choice question about {track_description}. " +
                   "The question should be for beginners. Provide the correct answer as the first option " +
                   "and three plausible but incorrect options. Format: Question: [question] Options: A) [option1] B) [option2] C) [option3] D) [option4]",
                   
            "medium": f"Create a medium difficulty multiple choice question about {track_description}. " +
                     "The question should be for intermediate learners. Provide the correct answer as the first option " +
                     "and three plausible but incorrect options. Format: Question: [question] Options: A) [option1] B) [option2] C) [option3] D) [option4]",
                     
            "hard": f"Create a challenging multiple choice question about {track_description}. " +
                   "The question should be for advanced learners. Provide the correct answer as the first option " +
                   "and three plausible but incorrect options. Format: Question: [question] Options: A) [option1] B) [option2] C) [option3] D) [option4]"
        }
        
        return prompts.get(difficulty, prompts["medium"])

    def _generate_with_model(self, prompt: str, max_length: int = 150) -> Tuple[str, List[str]]:
        """Generate text using the language model"""
        try:
            # Tokenize the input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.9,
                    top_k=50
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the generated text
            generated_text = generated_text.replace(prompt, "").strip()
            
            # Parse the question and options
            return self._parse_question_and_options(generated_text)
            
        except Exception as e:
            print(f"Error in model generation: {e}")
            return None, []

    def _parse_question_and_options(self, text: str) -> Tuple[str, List[str]]:
        """Parse the generated text to extract question and options"""
        # Try to find the question part
        question_match = re.search(r'Question:\s*(.*?)(?=Options:|$)', text, re.IGNORECASE | re.DOTALL)
        if question_match:
            question_text = question_match.group(1).strip()
        else:
            # If no question marker, take the first sentence as question
            sentences = re.split(r'[.!?]', text)
            question_text = sentences[0].strip() if sentences else "What is a key concept in this field?"
        
        # Try to find options
        options = []
        option_patterns = [
            r'A\)\s*(.*?)(?=B\)|C\)|D\)|$)',
            r'B\)\s*(.*?)(?=C\)|D\)|$)',
            r'C\)\s*(.*?)(?=D\)|$)',
            r'D\)\s*(.*?)(?=$)'
        ]
        
        for pattern in option_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                options.append(match.group(1).strip())
        
        # If we couldn't find formatted options, try to extract any list-like items
        if len(options) < 4:
            # Look for numbered or lettered items
            alt_options = re.findall(r'(?:[A-D]\.|\d+\.)\s*(.*?)(?=\n|$)', text)
            if len(alt_options) >= 4:
                options = alt_options[:4]
        
        # If we still don't have enough options, create some fallbacks
        if len(options) < 4:
            options = [
                "Correct answer",
                "Incorrect alternative 1", 
                "Incorrect alternative 2",
                "Incorrect alternative 3"
            ]
        
        return question_text, options

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
        """Create a fallback question if model generation fails"""
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
            # Small delay to avoid overwhelming the system
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
        with st.spinner("Loading AI model..."):
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