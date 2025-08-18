# environment.py
import numpy as np
import random
from data.questions import QUESTIONS
class AdaptiveAssessmentEnv:
    def __init__(self, questions, track):
        """
        Initialize the assessment environment
        """
        self.track = track
        self.questions = questions[track]
        self.levels = sorted(self.questions.keys())
        self.current_level = self.levels[len(self.levels)//2]  # يبدأ من المستوى المتوسط
        self.student_ability = 0.5
        self.question_history = []
        self.consecutive_correct = 0
        self.consecutive_incorrect = 0
        self.asked_questions = set()   # هنا هنسجل الأسئلة اللي اتسألت

    def get_question(self, level):
        """Get a question avoiding duplicates"""
        available = [
            q for q in self.questions[level]
            if q['text'] not in self.asked_questions
        ]
        
        if not available:
            return None
        
        question = random.choice(available)
        self.asked_questions.add(question['text'])
        return question
    
    def submit_answer(self, question, answer):
        is_correct = question['correct_answer'] == answer
        reward = 1 if is_correct else -1

        if is_correct:
            self.consecutive_correct += 1
            self.consecutive_incorrect = 0
        else:
            self.consecutive_incorrect += 1
            self.consecutive_correct = 0

        # تحديث القدرة
        level_factor = self.current_level / max(self.levels)
        if is_correct:
            self.student_ability += 0.1 * (1 + level_factor)
        else:
            self.student_ability -= 0.1 * (1 + (1 - level_factor))

        # حصرها بين 0 و 1
        self.student_ability = max(0, min(1, self.student_ability))

        # حفظ في التاريخ
        self.question_history.append({
            'question': question,
            'answer': answer,
            'is_correct': is_correct,
            'level': self.current_level
        })
        
        is_done = len(self.question_history) >= 3
        return reward, is_done