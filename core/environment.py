# environment.py
import numpy as np
import random
from data.questions import QUESTIONS
class AdaptiveAssessmentEnv:
    def __init__(self, track):
        self.track = track
        self.current_level = 2  # يبدأ بالمستوى المتوسط
        self.student_ability = 0.5
        self.question_history = []
    
    def submit_answer(self, question, is_correct):
        # تحديث القدرة
        reward = 0.1 if is_correct else -0.1
        self.student_ability = max(0.1, min(0.9, self.student_ability + reward))
        
        # تسجيل السؤال
        self.question_history.append({
            "question": question,
            "is_correct": is_correct,
            "level": self.current_level
        })
        
        # تحديد إذا انتهى الاختبار (بعد 10 أسئلة)
        return is_correct, len(self.question_history) >= 10