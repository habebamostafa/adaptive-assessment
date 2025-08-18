# environment.py
import numpy as np
import random
from data.questions import QUESTIONS
import numpy as np
import random

class AdaptiveAssessmentEnv:
    def __init__(self, track):
        self.track = track
        self.levels = [1, 2, 3]  # تعريف المستويات
        self.current_level = 2  # يبدأ بالمستوى المتوسط
        self.student_ability = 0.5
        self.question_history = []
        self.consecutive_correct = 0  # إضافة هذا المتغير
        self.consecutive_incorrect = 0  # وإضافة هذا المتغير
    
    def submit_answer(self, question, is_correct):
        # تحديث العدادات المتتالية
        if is_correct:
            self.consecutive_correct += 1
            self.consecutive_incorrect = 0
        else:
            self.consecutive_incorrect += 1
            self.consecutive_correct = 0

        # تحديث القدرة
        level_factor = self.current_level / max(self.levels)
        reward = 0.1 * level_factor if is_correct else -0.1 * (1 - level_factor)
        self.student_ability = max(0.1, min(0.9, self.student_ability + reward))
        
        # تسجيل السؤال
        self.question_history.append({
            "question": question,
            "is_correct": is_correct,
            "level": self.current_level
        })
        
        return is_correct, len(self.question_history) >= 10  # انتهاء بعد 10 أسئلة