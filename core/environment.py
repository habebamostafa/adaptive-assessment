# environment.py
import numpy as np
import random

class AdaptiveAssessmentEnv:
    def __init__(self, track):
        self.track = track
        self.levels = [1, 2, 3]  # المستويات المتاحة
        self.current_level = 2  # البداية من المستوى المتوسط
        self.student_ability = 0.5  # قدرة الطالب المبدئية
        self.question_history = []
        self.consecutive_correct = 0
        self.consecutive_incorrect = 0
        self.total_questions = 0
        self.max_questions = 10
        
    def reset(self):
        """إعادة تعيين البيئة"""
        self.current_level = 2
        self.student_ability = 0.5
        self.question_history = []
        self.consecutive_correct = 0
        self.consecutive_incorrect = 0
        self.total_questions = 0
        return self.get_state()
    
    def get_state(self):
        """الحصول على حالة البيئة الحالية"""
        return {
            'current_level': self.current_level,
            'student_ability': self.student_ability,
            'consecutive_correct': self.consecutive_correct,
            'consecutive_incorrect': self.consecutive_incorrect,
            'progress': self.total_questions / self.max_questions
        }
    
    def submit_answer(self, question, answer, is_correct):
        """Process answer and return reward"""
        reward = self._calculate_reward(is_correct, question['level'])
        
        # Update student ability
        self._update_ability(is_correct)
        
        # Track history
        self.question_history.append({
            'question': question,
            'answer': answer,
            'is_correct': is_correct,
            'level': question['level'],
            'student_ability_after': self.student_ability
        })
        
        done = len(self.question_history) >= self.max_questions
        return reward, done
    
    def _calculate_environment_reward(self, is_correct, level_factor):
        """حساب مكافأة البيئة بناءً على مناسبة السؤال للطالب"""
        # مكافأة أساسية للإجابة الصحيحة
        base_reward = 1.0 if is_correct else -1.0
        
        # مكافأة إضافية إذا كان السؤال مناسباً لمستوى الطالب
        level_appropriateness = 1 - abs(level_factor - self.student_ability)
        appropriateness_bonus = level_appropriateness * 0.5
        
        return base_reward + appropriateness_bonus
    
    def get_performance_stats(self):
        """الحصول على إحصائيات الأداء"""
        if not self.question_history:
            return {
                'accuracy': 0.0,
                'level_distribution': {level: 0 for level in self.levels},
                'improvement': 0.0
            }
        
        correct_answers = sum(1 for q in self.question_history if q['is_correct'])
        accuracy = correct_answers / len(self.question_history)
        
        level_distribution = {level: 0 for level in self.levels}
        for q in self.question_history:
            level_distribution[q['level']] += 1
        
        # حساب التحسن في الأداء
        if len(self.question_history) >= 2:
            first_half = self.question_history[:len(self.question_history)//2]
            second_half = self.question_history[len(self.question_history)//2:]
            
            first_accuracy = sum(1 for q in first_half if q['is_correct']) / len(first_half)
            second_accuracy = sum(1 for q in second_half if q['is_correct']) / len(second_half)
            improvement = second_accuracy - first_accuracy
        else:
            improvement = 0.0
        
        return {
            'accuracy': accuracy,
            'level_distribution': level_distribution,
            'improvement': improvement,
            'final_ability': self.student_ability
        }
    
    def _calculate_reward(self, is_correct, level):
        """Calculate dynamic reward based on answer and difficulty level"""
        base_reward = {
            1: 0.5,  # Easy
            2: 1.0,   # Medium
            3: 1.5    # Hard
        }
        
        if is_correct:
            reward = base_reward[level] * (1 + self.consecutive_correct * 0.2)
        else:
            penalty = -0.5 * (1 + self.consecutive_incorrect * 0.1)
            reward = max(penalty, -1.5)  # Cap penalty
            
        return reward