import random
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st

class RLAssessmentAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = {level: {'up': 0, 'stay': 0, 'down': 0} for level in env.levels}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3
        self.llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-large",
            task="text2text-generation",
            huggingfacehub_api_token=st.secrets["huggingfacehub_token"],
            temperature=0.7,
            do_sample=True,
            max_new_tokens=200
        
        )

    def choose_action(self, state=None):
        """اختيار الإجراء بناءً على السياسة epsilon-greedy مع تحسينات"""
        current_level = self.env.current_level
        
        # تقليل الاستكشاف مع تحسن الأداء
        dynamic_exploration = self.exploration_rate * (1 - self.env.student_ability)
        
        # إضافة منطق ذكي للقرار
        if self.env.consecutive_correct >= 2:
            # إذا كان الطالب يجيب بشكل صحيح متتالي، ارفع المستوى
            preferred_action = 'up'
        elif self.env.consecutive_incorrect >= 2:
            # إذا كان الطالب يخطئ متتالياً، اخفض المستوى
            preferred_action = 'down'
        else:
            # ابق في نفس المستوى
            preferred_action = 'stay'
        
        # استكشاف عشوائي أم استغلال المعرفة؟
        if random.random() < dynamic_exploration:
            # استكشاف عشوائي
            return random.choice(['up', 'stay', 'down'])
        else:
            # اختيار أفضل إجراء من Q-table أو الإجراء المفضل
            q_values = self.q_table[current_level]
            best_action = max(q_values, key=q_values.get)
            
            # إذا كانت القيم متقاربة، استخدم المنطق المباشر
            if max(q_values.values()) - min(q_values.values()) < 0.1:
                return preferred_action
            else:
                return best_action
        
    def update_q_table(self, state, action, reward, next_state):
        """تحديث Q-table باستخدام معادلة Bellman"""
        if next_state in self.q_table:
            best_next_value = max(self.q_table[next_state].values())
        else:
            best_next_value = 0
            
        current_q = self.q_table[state][action]
        
        # معادلة Q-learning
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * best_next_value - current_q
        )
        
        self.q_table[state][action] = new_q
        
    def calculate_reward(self, action, student_performance):
        """حساب المكافأة بناءً على مناسبة الإجراء للطالب"""
        if student_performance > 0.7:  # طالب متفوق
            if action == 'up': return 1.0
            elif action == 'stay': return 0.5
            else: return -0.5
        elif student_performance < 0.3:  # طالب يحتاج مساعدة
            if action == 'down': return 1.0
            elif action == 'stay': return 0.5
            else: return -0.5
        else:  # طالب متوسط
            if action == 'stay': return 1.0
            else: return 0.0

    def adjust_difficulty(self, action):
        """تعديل صعوبة السؤال وتحديث Q-table"""
        current_idx = self.env.levels.index(self.env.current_level)
        old_level = self.env.current_level
        
        # تطبيق الإجراء
        if action == 'up' and current_idx < len(self.env.levels) - 1:
            new_idx = current_idx + 1
        elif action == 'down' and current_idx > 0:
            new_idx = current_idx - 1
        else:
            new_idx = current_idx
            
        self.env.current_level = self.env.levels[new_idx]
        
        # حساب المكافأة وتحديث Q-table
        reward = self.calculate_reward(action, self.env.student_ability)
        self.update_q_table(old_level, action, reward, self.env.current_level)
        
        return reward