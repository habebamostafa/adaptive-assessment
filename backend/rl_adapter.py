#!/usr/bin/env python3
"""
RL Adapter Script - يربط بين Node.js backend والنظام الذكي للأسئلة
"""

import sys
import json
import random
from typing import Dict, List, Optional
import uuid

# استيراد النظام الذكي للأسئلة
try:
    from questions import AdaptiveMCQGenerator, get_adaptive_question, generate_interview_questions
except ImportError:
    # إذا مش موجود، هنعمل نسخة مبسطة
    class AdaptiveMCQGenerator:
        def __init__(self):
            self.question_pools = {
                "web": {
                    1: [
                        {
                            'id': 'web_1_1',
                            'text': "ما هي اللغة الأساسية لبناء صفحات الويب؟",
                            'options': ["HTML", "Python", "Java", "C++"],
                            'correct_answer': "HTML",
                            'explanation': "HTML هي لغة ترميز النص التشعبي المستخدمة لإنشاء صفحات الويب",
                            'level': 1
                        },
                        {
                            'id': 'web_1_2',
                            'text': "ماذا يعني CSS؟",
                            'options': ["Computer Style Sheets", "Creative Style Sheets", "Cascading Style Sheets", "Colorful Style Sheets"],
                            'correct_answer': "Cascading Style Sheets",
                            'explanation': "CSS تتحكم في التصميم والشكل البصري لصفحات الويب",
                            'level': 1
                        }
                    ],
                    2: [
                        {
                            'id': 'web_2_1',
                            'text': "أي React Hook يُستخدم للتأثيرات الجانبية؟",
                            'options': ["useState", "useEffect", "useContext", "useReducer"],
                            'correct_answer': "useEffect",
                            'explanation': "useEffect يتعامل مع التأثيرات الجانبية مثل استدعاء APIs",
                            'level': 2
                        }
                    ],
                    3: [
                        {
                            'id': 'web_3_1',
                            'text': "أي رمز HTTP يشير إلى 'غير موجود'؟",
                            'options': ["200", "301", "404", "500"],
                            'correct_answer': "404",
                            'explanation': "404 يشير إلى أن المورد المطلوب غير موجود على الخادم",
                            'level': 3
                        }
                    ]
                },
                "ai": {
                    1: [
                        {
                            'id': 'ai_1_1',
                            'text': "ما هي أكثر لغات البرمجة استخداماً في الذكاء الاصطناعي؟",
                            'options': ["Python", "C#", "Ruby", "Go"],
                            'correct_answer': "Python",
                            'explanation': "Python مستخدمة بكثرة في الذكاء الاصطناعي بسبب بساطتها ومكتباتها الواسعة",
                            'level': 1
                        }
                    ],
                    2: [
                        {
                            'id': 'ai_2_1',
                            'text': "أي خوارزمية الأفضل لمهام التصنيف؟",
                            'options': ["Linear Regression", "K-Means", "Decision Tree", "DBSCAN"],
                            'correct_answer': "Decision Tree",
                            'explanation': "أشجار القرار ممتازة لمشاكل التصنيف بسبب قابليتها للتفسير",
                            'level': 2
                        }
                    ],
                    3: [
                        {
                            'id': 'ai_3_1',
                            'text': "ماذا يعني CNN في التعلم العميق؟",
                            'options': ["Common Neural Network", "Convolutional Neural Network", "Complex Neural Node", "Centralized Neural Network"],
                            'correct_answer': "Convolutional Neural Network",
                            'explanation': "CNNs متخصصة في معالجة البيانات الشبكية مثل الصور",
                            'level': 3
                        }
                    ]
                },
                "cyber": {
                    1: [
                        {
                            'id': 'cyber_1_1',
                            'text': "ما هو أكثر أنواع الهجمات السيبرانية شيوعاً؟",
                            'options': ["Phishing", "DDoS", "MITM", "Zero-day"],
                            'correct_answer': "Phishing",
                            'explanation': "هجمات التصيد تخدع المستخدمين للكشف عن معلومات حساسة",
                            'level': 1
                        }
                    ],
                    2: [
                        {
                            'id': 'cyber_2_1',
                            'text': "ماذا يعني VPN؟",
                            'options': ["Virtual Private Network", "Verified Personal Node", "Virtual Public Network", "Verified Protocol Network"],
                            'correct_answer': "Virtual Private Network",
                            'explanation': "VPN تنشئ اتصالات آمنة عبر الشبكات العامة",
                            'level': 2
                        }
                    ],
                    3: [
                        {
                            'id': 'cyber_3_1',
                            'text': "أي خوارزمية تشفير غير متماثلة؟",
                            'options': ["AES", "RSA", "DES", "3DES"],
                            'correct_answer': "RSA",
                            'explanation': "RSA تستخدم مفاتيح مختلفة للتشفير وفك التشفير",
                            'level': 3
                        }
                    ]
                }
            }
            
        def get_question(self, track: str, level: int, exclude_used: List[str] = None):
            if track not in self.question_pools or level not in self.question_pools[track]:
                return None
            
            available_questions = self.question_pools[track][level].copy()
            
            if exclude_used:
                available_questions = [q for q in available_questions if q['id'] not in exclude_used]
            
            if not available_questions:
                return None
                
            return random.choice(available_questions)

# المتغيرات العامة لحفظ حالة الجلسات
active_sessions = {}
question_generator = AdaptiveMCQGenerator()

class RLAssessmentSystem:
    def __init__(self):
        self.sessions = {}
        
    def init_environment(self, data):
        """تهيئة بيئة التقييم"""
        session_data = {
            'track': data.get('track', 'web'),
            'max_questions': data.get('maxQuestions', 10),
            'confidence_threshold': data.get('confidenceThreshold', 0.8),
            'agent_type': data.get('agentType', 'main'),
            'adaptation_strategy': data.get('adaptationStrategy', 'rl_based'),
            'current_ability': 0.5,
            'current_level': 2,
            'questions_asked': 0,
            'used_questions': [],
            'performance_history': []
        }
        
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = session_data
        
        return {
            'success': True,
            'session_id': session_id,
            'message': 'Environment initialized successfully'
        }
    
    def get_question(self, data):
        """الحصول على سؤال تكيفي"""
        session_id = data.get('sessionId')
        current_state = data.get('currentState', {})
        
        # تحديد المستوى بناءً على قدرة الطالب
        ability = current_state.get('ability', 0.5)
        if ability < 0.3:
            level = 1  # سهل
        elif ability < 0.7:
            level = 2  # متوسط
        else:
            level = 3  # صعب
        
        # إذا كان لدينا session، نستخدم بياناتها
        track = 'web'  # default
        used_questions = []
        
        if session_id and session_id in self.sessions:
            session_data = self.sessions[session_id]
            track = session_data['track']
            used_questions = session_data['used_questions']
        
        # الحصول على السؤال
        question = question_generator.get_question(track, level, used_questions)
        
        if not question:
            return {'error': 'No questions available'}
        
        # إضافة معلومات إضافية للسؤال
        question['number'] = len(used_questions) + 1
        question['total'] = 10  # عدد الأسئلة الإجمالي
        question['level'] = level
        
        return question
    
    def submit_answer(self, data):
        """معالجة إجابة ومن ثم تحديث النموذج"""
        session_id = data.get('sessionId')
        question_data = data.get('questionData', {})
        selected_answer = data.get('selectedAnswer')
        current_state = data.get('currentState', {})
        
        # تحديد صحة الإجابة
        correct_answer = question_data.get('correct_answer')
        is_correct = selected_answer == correct_answer
        
        # حساب القدرة الجديدة (نموذج تكيفي مبسط)
        current_ability = current_state.get('ability', 0.5)
        
        if is_correct:
            new_ability = min(1.0, current_ability + 0.1)  # زيادة القدرة
        else:
            new_ability = max(0.0, current_ability - 0.05)  # تقليل القدرة
        
        # تحديد المستوى الجديد
        if new_ability < 0.3:
            new_level = 1
        elif new_ability < 0.7:
            new_level = 2
        else:
            new_level = 3
        
        # حساب الثقة (بناءً على الاستقرار في الإجابات)
        confidence = min(0.95, abs(new_ability - 0.5) * 2)
        
        # تحديث بيانات الجلسة
        if session_id and session_id in self.sessions:
            session_data = self.sessions[session_id]
            session_data['current_ability'] = new_ability
            session_data['current_level'] = new_level
            session_data['questions_asked'] += 1
            session_data['used_questions'].append(question_data.get('id', ''))
            
            session_data['performance_history'].append({
                'question_number': session_data['questions_asked'],
                'is_correct': is_correct,
                'ability': new_ability,
                'level': new_level
            })
            
            # تحديد إذا كان التقييم مكتمل
            done = (session_data['questions_asked'] >= session_data['max_questions'] or 
                   confidence >= session_data['confidence_threshold'])
            
            if done:
                # حساب إحصائيات نهائية
                correct_count = sum(1 for p in session_data['performance_history'] if p['is_correct'])
                
                return {
                    'isCorrect': is_correct,
                    'explanation': question_data.get('explanation', ''),
                    'newAbility': new_ability,
                    'newLevel': new_level,
                    'confidence': confidence,
                    'done': True,
                    'recommendedLevel': new_level,
                    'agentMetrics': {
                        'final_ability': new_ability,
                        'confidence_score': confidence,
                        'total_questions': session_data['questions_asked'],
                        'correct_answers': correct_count
                    },
                    'levelPerformance': self._calculate_level_performance(session_data['performance_history'])
                }
        
        return {
            'isCorrect': is_correct,
            'explanation': question_data.get('explanation', ''),
            'newAbility': new_ability,
            'newLevel': new_level,
            'confidence': confidence,
            'done': False
        }
    
    def get_next_question(self, data):
        """الحصول على السؤال التالي"""
        return self.get_question(data)
    
    def get_analytics(self, data):
        """تحليلات الأداء"""
        session_id = data.get('sessionId')
        
        if session_id not in self.sessions:
            return {'error': 'Session not found'}
        
        session_data = self.sessions[session_id]
        performance_history = session_data['performance_history']
        
        if not performance_history:
            return {'message': 'No performance data available'}
        
        # حساب الإحصائيات
        total_questions = len(performance_history)
        correct_answers = sum(1 for p in performance_history if p['is_correct'])
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        # تحليل الأداء عبر الوقت
        ability_progression = [p['ability'] for p in performance_history]
        level_distribution = {}
        for p in performance_history:
            level = p['level']
            if level not in level_distribution:
                level_distribution[level] = {'total': 0, 'correct': 0}
            level_distribution[level]['total'] += 1
            if p['is_correct']:
                level_distribution[level]['correct'] += 1
        
        return {
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'final_ability': session_data['current_ability'],
            'ability_progression': ability_progression,
            'level_distribution': level_distribution,
            'performance_trends': self._analyze_trends(performance_history)
        }
    
    def get_recommendations(self, data):
        """توصيات تكيفية للتعلم"""
        current_ability = data.get('currentAbility', 0.5)
        track = data.get('track', 'web')
        assessment_history = data.get('assessmentHistory', [])
        
        # تحديد المستوى الموصى به
        if current_ability < 0.3:
            recommended_difficulty = 1
        elif current_ability < 0.7:
            recommended_difficulty = 2
        else:
            recommended_difficulty = 3
        
        # توصيات الدروس بناءً على المسار والقدرة
        lesson_recommendations = self._generate_lesson_recommendations(track, current_ability)
        
        # تحليل نقاط القوة والضعف
        strengths, weaknesses = self._analyze_strengths_weaknesses(assessment_history)
        
        # مسار التعلم المقترح
        learning_path = self._generate_learning_path(track, current_ability)
        
        # تقدير الوقت للإتقان
        estimated_time = self._estimate_mastery_time(current_ability, assessment_history)
        
        return {
            'recommendedDifficulty': recommended_difficulty,
            'lessons': lesson_recommendations,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'learningPath': learning_path,
            'estimatedTime': estimated_time
        }
    
    def get_tracks(self, data):
        """الحصول على المسارات المتاحة"""
        tracks = {
            'web': {
                'name': 'تطوير الويب',
                'description': 'HTML, CSS, JavaScript, React, Node.js',
                'levels': 3,
                'total_questions': len(question_generator.question_pools.get('web', {}))
            },
            'ai': {
                'name': 'الذكاء الاصطناعي',
                'description': 'Machine Learning, Deep Learning, NLP',
                'levels': 3,
                'total_questions': len(question_generator.question_pools.get('ai', {}))
            },
            'cyber': {
                'name': 'الأمن السيبراني',
                'description': 'Network Security, Cryptography, Ethical Hacking',
                'levels': 3,
                'total_questions': len(question_generator.question_pools.get('cyber', {}))
            }
        }
        
        return {'tracks': tracks}
    
    def _calculate_level_performance(self, performance_history):
        """حساب الأداء حسب المستوى"""
        level_performance = {}
        
        for performance in performance_history:
            level = performance['level']
            if level not in level_performance:
                level_performance[level] = {'total': 0, 'correct': 0, 'times': []}
            
            level_performance[level]['total'] += 1
            if performance['is_correct']:
                level_performance[level]['correct'] += 1
        
        # حساب الدقة لكل مستوى
        for level in level_performance:
            total = level_performance[level]['total']
            correct = level_performance[level]['correct']
            level_performance[level]['accuracy'] = correct / total if total > 0 else 0
            level_performance[level]['avgTime'] = 30  # وقت افتراضي
        
        return level_performance
    
    def _analyze_trends(self, performance_history):
        """تحليل اتجاهات الأداء"""
        if len(performance_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_performance = performance_history[-3:]  # آخر 3 أسئلة
        early_performance = performance_history[:3]    # أول 3 أسئلة
        
        recent_accuracy = sum(1 for p in recent_performance if p['is_correct']) / len(recent_performance)
        early_accuracy = sum(1 for p in early_performance if p['is_correct']) / len(early_performance)
        
        if recent_accuracy > early_accuracy + 0.1:
            trend = 'improving'
        elif recent_accuracy < early_accuracy - 0.1:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'recent_accuracy': recent_accuracy,
            'early_accuracy': early_accuracy,
            'improvement': recent_accuracy - early_accuracy
        }
    
    def _generate_lesson_recommendations(self, track, ability):
        """إنشاء توصيات الدروس"""
        lessons = {
            'web': [
                {'title': 'HTML الأساسي', 'level': 1, 'duration': 30, 'description': 'تعلم أساسيات HTML'},
                {'title': 'CSS المتقدم', 'level': 2, 'duration': 45, 'description': 'تطوير مهارات التصميم'},
                {'title': 'JavaScript ES6+', 'level': 3, 'duration': 60, 'description': 'البرمجة المتقدمة'}
            ],
            'ai': [
                {'title': 'مقدمة في Python', 'level': 1, 'duration': 40, 'description': 'أساسيات البرمجة'},
                {'title': 'Machine Learning', 'level': 2, 'duration': 90, 'description': 'تعلم الآلة'},
                {'title': 'Deep Learning', 'level': 3, 'duration': 120, 'description': 'التعلم العميق'}
            ],
            'cyber': [
                {'title': 'أمان الشبكات', 'level': 1, 'duration': 45, 'description': 'أساسيات الأمان'},
                {'title': 'Ethical Hacking', 'level': 2, 'duration': 75, 'description': 'الاختراق الأخلاقي'},
                {'title': 'Cryptography', 'level': 3, 'duration': 90, 'description': 'علم التشفير'}
            ]
        }
        
        track_lessons = lessons.get(track, [])
        
        # تصفية الدروس بناءً على مستوى القدرة
        if ability < 0.3:
            recommended_level = 1
        elif ability < 0.7:
            recommended_level = 2
        else:
            recommended_level = 3
        
        return [lesson for lesson in track_lessons if lesson['level'] <= recommended_level + 1]
    
    def _analyze_strengths_weaknesses(self, assessment_history):
        """تحليل نقاط القوة والضعف"""
        if not assessment_history:
            return [], []
        
        # تحليل بسيط بناءً على الأداء التاريخي
        recent_assessments = assessment_history[-3:] if len(assessment_history) >= 3 else assessment_history
        
        strengths = []
        weaknesses = []
        
        for assessment in recent_assessments:
            if assessment.get('finalAbility', 0) > 0.7:
                strengths.append(f"أداء جيد في {assessment.get('track', 'المجال')}")
            elif assessment.get('finalAbility', 0) < 0.3:
                weaknesses.append(f"يحتاج تحسين في {assessment.get('track', 'المجال')}")
        
        if not strengths:
            strengths = ["استمرارية في التعلم", "الرغبة في التطوير"]
        
        if not weaknesses:
            weaknesses = ["يمكن تحسين سرعة الحل", "التركيز على الممارسة"]
        
        return strengths[:3], weaknesses[:3]  # أقصى 3 عناصر لكل قائمة
    
    def _generate_learning_path(self, track, ability):
        """إنشاء مسار تعلم مخصص"""
        paths = {
            'web': ['HTML/CSS الأساسي', 'JavaScript', 'React', 'Node.js', 'Full Stack'],
            'ai': ['Python للمبتدئين', 'Data Science', 'Machine Learning', 'Deep Learning', 'AI Applications'],
            'cyber': ['Network Security', 'Penetration Testing', 'Incident Response', 'Advanced Threats', 'Security Architecture']
        }
        
        track_path = paths.get(track, ['تعلم الأساسيات', 'تطوير المهارات', 'التطبيق العملي'])
        
        # تحديد نقطة البداية بناءً على القدرة
        if ability < 0.3:
            start_index = 0
        elif ability < 0.7:
            start_index = 1
        else:
            start_index = 2
        
        return track_path[start_index:]
    
    def _estimate_mastery_time(self, ability, assessment_history):
        """تقدير الوقت المطلوب للإتقان"""
        # حساب بسيط بناءً على القدرة الحالية والتقدم
        base_time = 100  # ساعات أساسية
        
        if ability > 0.8:
            return 20  # قريب من الإتقان
        elif ability > 0.6:
            return 40
        elif ability > 0.4:
            return 60
        else:
            return 80
        
        return min(base_time, 100)

# إنشاء نسخة من النظام
rl_system = RLAssessmentSystem()

def main():
    """الدالة الرئيسية للتعامل مع طلبات Node.js"""
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Missing action or data'}))
        return
    
    action = sys.argv[1]
    try:
        data = json.loads(sys.argv[2])
    except json.JSONDecodeError:
        print(json.dumps({'error': 'Invalid JSON data'}))
        return
    
    try:
        if action == 'init_environment':
            result = rl_system.init_environment(data)
        elif action == 'get_question':
            result = rl_system.get_question(data)
        elif action == 'submit_answer':
            result = rl_system.submit_answer(data)
        elif action == 'get_next_question':
            result = rl_system.get_next_question(data)
        elif action == 'get_analytics':
            result = rl_system.get_analytics(data)
        elif action == 'get_recommendations':
            result = rl_system.get_recommendations(data)
        elif action == 'get_tracks':
            result = rl_system.get_tracks(data)
        else:
            result = {'error': f'Unknown action: {action}'}
        
        print(json.dumps(result, ensure_ascii=False))
    
    except Exception as e:
        print(json.dumps({'error': str(e)}, ensure_ascii=False))

if __name__ == '__main__':
    main()