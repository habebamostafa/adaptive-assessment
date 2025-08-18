import streamlit as st
from core.environment import AdaptiveAssessmentEnv
from core.agent import RLAssessmentAgent
from data.questions import generate_question  # استيراد الدالة
import json

# إعداد حالة الجلسة
if "quiz" not in st.session_state:
    st.session_state.quiz = {
        "initialized": False,
        "track": None,
        "questions": [],
        "current_q": None
    }

st.title("🎯 نظام التقييم التكيفي الذكي")

# اختيار المسار
tracks = ["web", "ai", "cyber", "data"]  # تأكد من تطابق الأسماء مع QUESTIONS
if not st.session_state.quiz["initialized"]:
    track = st.selectbox("اختر التخصص:", tracks)
    
    if st.button("بدء الاختبار"):
        st.session_state.env = AdaptiveAssessmentEnv(track)
        st.session_state.agent = RLAssessmentAgent(st.session_state.env)
        st.session_state.quiz.update({
            "initialized": True,
            "track": track,
            "current_q": generate_question(track, st.session_state.env.current_level)
        })
        st.rerun()

# واجهة الاختبار
elif st.session_state.quiz["initialized"] and not st.session_state.quiz.get("completed", False):
    q = st.session_state.quiz["current_q"]
    
    st.subheader(f"السؤال {len(st.session_state.env.question_history)+1}")
    st.markdown(f"**المستوى {st.session_state.env.current_level}**")
    st.markdown(f"### {q['text']}")
    
    answer = st.radio("الخيارات:", q["options"], key=f"q{len(st.session_state.env.question_history)}")
    
    if st.button("إرسال الإجابة"):
        # معالجة الإجابة
        is_correct = answer == q["correct_answer"]
        reward, done = st.session_state.env.submit_answer(q, is_correct)
        
        # تحديث الصعوبة
        state = st.session_state.env.current_level
        action = st.session_state.agent.choose_action(state)
        st.session_state.agent.adjust_difficulty(action)
        
        # تسجيل السؤال
        st.session_state.quiz["questions"].append(q)
        
        if not done:
            # إنشاء سؤال جديد بناءً على المستوى الحالي
            st.session_state.quiz["current_q"] = generate_question(
                st.session_state.quiz["track"], 
                st.session_state.env.current_level
            )
            st.rerun()
        else:
            st.session_state.quiz["completed"] = True
            st.rerun()

# عرض النتائج
if st.session_state.quiz.get("completed", False):
    st.success("✅ اكتمل الاختبار!")
    correct = sum(1 for q in st.session_state.env.question_history if q['is_correct'])
    total = len(st.session_state.env.question_history)
    
    st.metric("الدرجة النهائية", f"{correct}/{total}")
    st.metric("مستوى الطالب", f"{st.session_state.env.student_ability:.2f}")
    
    st.subheader("تفاصيل الإجابات:")
    for i, q in enumerate(st.session_state.env.question_history, 1):
        status = "✓" if q['is_correct'] else "✗"
        st.write(f"{i}. {status} المستوى {q['level']}: {q['question']['text']}")
        st.write(f"   إجابتك: {q['answer']} (الصحيح: {q['question']['correct_answer']})")
    
    if st.button("إعادة الاختبار"):
        st.session_state.clear()
        st.rerun()