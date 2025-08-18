import streamlit as st
from core.environment import AdaptiveAssessmentEnv
from core.agent import RLAssessmentAgent
import json

# إعداد حالة الجلسة
if "quiz" not in st.session_state:
    st.session_state.quiz = {
        "initialized": False,
        "track": None,
        "level": None,
        "questions": [],
        "current_q": None
    }

st.title("🎯 نظام التقييم التكيفي الذكي")

# اختيار المسار
tracks = ["Web Development", "AI", "Cyber Security", "Data Science"]
if not st.session_state.quiz["initialized"]:
    track = st.selectbox("اختر التخصص:", tracks)
    level = st.slider("المستوى الأولي:", 1, 3, 2)
    
    if st.button("بدء الاختبار"):
        st.session_state.env = AdaptiveAssessmentEnv(track)
        st.session_state.agent = RLAssessmentAgent(st.session_state.env)
        st.session_state.quiz.update({
            "initialized": True,
            "track": track,
            "level": level,
            "current_q": generate_question(track, level)
        })
        st.rerun()

# واجهة الاختبار
elif st.session_state.quiz["initialized"]:
    q = st.session_state.quiz["current_q"]
    
    st.subheader(f"المستوى: {st.session_state.env.current_level}")
    st.markdown(f"### {q['text']}")
    
    answer = st.radio("الخيارات:", q["options"])
    
    if st.button("إرسال الإجابة"):
        # معالجة الإجابة
        is_correct = answer == q["correct_answer"]
        reward, done = st.session_state.env.submit_answer(q, is_correct)
        
        # تحديث الصعوبة
        action = st.session_state.agent.choose_action()
        st.session_state.agent.adjust_difficulty(action)
        
        # تسجيل السؤال
        st.session_state.quiz["questions"].append(q)
        
        if not done:
            # إنشاء سؤال جديد بناءً على المستوى الحالي
            new_level = st.session_state.env.current_level
            st.session_state.quiz["current_q"] = generate_question(
                st.session_state.quiz["track"], 
                new_level
            )
            st.rerun()
        else:
            st.session_state.quiz["completed"] = True
            st.rerun()

# عرض النتائج
if st.session_state.quiz.get("completed", False):
    st.success("✅ اكتمل الاختبار!")
    correct = sum(q["is_correct"] for q in st.session_state.env.question_history)
    st.write(f"النتيجة: {correct}/{len(st.session_state.env.question_history)}")
    
    if st.button("إعادة الاختبار"):
        st.session_state.clear()
        st.rerun()