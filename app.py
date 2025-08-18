import streamlit as st
from core.environment import AdaptiveAssessmentEnv
from core.agent import RLAssessmentAgent
from data.questions import QUESTIONS

# اختيار التراك
track = "web"
if "env" not in st.session_state:
    st.session_state.env = AdaptiveAssessmentEnv(QUESTIONS, track)
    st.session_state.agent = RLAssessmentAgent(st.session_state.env)
    st.session_state.question = st.session_state.env.get_question(st.session_state.env.current_level)

st.title("📘 Adaptive Assessment Quiz")

if st.session_state.question:
    q = st.session_state.question
    st.subheader(f"Level {st.session_state.env.current_level}: {q['text']}")
    user_answer = st.radio("اختر إجابة:", q["options"], key=q['text'])

    if st.button("Submit Answer"):
        reward, done = st.session_state.env.submit_answer(q, user_answer)
        state = st.session_state.env.current_level
        action = st.session_state.agent.choose_action(state)
        st.session_state.agent.adjust_difficulty(action)
        
        if not done:
            st.session_state.question = st.session_state.env.get_question(st.session_state.env.current_level)
        else:
            st.success("✅ انتهى الاختبار")
            st.write(f"🎯 الدرجة النهائية: {sum(1 for q in st.session_state.env.question_history if q['is_correct'])} / {len(st.session_state.env.question_history)}")
            st.write(f"📈 مستوى الطالب المتوقع: {st.session_state.env.student_ability:.2f}")
else:
    st.warning("No more questions available.")
