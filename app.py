import streamlit as st
from core.environment import AdaptiveAssessmentEnv
from core.agent import RLAssessmentAgent
from data.questions import QUESTIONS

# Initialize session state
if "env" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.answer_confirmed = False
    st.session_state.show_results = False

st.title("📘 Adaptive Assessment Quiz")

# Track selection (only show if not initialized)
if not st.session_state.get("initialized", False):
    track = st.selectbox(
        "اختر التخصص:",
        options=list(QUESTIONS.keys()),
        format_func=lambda x: x.upper()
    )
    
    if st.button("بدء الاختبار"):
        st.session_state.env = AdaptiveAssessmentEnv(QUESTIONS, track)
        st.session_state.agent = RLAssessmentAgent(st.session_state.env)
        st.session_state.question = st.session_state.env.get_question(st.session_state.env.current_level)
        st.session_state.initialized = True
        st.session_state.answer_confirmed = False
        st.rerun()

# Main quiz interface
if st.session_state.get("initialized", False) and not st.session_state.get("show_results", False):
    if st.session_state.question:
        q = st.session_state.question
        st.subheader(f"Level {st.session_state.env.current_level}")
        st.markdown(f"**{q['text']}**")
        
        # Store selected answer in session state
        if "selected_answer" not in st.session_state:
            st.session_state.selected_answer = None
            
        st.session_state.selected_answer = st.radio(
            "اختر إجابة:",
            q["options"],
            key=q['text']
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("تأكيد الإجابة", disabled=st.session_state.answer_confirmed):
                st.session_state.answer_confirmed = True
                
        if st.session_state.answer_confirmed:
            with col2:
                if st.button("التالي"):
                    # Process answer
                    reward, done = st.session_state.env.submit_answer(
                        q, 
                        st.session_state.selected_answer
                    )
                    state = st.session_state.env.current_level
                    action = st.session_state.agent.choose_action(state)
                    st.session_state.agent.adjust_difficulty(action)
                    
                    # Reset for next question
                    st.session_state.answer_confirmed = False
                    st.session_state.selected_answer = None
                    
                    if not done:
                        st.session_state.question = st.session_state.env.get_question(
                            st.session_state.env.current_level
                        )
                    else:
                        st.session_state.show_results = True
                    st.rerun()

# Show results when assessment is complete
if st.session_state.get("show_results", False):
    st.success("✅ انتهى الاختبار")
    
    correct = sum(1 for q in st.session_state.env.question_history if q['is_correct'])
    total = len(st.session_state.env.question_history)
    
    st.metric("الدرجة النهائية", f"{correct}/{total}")
    st.metric("مستوى الطالب المتوقع", f"{(st.session_state.env.student_ability)*100:.2f}")
    
    st.subheader("تفاصيل الإجابات:")
    for i, q in enumerate(st.session_state.env.question_history, 1):
        status = "✓" if q['is_correct'] else "✗"
        st.write(f"{i}. {status} Level {q['level']}: {q['question']['text']}")
        st.write(f"   إجابتك: {q['answer']} (الإجابة الصحيحة: {q['question']['correct_answer']})")
    
    if st.button("إعادة الاختبار"):
        st.session_state.clear()
        st.rerun()