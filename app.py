import streamlit as st
from adaptive_assessment.environment import AdaptiveAssessmentEnv
from adaptive_assessment.agent import AdaptiveAgent

# نبدأ environment + agent
env = AdaptiveAssessmentEnv()
agent = AdaptiveAgent()

# نحتفظ بالـ session state (عشان الجلسة ماتروحش بين clicks)
if "state" not in st.session_state:
    st.session_state.state = env.reset()
    st.session_state.done = False
    st.session_state.score = 0
    st.session_state.history = []

st.title("📘 Adaptive Assessment Quiz")
st.write("كل ما تجاوب صح، الأسئلة هتصعب. لو غلط، هتسهل.")

# نعرض السؤال الحالي
if not st.session_state.done:
    question = st.session_state.state["question"]
    choices = st.session_state.state["choices"]

    st.subheader(question)
    user_answer = st.radio("اختار الإجابة:", choices)

    if st.button("Submit"):
        # نحدد الـ index اللي الطالب اختاره
        action = choices.index(user_answer)

        # نلعب خطوة واحدة
        next_state, reward, done, info = env.step(action)

        # نحدث agent
        agent.update(reward)

        # نحفظ النتيجة
        st.session_state.score += reward
        st.session_state.history.append((question, user_answer, reward))

        # نحدث الحالة
        st.session_state.state = next_state
        st.session_state.done = done

else:
    st.success("انتهى الاختبار ✅")
    st.write(f"النتيجة النهائية: {st.session_state.score}")

    st.subheader("📊 History:")
    for q, a, r in st.session_state.history:
        st.write(f"- {q} | إجابتك: {a} | {'✅' if r > 0 else '❌'}")

    if st.button("إعادة الاختبار"):
        st.session_state.state = env.reset()
        st.session_state.done = False
        st.session_state.score = 0
        st.session_state.history = []
