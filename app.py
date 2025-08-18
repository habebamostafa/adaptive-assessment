import json
import streamlit as st
from core.environment import AdaptiveAssessmentEnv
from core.agent import RLAssessmentAgent
from data.questions import generate_question
import json
import plotly.express as px
import pandas as pd

# إعداد الصفحة
st.set_page_config(
    page_title="نظام التقييم التكيفي الذكي",
    page_icon="🎯",
    layout="wide"
)

# إعداد حالة الجلسة
if "quiz" not in st.session_state:
    st.session_state.quiz = {
        "initialized": False,
        "track": None,
        "questions": [],
        "current_q": None,
        "completed": False
    }

st.title("🎯 نظام التقييم التكيفي الذكي")
st.markdown("---")

# اختيار المسار
tracks = {
    "web": "🌐 تطوير الويب",
    "ai": "🤖 الذكاء الاصطناعي", 
    "cyber": "🔒 الأمن السيبراني",
    "data": "📊 علم البيانات"
}

if not st.session_state.quiz["initialized"]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("اختر تخصصك")
        track_options = list(tracks.keys())
        track_labels = [tracks[track] for track in track_options]
        
        selected_idx = st.selectbox(
            "التخصص:",
            range(len(track_options)),
            format_func=lambda x: track_labels[x]
        )
        track = track_options[selected_idx]
        
        st.markdown(f"""
        ### معلومات الاختبار:
        - **عدد الأسئلة**: 10 أسئلة
        - **المستويات**: 3 مستويات (سهل، متوسط، صعب)
        - **التكيف**: النظام يتكيف مع مستوى أدائك
        - **الوقت**: غير محدود
        """)
        
        if st.button("🚀 بدء الاختبار", type="primary", use_container_width=True):
            try:
                st.session_state.env = AdaptiveAssessmentEnv(track)
                st.session_state.agent = RLAssessmentAgent(st.session_state.env)
                st.session_state.quiz.update({
                    "initialized": True,
                    "track": track,
                    "current_q": generate_question(track, st.session_state.env.current_level)
                })
                st.rerun()
            except Exception as e:
                st.error(f"خطأ في بدء الاختبار: {str(e)}")
    
    with col2:
        st.markdown("### 📈 مزايا النظام")
        st.markdown("""
        - تقييم تكيفي ذكي
        - أسئلة متدرجة الصعوبة  
        - تتبع الأداء الفوري
        - تحليل مفصل للنتائج
        - واجهة سهلة الاستخدام
        """)

# واجهة الاختبار
elif st.session_state.quiz["initialized"] and not st.session_state.quiz.get("completed", False):
    try:
        q = st.session_state.quiz["current_q"]
        question_num = len(st.session_state.env.question_history) + 1
        progress = question_num / st.session_state.env.max_questions
        
        # شريط التقدم
        st.progress(progress, text=f"السؤال {question_num} من {st.session_state.env.max_questions}")
        
        # معلومات الحالة الحالية
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("المستوى الحالي", st.session_state.env.current_level)
        with col2:
            st.metric("قدرة الطالب", f"{st.session_state.env.student_ability:.2f}")
        with col3:
            st.metric("إجابات صحيحة متتالية", st.session_state.env.consecutive_correct)
        with col4:
            st.metric("إجابات خاطئة متتالية", st.session_state.env.consecutive_incorrect)
        
        st.markdown("---")
        
        # السؤال
        st.subheader(f"السؤال {question_num}")
        
        # مستوى الصعوبة
        difficulty_colors = {1: "🟢", 2: "🟡", 3: "🔴"}
        difficulty_names = {1: "سهل", 2: "متوسط", 3: "صعب"}
        
        st.markdown(
            f"**المستوى**: {difficulty_colors[st.session_state.env.current_level]} "
            f"{difficulty_names[st.session_state.env.current_level]}"
        )
        
        st.markdown(f"### {q['text']}")
        
        # الخيارات
        answer = st.radio(
            "اختر الإجابة الصحيحة:",
            q["options"],
            key=f"q{question_num}",
            index=None
        )
        
        if answer and st.button("✅ إرسال الإجابة", type="primary", use_container_width=True):
            # معالجة الإجابة
            is_correct = answer == q["correct_answer"]
            
            # تسجيل الإجابة في البيئة
            reward, done = st.session_state.env.submit_answer(q, answer, is_correct)
            
            # عرض النتيجة الفورية
            if is_correct:
                st.success("✅ إجابة صحيحة!")
            else:
                st.error(f"❌ إجابة خاطئة. الإجابة الصحيحة: {q['correct_answer']}")
            
            # قرار الوكيل لتعديل المستوى
            if not done:
                action = st.session_state.agent.choose_action()
                st.session_state.agent.adjust_difficulty(action)
                
                # إظهار قرار الوكيل
                action_messages = {
                    'up': "⬆️ رفع مستوى الصعوبة",
                    'down': "⬇️ تخفيض مستوى الصعوبة", 
                    'stay': "➡️ البقاء في نفس المستوى"
                }
                st.info(f"قرار النظام: {action_messages[action]}")
            
            if not done:
                # إنشاء السؤال التالي
                st.session_state.quiz["current_q"] = generate_question(
                    st.session_state.quiz["track"], 
                    st.session_state.env.current_level
                )
                
                # انتظار قصير ثم إعادة تحميل
                st.balloons()
                st.rerun()
            else:
                st.session_state.quiz["completed"] = True
                st.success("🎉 تم انتهاء الاختبار!")
                st.rerun()
                
    except Exception as e:
        st.error(f"خطأ في عرض السؤال: {str(e)}")
        st.json(st.session_state.quiz["current_q"])  # لتشخيص المشكلة

# عرض النتائج
elif st.session_state.quiz.get("completed", False):
    st.success("🎉 تهانينا! لقد أكملت الاختبار بنجاح!")
    
    # إحصائيات الأداء
    stats = st.session_state.env.get_performance_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("الدرجة النهائية", 
                 f"{int(stats['accuracy'] * 100)}%",
                 f"{stats['improvement']:.2%}")
    with col2:
        st.metric("المستوى النهائي", 
                 st.session_state.env.current_level)
    with col3:
        st.metric("قدرة الطالب النهائية", 
                 f"{stats['final_ability']:.2f}")
    
    # رسم بياني لتوزيع المستويات
    st.subheader("📊 تحليل الأداء")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # توزيع الأسئلة حسب المستوى
        level_data = pd.DataFrame([
            {"المستوى": f"المستوى {level}", "عدد الأسئلة": count}
            for level, count in stats['level_distribution'].items()
        ])
        
        if not level_data.empty:
            fig = px.pie(level_data, values="عدد الأسئلة", names="المستوى", 
                        title="توزيع الأسئلة حسب المستوى")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # تطور الأداء عبر الأسئلة
        if st.session_state.env.question_history:
            performance_data = []
            for i, q in enumerate(st.session_state.env.question_history):
                performance_data.append({
                    "رقم السؤال": i + 1,
                    "النتيجة": 1 if q['is_correct'] else 0,
                    "المستوى": q['level']
                })
            
            df = pd.DataFrame(performance_data)
            fig = px.line(df, x="رقم السؤال", y="النتيجة", 
                         title="تطور الأداء عبر الأسئلة",
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
    
    # تفاصيل الأسئلة
    st.subheader("📋 تفاصيل الإجابات")
    
    for i, q_record in enumerate(st.session_state.env.question_history, 1):
        with st.expander(f"السؤال {i} - {'✅ صحيح' if q_record['is_correct'] else '❌ خطأ'}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**السؤال**: {q_record['question']['text']}")
                st.write(f"**إجابتك**: {q_record['answer']}")
                st.write(f"**الإجابة الصحيحة**: {q_record['question']['correct_answer']}")
            with col2:
                st.metric("المستوى", q_record['level'])
                st.metric("القدرة بعد السؤال", f"{q_record['student_ability_after']:.2f}")
    
    # خيارات ما بعد الانتهاء
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 اختبار جديد", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    with col2:
        if st.button("📊 تصدير النتائج", use_container_width=True):
            # إنشاء ملف JSON للنتائج
            results = {
                "track": tracks[st.session_state.quiz["track"]],
                "performance": stats,
                "questions": st.session_state.env.question_history
            }
            st.download_button(
                "تحميل النتائج (JSON)",
                json.dumps(results, ensure_ascii=False, indent=2),
                file_name=f"assessment_results_{st.session_state.quiz['track']}.json",
                mime="application/json"
            )