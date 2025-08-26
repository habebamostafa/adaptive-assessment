import streamlit as st
import requests
import sqlite3
import pandas as pd
from datetime import datetime
import json
import os

# Page configuration
st.set_page_config(
    page_title="نظام التقييم الذكي",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database setup
def init_database():
    """Initialize SQLite database for students"""
    conn = sqlite3.connect('students.db')
    cursor = conn.cursor()
    
    # Create students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            track TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create assessment_results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            app_type TEXT NOT NULL,
            score REAL,
            ability_level REAL,
            questions_answered INTEGER,
            time_spent INTEGER,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            details TEXT,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')
    
    # Create interview_sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interview_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            track TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            questions_count INTEGER,
            engagement_score REAL,
            feedback TEXT,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Student management functions
def add_student(name, email, track):
    """Add new student to database"""
    try:
        conn = sqlite3.connect('students.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students (name, email, track) VALUES (?, ?, ?)",
            (name, email, track)
        )
        student_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return student_id
    except sqlite3.IntegrityError:
        return None

def get_student_by_email(email):
    """Get student by email"""
    conn = sqlite3.connect('students.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE email = ?", (email,))
    student = cursor.fetchone()
    conn.close()
    return student

def get_all_students():
    """Get all students"""
    conn = sqlite3.connect('students.db')
    df = pd.read_sql_query("SELECT * FROM students", conn)
    conn.close()
    return df

def save_assessment_result(student_id, app_type, score, ability_level, 
                          questions_answered, time_spent, details):
    """Save assessment result to database"""
    conn = sqlite3.connect('students.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO assessment_results 
        (student_id, app_type, score, ability_level, questions_answered, time_spent, details)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (student_id, app_type, score, ability_level, questions_answered, time_spent, details))
    conn.commit()
    conn.close()

def get_student_results(student_id):
    """Get all results for a student"""
    conn = sqlite3.connect('students.db')
    df = pd.read_sql_query(
        "SELECT * FROM assessment_results WHERE student_id = ? ORDER BY completed_at DESC",
        conn, params=(student_id,)
    )
    conn.close()
    return df

# n8n integration functions
def trigger_n8n_workflow(workflow_name, data):
    """Trigger n8n workflow"""
    try:
        # Replace with your local n8n webhook URL
        n8n_url = f"http://localhost:5678/webhook/{workflow_name}"
        response = requests.post(n8n_url, json=data, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Initialize database
init_database()

# Session state initialization
if 'current_student' not in st.session_state:
    st.session_state.current_student = None
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

# CSS styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .app-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 2px solid #f0f2f6;
        transition: all 0.3s ease;
    }
    
    .app-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    
    .student-info {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 نظام التقييم الذكي المتكامل</h1>
    <p>اختبر مهاراتك التقنية واحصل على تقييم شامل</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for login/registration
with st.sidebar:
    st.header("👤 تسجيل الدخول")
    
    # Admin toggle
    is_admin = st.checkbox("وضع المشرف", value=st.session_state.is_admin)
    st.session_state.is_admin = is_admin
    
    if not st.session_state.current_student and not is_admin:
        # Student login/registration
        tab1, tab2 = st.tabs(["تسجيل دخول", "حساب جديد"])
        
        with tab1:
            email = st.text_input("البريد الإلكتروني")
            if st.button("دخول", use_container_width=True):
                student = get_student_by_email(email)
                if student:
                    st.session_state.current_student = {
                        'id': student[0],
                        'name': student[1],
                        'email': student[2],
                        'track': student[3]
                    }
                    st.success(f"مرحباً {student[1]}")
                    st.rerun()
                else:
                    st.error("البريد الإلكتروني غير موجود")
        
        with tab2:
            name = st.text_input("الاسم")
            email_reg = st.text_input("البريد الإلكتروني", key="reg_email")
            track = st.selectbox("المجال", [
                "Web Development", 
                "AI/ML", 
                "Mobile Development",
                "Data Science",
                "Cybersecurity"
            ])
            
            if st.button("إنشاء حساب", use_container_width=True):
                if name and email_reg:
                    student_id = add_student(name, email_reg, track)
                    if student_id:
                        st.session_state.current_student = {
                            'id': student_id,
                            'name': name,
                            'email': email_reg,
                            'track': track
                        }
                        st.success("تم إنشاء الحساب بنجاح!")
                        
                        # Trigger n8n workflow for new student
                        trigger_n8n_workflow("new_student", {
                            'student_id': student_id,
                            'name': name,
                            'email': email_reg,
                            'track': track
                        })
                        
                        st.rerun()
                    else:
                        st.error("البريد الإلكتروني مستخدم بالفعل")
                else:
                    st.error("يرجى ملء جميع الحقول")

# Main content
if st.session_state.is_admin:
    # Admin dashboard
    st.header("🔧 لوحة تحكم المشرف")
    
    tab1, tab2, tab3 = st.tabs(["إحصائيات عامة", "إدارة الطلاب", "تقارير"])
    
    with tab1:
        # General statistics
        students_df = get_all_students()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>👥</h3>
                <h2>{}</h2>
                <p>إجمالي الطلاب</p>
            </div>
            """.format(len(students_df)), unsafe_allow_html=True)
        
        with col2:
            # Get assessment count
            conn = sqlite3.connect('students.db')
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM assessment_results")
            assessment_count = cursor.fetchone()[0]
            conn.close()
            
            st.markdown("""
            <div class="metric-card">
                <h3>📝</h3>
                <h2>{}</h2>
                <p>اختبارات مكتملة</p>
            </div>
            """.format(assessment_count), unsafe_allow_html=True)
        
        with col3:
            # Get interview count
            conn = sqlite3.connect('students.db')
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM interview_sessions")
            interview_count = cursor.fetchone()[0]
            conn.close()
            
            st.markdown("""
            <div class="metric-card">
                <h3>🎤</h3>
                <h2>{}</h2>
                <p>مقابلات مكتملة</p>
            </div>
            """.format(interview_count), unsafe_allow_html=True)
        
        with col4:
            # Average score
            conn = sqlite3.connect('students.db')
            cursor = conn.cursor()
            cursor.execute("SELECT AVG(score) FROM assessment_results WHERE score IS NOT NULL")
            avg_score = cursor.fetchone()[0] or 0
            conn.close()
            
            st.markdown("""
            <div class="metric-card">
                <h3>📊</h3>
                <h2>{:.1%}</h2>
                <p>متوسط الدرجات</p>
            </div>
            """.format(avg_score), unsafe_allow_html=True)
        
        # Track distribution
        if not students_df.empty:
            st.subheader("توزيع الطلاب حسب المجال")
            track_counts = students_df['track'].value_counts()
            st.bar_chart(track_counts)
    
    with tab2:
        # Student management
        st.subheader("قائمة الطلاب")
        
        if not students_df.empty:
            # Add search functionality
            search_term = st.text_input("البحث عن طالب...")
            if search_term:
                filtered_df = students_df[
                    students_df['name'].str.contains(search_term, case=False, na=False) |
                    students_df['email'].str.contains(search_term, case=False, na=False)
                ]
            else:
                filtered_df = students_df
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Student details
            if st.selectbox("عرض تفاصيل الطالب", [""] + filtered_df['name'].tolist()):
                selected_name = st.selectbox("عرض تفاصيل الطالب", [""] + filtered_df['name'].tolist())
                if selected_name:
                    student_info = filtered_df[filtered_df['name'] == selected_name].iloc[0]
                    student_results = get_student_results(student_info['id'])
                    
                    st.markdown(f"""
                    <div class="student-info">
                        <h4>📋 معلومات الطالب</h4>
                        <p><strong>الاسم:</strong> {student_info['name']}</p>
                        <p><strong>البريد:</strong> {student_info['email']}</p>
                        <p><strong>المجال:</strong> {student_info['track']}</p>
                        <p><strong>تاريخ التسجيل:</strong> {student_info['created_at']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if not student_results.empty:
                        st.subheader("نتائج الطالب")
                        st.dataframe(student_results, use_container_width=True)
        else:
            st.info("لا يوجد طلاب مسجلين حالياً")
    
    with tab3:
        # Reports
        st.subheader("التقارير والتحليلات")
        
        # Performance trends
        conn = sqlite3.connect('students.db')
        results_df = pd.read_sql_query("""
            SELECT DATE(completed_at) as date, AVG(score) as avg_score, COUNT(*) as count
            FROM assessment_results 
            WHERE completed_at >= date('now', '-30 days')
            GROUP BY DATE(completed_at)
            ORDER BY date
        """, conn)
        conn.close()
        
        if not results_df.empty:
            st.subheader("الأداء خلال آخر 30 يوم")
            st.line_chart(results_df.set_index('date')['avg_score'])

elif st.session_state.current_student:
    # Student interface
    student = st.session_state.current_student
    
    st.markdown(f"""
    <div class="student-info">
        <h3>👋 مرحباً {student['name']}</h3>
        <p>📧 {student['email']} | 🎯 {student['track']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Logout button
    if st.button("تسجيل خروج", use_container_width=True):
        st.session_state.current_student = None
        st.rerun()
    
    # App selection
    st.header("🎯 اختر نوع التقييم")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="app-card">
            <h3>🎯 الاختبار التكيفي</h3>
            <p>اختبار ذكي يتكيف مع مستواك ويقيم قدراتك التقنية بدقة</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("بدء الاختبار التكيفي", use_container_width=True, type="primary"):
            # Trigger n8n workflow
            trigger_n8n_workflow("start_assessment", {
                'student_id': student['id'],
                'app_type': 'adaptive',
                'track': student['track']
            })
            
            st.success("يتم تحضير الاختبار...")
            st.info("سيتم توجيهك إلى تطبيق الاختبار قريباً")
    
    with col2:
        st.markdown("""
        <div class="app-card">
            <h3>🤖 المقابلة الذكية</h3>
            <p>مقابلة تفاعلية مع الذكاء الاصطناعي لتقييم مهاراتك التواصلية والتقنية</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("بدء المقابلة الذكية", use_container_width=True, type="primary"):
            # Trigger n8n workflow
            trigger_n8n_workflow("start_interview", {
                'student_id': student['id'],
                'app_type': 'interview',
                'track': student['track']
            })
            
            st.success("يتم تحضير المقابلة...")
            st.info("سيتم توجيهك إلى تطبيق المقابلة قريباً")
    
    with col3:
        st.markdown("""
        <div class="app-card">
            <h3>📊 تطبيق إضافي</h3>
            <p>تطبيق ثالث للتقييم أو المراجعة أو التدريب الإضافي</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("بدء التطبيق الثالث", use_container_width=True, type="primary"):
            # Trigger n8n workflow
            trigger_n8n_workflow("start_third_app", {
                'student_id': student['id'],
                'app_type': 'third',
                'track': student['track']
            })
            
            st.success("يتم تحضير التطبيق...")
            st.info("سيتم توجيهك إلى التطبيق قريباً")
    
    # Student's previous results
    st.header("📈 نتائجك السابقة")
    student_results = get_student_results(student['id'])
    
    if not student_results.empty:
        for _, result in student_results.iterrows():
            with st.expander(f"{result['app_type']} - {result['completed_at']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("الدرجة", f"{result['score']:.1%}" if result['score'] else "غير متوفر")
                with col2:
                    st.metric("مستوى القدرة", f"{result['ability_level']:.1%}" if result['ability_level'] else "غير متوفر")
                with col3:
                    st.metric("الأسئلة المجابة", result['questions_answered'] or 0)
                
                if result['details']:
                    st.text_area("التفاصيل", result['details'], disabled=True)
    else:
        st.info("لم تقم بأي اختبارات بعد. ابدأ اختبارك الأول!")

else:
    # Welcome page for non-logged users
    st.header("🎯 مرحباً بك في نظام التقييم الذكي")
    
    st.markdown("""
    ### 🌟 ما يميز نظامنا:
    
    - **🎯 اختبار تكيفي ذكي**: يتكيف مع مستواك ويقدم أسئلة مناسبة لقدراتك
    - **🤖 مقابلة بالذكاء الاصطناعي**: تجربة مقابلة واقعية مع ردود فعل فورية
    - **📊 تحليل شامل**: تقارير مفصلة عن أدائك ونقاط القوة والضعف
    - **🎪 متعدد التخصصات**: يغطي مختلف المجالات التقنية
    
    ### 🚀 لبدء رحلتك:
    
    1. سجل دخولك أو أنشئ حساب جديد من الشريط الجانبي
    2. اختر نوع التقييم الذي تريده
    3. ابدأ واحصل على تقييم فوري لمهاراتك
    
    ---
    
    📧 للحصول على الدعم، تواصل معنا على: support@assessment-system.com
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🎓 نظام التقييم الذكي المتكامل | تطوير فريق التطوير التعليمي</p>
    <p>⚡ مدعوم بالذكاء الاصطناعي | 🔐 آمن ومحمي</p>
</div>
""", unsafe_allow_html=True)