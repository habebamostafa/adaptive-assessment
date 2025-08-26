from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import json

app = FastAPI()

class StudentSession(BaseModel):
    student_id: int
    app_type: str
    session_data: dict

@app.post("/api/start_session")
async def start_session(session: StudentSession):
    # إعداد جلسة جديدة
    conn = sqlite3.connect('data/students.db')
    cursor = conn.cursor()
    
    # حفظ بيانات الجلسة
    cursor.execute('''
        INSERT INTO app_sessions (student_id, app_type, session_data, created_at)
        VALUES (?, ?, ?, datetime('now'))
    ''', (session.student_id, session.app_type, json.dumps(session.session_data)))
    
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {"session_id": session_id, "status": "started"}

@app.post("/api/save_results")
async def save_results(results: dict):
    # حفظ نتائج التطبيق
    conn = sqlite3.connect('students.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO assessment_results 
        (student_id, app_type, score, details, completed_at)
        VALUES (?, ?, ?, ?, datetime('now'))
    ''', (
        results['student_id'], 
        results['app_type'], 
        results['score'], 
        json.dumps(results)
    ))
    
    conn.commit()
    conn.close()
    
    return {"status": "saved"}

@app.get("/api/student/{student_id}/context")
async def get_student_context(student_id: int):
    # جلب سياق الطالب
    conn = sqlite3.connect('data/students.db')
    cursor = conn.cursor()
    
    # بيانات الطالب
    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    student = cursor.fetchone()
    
    # النتائج السابقة
    cursor.execute('''
        SELECT app_type, score, ability_level, completed_at 
        FROM assessment_results 
        WHERE student_id = ? 
        ORDER BY completed_at DESC
    ''', (student_id,))
    results = cursor.fetchall()
    
    conn.close()
    
    return {
        "student": student,
        "previous_results": results
    }