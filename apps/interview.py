import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile
import wave
import os
import openai

# --------------------------
# إعدادات OpenAI (لو هتستخدم GPT أو Whisper)
# --------------------------
# openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🎤 Interview Coach with Voice Recording")

# اختيار المدة
duration = st.slider("⏱ اختر مدة التسجيل (ثواني):", 5, 30, 10)

# زر تسجيل
if st.button("🎙️ ابدأ التسجيل"):
    st.write("🔴 جاري التسجيل... اتكلم دلوقتي")

    # تسجيل الصوت
    fs = 16000  # معدل العينة
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()  # استنى التسجيل يخلص

    # حفظ الملف مؤقتاً
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_wav.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())

    st.audio(temp_wav.name, format="audio/wav")
    st.success("✅ التسجيل خلص!")

    # --------------------------
    # هنا ممكن تبعت الملف لـ Whisper أو GPT لتحليل المقابلة
    # --------------------------
    # مثال: إرسال ل OpenAI Whisper (لو عندك API Key)
    """
    with open(temp_wav.name, "rb") as audio_file:
        transcript = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    st.write("📝 النص المحول:", transcript["text"])
    """

    # --------------------------
    # تحليل الأداء (مثال بسيط)
    # --------------------------
    st.subheader("📊 Feedback:")
    st.write("⚡ حاول تتكلم بثقة أكتر.")
    st.write("💡 قلل من استخدام كلمة 'ممم' أو 'يعني'.")
    st.write("👌 لغة جسدك مهمة حتى لو مش ظاهرة هنا.")
