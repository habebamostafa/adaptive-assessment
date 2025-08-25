"""
AI Interview Coach – full open‑source Streamlit app (voice ↔ text ↔ evaluation ↔ voice)
--------------------------------------------------------------------------------------

How it works (MVP):
1) User records an answer by voice (or uploads audio).  
2) Speech‑to‑Text using faster‑whisper (open‑source, multilingual → Arabic/English supported).  
3) NLP evaluation:
   - Semantic similarity vs a reference answer using a multilingual Sentence‑Transformer.  
   - Heuristic rubric scores: Structure (STAR), Relevance, Depth/Impact, Clarity/Brevity.  
4) Generates textual feedback + an improved STAR‑formatted answer (rule‑based templating).  
5) Text‑to‑Speech with pyttsx3 (offline, open‑source) → plays feedback audio + lets you download.

Notes:
- This MVP avoids any closed APIs. All models are open‑source.
- For production accuracy, you can later swap/augment with your preferred models.

Quick start (locally):
----------------------
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
pip install -U pip
pip install streamlit streamlit-audiorec streamlit-extras faster-whisper sentence-transformers numpy librosa soundfile pyttsx3 pydub language-tool-python scikit-learn

# (Optional) On Linux you might need extra audio backends for pyttsx3 (espeak, aplay):
#   sudo apt-get install espeak ffmpeg libespeak1

# Run
streamlit run app.py

File layout:
- Single file (this one) is app.py

"""
from __future__ import annotations
import os
import io
import re
import time
import json
import queue
import tempfile
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import librosa

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

# Voice recording (browser) – lightweight open source component
# If this fails in your env, user can upload a WAV/MP3 file instead.
from audiorecorder import audiorecorder  # provided by `streamlit-audiorec`

# Speech‑to‑Text (open source, multilingual)
from faster_whisper import WhisperModel

# Semantic similarity (open source, multilingual)
from sentence_transformers import SentenceTransformer, util

# Offline TTS (open source)
import pyttsx3
from pydub import AudioSegment

# -----------------------------
# Config & small in‑code datasets
# -----------------------------
st.set_page_config(page_title="AI Interview Coach (Open Source)", page_icon="🎤", layout="wide")

ROLES = {
    "Software Engineer (Intern/Junior)": {
        "behavioral": [
            "Tell me about a time you debugged a tough issue.",
            "Describe a project where you collaborated with a team to deliver under a deadline.",
            "When you disagreed with a teammate, how did you handle it?",
        ],
        "technical": [
            "Explain Big O of your favorite data structure and when to use it.",
            "How would you design a simple URL shortener?",
        ],
        "reference_answers": {
            "Tell me about a time you debugged a tough issue.": (
                "In my OS course project (Situation), our server randomly crashed under load (Task). "
                "I added logging, reproduced with a load test, and used a binary search over commits to isolate a race condition (Action). "
                "Fixing it reduced crash rate from 20% to 0% and improved latency by 30% (Result)."
            )
        },
    },
    "Data Analyst (Entry)": {
        "behavioral": [
            "Tell me about a time you used data to influence a decision.",
            "Describe how you prioritize conflicting requests.",
        ],
        "technical": [
            "How do you handle missing data?",
            "What is the difference between inner and left join?",
        ],
        "reference_answers": {
            "Tell me about a time you used data to influence a decision.": (
                "At a student club (Situation), event attendance dropped (Task). "
                "I analyzed survey data and ran A/B tests on timing (Action). "
                "We switched to weekend slots and increased attendance by 45% (Result)."
            )
        },
    },
    "AI/ML Engineer (Junior)": {
        "behavioral": [
            "Tell me about an ML project you delivered end‑to‑end.",
            "How do you deal with ambiguous problem statements?",
        ],
        "technical": [
            "Explain bias‑variance tradeoff.",
            "What metrics would you use for imbalanced classification?",
        ],
        "reference_answers": {
            "Tell me about an ML project you delivered end‑to‑end.": (
                "For a capstone (Situation), we needed to predict churn (Task). "
                "I led data prep, trained XGBoost, did SHAP for interpretability (Action). "
                "AUROC improved from 0.68 to 0.84; a simple rule‑based campaign reduced churn by 12% (Result)."
            )
        },
    },
}

LANGS = ["auto", "en", "ar"]
DEFAULT_MODEL_SIZE = os.environ.get("WHISPER_SIZE", "small")  # small/base/medium etc.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# -----------------------------
# Utility: Audio & Files
# -----------------------------
def save_bytes_to_wav(data: bytes, path: str, sr: int = 16000):
    with io.BytesIO(data) as bio:
        y, _ = librosa.load(bio, sr=sr, mono=True)
        sf.write(path, y, sr)


def ensure_wav(file_bytes: bytes, tmpdir: str) -> str:
    """Accepts raw bytes (possibly webm/mp3/wav), returns a WAV path."""
    in_path = os.path.join(tmpdir, "input_audio")
    with open(in_path, "wb") as f:
        f.write(file_bytes)
    # Try decode with pydub (ffmpeg required)
    audio = AudioSegment.from_file(in_path)
    wav_path = os.path.join(tmpdir, "input.wav")
    audio.set_frame_rate(16000).set_channels(1).export(wav_path, format="wav")
    return wav_path


# -----------------------------
# Speech‑to‑Text
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str = DEFAULT_MODEL_SIZE):
    model = WhisperModel(model_size, device="auto", compute_type="auto")
    return model


def transcribe_audio(wav_path: str, lang: str = "auto") -> dict:
    model = load_whisper()
    segments, info = model.transcribe(wav_path, language=None if lang == "auto" else lang, vad_filter=True)
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text.strip())
    text = " ".join(text_parts).strip()
    return {"text": text, "language": info.language}


# -----------------------------
# NLP evaluation
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedder(name: str = EMBED_MODEL):
    return SentenceTransformer(name)


def rubric_scores(answer: str, question: str, ref_answer: str | None) -> dict:
    embedder = load_embedder()
    dims = {
        "Structure_STAR": 0.25,
        "Relevance": 0.25,
        "Depth_Impact": 0.25,
        "Clarity_Brevity": 0.15,
        "Confidence_Tone": 0.10,
    }

    # Relevance via semantic similarity (answer vs question + ref)
    sim_q = float(util.cos_sim(embedder.encode([answer], convert_to_tensor=True),
                               embedder.encode([question], convert_to_tensor=True))[0][0])
    sim_r = 0.0
    if ref_answer:
        sim_r = float(util.cos_sim(embedder.encode([answer], convert_to_tensor=True),
                                   embedder.encode([ref_answer], convert_to_tensor=True))[0][0])
    relevance = max(0.0, (sim_q * 0.4 + sim_r * 0.6))

    # Structure (STAR) heuristic: count presence of markers/temporal connectors/metrics
    star_cues = [r"situation|context|challenge|في موقف|موقف|تحدي",
                 r"task|goal|هدفي|مهمتي",
                 r"action|خطوات|اتخذت|نفذت|عملت",
                 r"result|الأثر|النتيجة|حققنا|٪|%|increased|reduced|improved|\d+%|\d+"]
    star_hits = sum(1 for pat in star_cues if re.search(pat, answer, flags=re.I))
    structure = star_hits / len(star_cues)

    # Depth/Impact: numbers, metrics, ownership verbs
    impact_cues = [r"\d+%|\d+|latency|accuracy|AUROC|users|مبيعات|إيرادات|نسبة|عدد", r"I |led|own|قدت|مسؤول"]
    impact_hits = sum(1 for pat in impact_cues if re.search(pat, answer, flags=re.I))
    depth = min(1.0, 0.5 + 0.25 * impact_hits)

    # Clarity/Brevity: penalize overlong rambling; encourage 90–180 seconds (~180–350 words)
    wc = len(answer.split())
    if wc == 0:
        clarity = 0.0
    elif wc < 50:
        clarity = 0.5
    elif wc <= 350:
        clarity = 1.0
    else:
        clarity = max(0.2, 1.0 - (wc - 350) / 500)

    # Confidence/Tone: cue words (positive/ownership) – very rough
    tone_cues = [r"confident|proud|learned|تحسّن|تعلمت|مسؤول|واجهت بنجاح|initiative|ownership"]
    tone = 0.6 if any(re.search(p, answer, flags=re.I) for p in tone_cues) else 0.4

    # Map to 1–4 per dimension
    def to_1_4(x):
        if x >= 0.85: return 4
        if x >= 0.65: return 3
        if x >= 0.45: return 2
        return 1

    scores_0_1 = {
        "Structure_STAR": structure,
        "Relevance": max(0.0, min(1.0, relevance)),
        "Depth_Impact": depth,
        "Clarity_Brevity": clarity,
        "Confidence_Tone": tone,
    }
    scores_1_4 = {k: to_1_4(v) for k, v in scores_0_1.items()}

    overall = int(sum(scores_1_4[k] * w for k, w in dims.items()) / 4 * 100)
    return {"scores": scores_1_4, "overall": overall, "aux": scores_0_1}


def improve_answer_star(answer: str) -> str:
    # Very lightweight STAR rewriter (rule‑based). For production, swap with an LLM.
    sents = re.split(r"(?<=[.!؟\?])\s+", answer.strip())
    sents = [s for s in sents if s]
    if not sents:
        return ""
    # Heuristic grouping
    situation = sents[0]
    task = next((s for s in sents[1:] if re.search(r"task|هدف|مهمتي|كان مطلوب", s, re.I)), "")
    action = " ".join(s for s in sents if re.search(r"I |we |اتخذ|نفذت|عملت|حللت|اختبرت|بنيت", s, re.I))
    result_sents = [s for s in sents if re.search(r"%|نتيجة|أثر|زادت|انخفضت|تحسن|increased|reduced|improved|saved", s, re.I)]
    result = result_sents[-1] if result_sents else "I measured the outcome and documented key lessons learned."

    tmpl = (
        "Situation: {s}\n"
        "Task: {t}\n"
        "Action: {a}\n"
        "Result: {r}\n"
        "Lessons: Focus on metrics, your unique contribution, and what you'd do differently next time."
    )
    return tmpl.format(s=situation, t=task or "Clarify the goal and constraints.", a=action or "Explain 2–3 concrete steps you took.", r=result)


# -----------------------------
# TTS
# -----------------------------
@st.cache_resource(show_spinner=False)
def init_tts():
    engine = pyttsx3.init()
    # Try a slightly faster rate; keep natural
    rate = engine.getProperty('rate')
    engine.setProperty('rate', min(200, int(rate * 1.05)))
    return engine


def synthesize_feedback_tts(text: str, out_path: str):
    engine = init_tts()
    engine.save_to_file(text, out_path)
    engine.runAndWait()


# -----------------------------
# UI
# -----------------------------
st.title("🎤 AI Interview Coach – Open Source")
st.caption("Voice → Text → Evaluation → Voice. Multilingual (Arabic/English). No closed APIs.")

col1, col2 = st.columns([1.2, 1])
with col1:
    role = st.selectbox("Target Role", list(ROLES.keys()))
    cat = st.radio("Question Type", ["behavioral", "technical"], horizontal=True)
    q_options = ROLES[role][cat]
    question = st.selectbox("Question", q_options)
    st.info(question)

    add_vertical_space(1)
    st.markdown("**Record your answer** (or upload an audio file):")
    audio = audiorecorder("Start Recording", "Stop Recording")
    uploaded = st.file_uploader("...or upload audio (wav/mp3/m4a/webm)", type=["wav", "mp3", "m4a", "webm"], accept_multiple_files=False)

    with st.expander("Optional: Language hint for ASR"):
        lang = st.selectbox("Language", LANGS, index=0, help="auto = let Whisper detect language")

    do_eval = st.button("⬆️ Transcribe & Evaluate", type="primary")

with col2:
    st.subheader("Rubric (1–4)")
    st.markdown("Structure (STAR), Relevance, Depth/Impact, Clarity/Brevity, Confidence/Tone")
    placeholder_scores = st.empty()
    placeholder_overall = st.empty()

add_vertical_space(1)
res_container = st.container()

if do_eval:
    with st.spinner("Processing audio ..."):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = None
            if audio and len(audio) > 0:
                # audiorecorder returns pydub AudioSegment under the hood → export to wav
                wav_path = os.path.join(tmp, "rec.wav")
                audio.export(wav_path, format="wav")
            elif uploaded is not None:
                wav_path = ensure_wav(uploaded.read(), tmp)
            else:
                st.warning("Please record or upload audio first.")

            if wav_path:
                asr = transcribe_audio(wav_path, lang)
                transcript = asr["text"].strip()
                detected_lang = asr.get("language", "auto")

                ref_answer = ROLES[role]["reference_answers"].get(question, None)
                eval_out = rubric_scores(transcript, question, ref_answer)
                improved = improve_answer_star(transcript)

                with res_container:
                    st.markdown("### Transcript")
                    st.write(transcript if transcript else "(No speech detected)")

                    st.markdown("### Scores")
                    c1, c2 = st.columns(2)
                    with c1:
                        for k, v in eval_out["scores"].items():
                            st.slider(k, min_value=0, max_value=4, value=int(v), disabled=True)
                    with c2:
                        st.metric("Overall", f"{eval_out['overall']} / 100")
                        st.caption(f"ASR detected language: {detected_lang}")

                    st.markdown("### Feedback & Improved STAR Answer")
                    bullet_strengths = []
                    if eval_out["scores"]["Relevance"] >= 3:
                        bullet_strengths.append("Directly relevant to the question.")
                    if eval_out["scores"]["Structure_STAR"] >= 3:
                        bullet_strengths.append("Clear structure.")
                    if eval_out["scores"]["Depth_Impact"] >= 3:
                        bullet_strengths.append("Good use of metrics/impact.")
                    if not bullet_strengths:
                        bullet_strengths.append("Clear key message.")

                    gaps = []
                    if eval_out["scores"]["Depth_Impact"] <= 2:
                        gaps.append("Add concrete numbers and measurable outcomes.")
                    if eval_out["scores"]["Structure_STAR"] <= 2:
                        gaps.append("Follow the STAR flow explicitly.")
                    if eval_out["scores"]["Clarity_Brevity"] <= 2:
                        gaps.append("Be more concise; aim for 90–180 seconds.")

                    st.write("**Strengths:**")
                    for s in bullet_strengths:
                        st.write("- ", s)
                    st.write("**Gaps:**")
                    for g in gaps:
                        st.write("- ", g)

                    st.code(improved, language="markdown")

                    # Synthesize voice feedback
                    feedback_text = (
                        f"Overall score {eval_out['overall']} out of 100. "
                        f"Strengths: {', '.join(bullet_strengths)}. "
                        f"Gaps: {', '.join(gaps)}. "
                        "Try the improved STAR structure provided."
                    )
                    fb_path = os.path.join(tmp, "feedback.wav")
                    synthesize_feedback_tts(feedback_text, fb_path)
                    audio_bytes = open(fb_path, "rb").read()

                    st.markdown("### Audio Feedback")
                    st.audio(audio_bytes, format="audio/wav")

                    # Offer downloads
                    st.download_button("Download Transcript", transcript.encode("utf-8"), file_name="transcript.txt")
                    st.download_button("Download Feedback Audio", data=audio_bytes, file_name="feedback.wav")

                    # Minimal session report (JSON)
                    report = {
                        "role": role,
                        "question": question,
                        "transcript": transcript,
                        "scores": eval_out["scores"],
                        "overall": eval_out["overall"],
                        "improved_answer": improved,
                        "language": detected_lang,
                    }
                    st.download_button("Download JSON Report", json.dumps(report, ensure_ascii=False, indent=2), file_name="interview_report.json")

st.divider()
st.markdown(
    "**Tips**: For best accuracy, speak clearly in a quiet room. Add 1–2 metrics (%, time saved, users) to boost your Depth/Impact score."
)
