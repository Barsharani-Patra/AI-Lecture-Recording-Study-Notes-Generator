"""AI Lecture Recording and Study Notes Generator.

Purpose
-------
This Streamlit application helps turn lecture recordings or existing transcripts
into reviewable study materials. It combines Whisper speech recognition with
Gemini text generation and presents the resulting transcript and materials in a
single interactive interface.

Workflow
--------
1. Provide lecture content by recording from a microphone, uploading an audio
   file, or pasting transcript text.
2. For audio, transcribe it with a selectable Whisper model and language.
   Pasted text skips transcription.
3. Generate study notes from the transcript with Gemini. Choose a note style
   (detailed, Cornell, concise bullets, or exam revision) and detail level.
4. Review the transcript and notes, then optionally generate a glossary, exam
   questions, action items, flashcards, a quiz, or transcript-based answers.
5. Download the study pack or transcript, or save the files locally.

Inputs and features
-------------------
Audio uploads accept MP3, WAV, M4A, MP4, OGG, and FLAC files. The language
selector supports automatic detection, English, Hindi, Spanish, French, German,
Arabic, Chinese, and Japanese. Long transcripts are limited to 150,000
characters when sent for generation.

Exports
-------
The study pack can be exported as Markdown, plain text, Word, PDF, or
PowerPoint. A timestamped transcript is available as a separate text file.
Using the local-save button writes output files to the `exports` directory
beside this application.

Configuration and data handling
-------------------------------
Enter a Gemini API key in the sidebar or provide it through the GEMINI_API_KEY
environment variable. FFmpeg must be installed and available to Whisper for
audio decoding. Audio is written to a temporary file for transcription and
removed afterward. Active transcripts, generated materials, and session
history are held in Streamlit session state rather than a persistent database.
Transcript text is sent to Gemini for note generation, so only submit content
you are authorized to process.

Running the app
---------------
Activate the project virtual environment, change to this file's directory, and
run `streamlit run app.py`.
"""

import io
import json
import os
import re
import tempfile
from datetime import datetime

import streamlit as st
import whisper
from google import genai
from google.genai import types

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Lecture Notes Generator", page_icon="🎓", layout="wide")

MODEL = "gemini-3.5-flash"  # change if Google renames it
MAX_CHARS = 150_000  # safety limit for very long transcripts

LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Chinese": "zh",
    "Japanese": "ja",
}

NOTE_STYLES = {
    "Detailed structured notes": (
        "Create detailed, well-organised study notes. Use markdown headings and "
        "sub-headings for each topic, bullet points for key ideas, bold for important "
        "terms, short examples where the lecturer gave them, and finish with a "
        "'Summary' section of 4-6 lines."
    ),
    "Cornell notes": (
        "Create Cornell-style notes. Use three sections: 'Cues / Questions' (a list of "
        "questions a student should be able to answer), 'Notes' (organised main content "
        "with headings and bullets), and 'Summary' (a short paragraph)."
    ),
    "Concise bullet summary": (
        "Create a short, high-level bullet summary. Maximum 15 bullets, each one line, "
        "grouped under 2-4 headings."
    ),
    "Exam revision sheet": (
        "Create a one-page exam revision sheet: key definitions, formulas or rules, "
        "important points, common mistakes, and a quick checklist of what to remember."
    ),
}

DETAIL_LEVELS = {
    1: "Keep it brief and to the point.",
    2: "Use a balanced level of detail.",
    3: "Be thorough and include as much detail from the lecture as possible.",
}

SYSTEM_PROMPT = (
    "You are an expert teaching assistant who turns lecture transcripts into clear, "
    "accurate study material. Use only information from the transcript; if something "
    "is unclear or seems mis-transcribed, say so briefly rather than guessing."
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
# Keep the active lecture and generated study materials available across reruns.
DEFAULTS = {
    "transcript": "",
    "timestamped": "",
    "segments": [],
    "duration": 0.0,
    "detected_lang": "",
    "recording_widget_version": 0,
    "notes": "",
    "extras": {},       # glossary, exam questions, action items
    "flashcards": [],
    "quiz": [],
    "qa": [],           # list of (question, answer)
    "history": [],      # past lectures in this session
    "title": "",
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def fmt_time(seconds: float) -> str:
    """Format a duration as MM:SS or HH:MM:SS."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@st.cache_resource(show_spinner=False)
def load_whisper(size: str):
    """Load each Whisper model once and reuse it across Streamlit reruns."""
    return whisper.load_model(size)


def get_api_key() -> str:
    """Read the Gemini key from the UI first, then the environment."""
    return st.session_state.get("api_key") or os.getenv("GEMINI_API_KEY", "")


def ask_gemini(prompt: str, max_tokens: int = 3000) -> str:
    """Send a prompt and the app's system instruction to the configured Gemini model."""
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text


def extract_json(text: str):
    """Pull a JSON array/object out of a model reply, even if wrapped in fences."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    starts = [i for i in (text.find("["), text.find("{")) if i != -1]
    if not starts:
        raise ValueError("No JSON found in response")
    start = min(starts)
    end = max(text.rfind("]"), text.rfind("}")) + 1
    return json.loads(text[start:end])


def transcript_for_prompt() -> str:
    """Limit transcript length before sending it to the Gemini API."""
    text = st.session_state.transcript
    if len(text) > MAX_CHARS:
        st.warning("Transcript is very long, so only the first part is used for generation.")
        text = text[:MAX_CHARS]
    return text


def transcribe_audio(audio_file, model_size: str, language: str | None):
    """Transcribe uploaded audio and remove its temporary file even on failure."""
    suffix = os.path.splitext(getattr(audio_file, "name", "") or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.getvalue())
        path = tmp.name
    try:
        model = load_whisper(model_size)
        options = {"fp16": False}
        if language:
            options["language"] = language
        return model.transcribe(path, **options)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def reset_outputs():
    """Clear generated materials before processing a different lecture."""
    for key in ("notes", "extras", "flashcards", "quiz", "qa"):
        st.session_state[key] = DEFAULTS[key].copy() if hasattr(DEFAULTS[key], "copy") else DEFAULTS[key]


def clear_recording():
    """Change the recorder widget key so Streamlit creates a fresh recorder."""
    st.session_state["recording_widget_version"] += 1


def build_markdown() -> str:
    """Combine the current notes and optional study aids into one Markdown pack."""
    s = st.session_state
    title = s.title or "Lecture Study Pack"
    parts = [f"# {title}", f"_Generated on {datetime.now():%d %b %Y, %H:%M}_", ""]
    if s.notes:
        parts += ["## Study Notes", s.notes, ""]
    for name, content in s.extras.items():
        parts += [f"## {name}", content, ""]
    if s.flashcards:
        parts.append("## Flashcards")
        for i, card in enumerate(s.flashcards, 1):
            parts.append(f"**{i}. Q:** {card['question']}  \n**A:** {card['answer']}")
        parts.append("")
    if s.quiz:
        parts.append("## Practice Quiz")
        for i, q in enumerate(s.quiz, 1):
            parts.append(f"**{i}. {q['question']}**")
            for letter, option in zip("ABCD", q["options"]):
                parts.append(f"- {letter}. {option}")
            parts.append(f"\n_Answer: {q['answer']}. {q.get('explanation', '')}_\n")
    if s.qa:
        parts.append("## Questions & Answers")
        for q, a in s.qa:
            parts.append(f"**Q:** {q}\n\n{a}\n")
    return "\n".join(parts)


def build_docx(markdown_text: str):
    """Convert Markdown headings and bullets into a downloadable Word document."""
    try:
        from docx import Document
    except ImportError:
        return None
    doc = Document()
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        clean = stripped.replace("**", "").replace("_", "")
        if stripped.startswith("# "):
            doc.add_heading(clean[2:], 0)
        elif stripped.startswith("## "):
            doc.add_heading(clean[3:], 1)
        elif stripped.startswith("### "):
            doc.add_heading(clean[4:], 2)
        elif stripped.startswith(("- ", "* ")):
            doc.add_paragraph(clean[2:], style="List Bullet")
        else:
            doc.add_paragraph(clean)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def build_plain_text(markdown_text: str) -> str:
    """Remove basic Markdown syntax to produce a readable plain-text version."""
    lines = []
    for line in markdown_text.splitlines():
        clean = re.sub(r"^#{1,6}\s*", "", line.strip())
        clean = re.sub(r"\*\*(.*?)\*\*", r"\1", clean)
        clean = re.sub(r"(?<!\w)_(.*?)_(?!\w)", r"\1", clean)
        clean = re.sub(r"`([^`]*)`", r"\1", clean)
        if clean.startswith(("- ", "* ")):
            clean = f"• {clean[2:]}"
        lines.append(clean)
    return "\n".join(lines).strip() + "\n"


def build_pdf(markdown_text: str) -> bytes:
    """Render the study pack as a PDF document and return its bytes."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    from xml.sax.saxutils import escape

    font_name = "Helvetica"
    if os.name == "nt":
        font_path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf")
        if os.path.exists(font_path):
            font_name = "LectureNotesArial"
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, font_path))

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("LectureBody", parent=styles["BodyText"], fontName=font_name)
    heading_style = ParagraphStyle(
        "LectureHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        spaceBefore=10,
        spaceAfter=5,
    )
    title_style = ParagraphStyle(
        "LectureTitle",
        parent=styles["Title"],
        fontName=font_name,
        spaceAfter=16,
    )

    elements = []
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if not stripped:
            elements.append(Spacer(1, 5))
            continue
        plain_line = build_plain_text(stripped).strip()
        escaped_line = escape(plain_line)
        if stripped.startswith("# "):
            elements.append(Paragraph(escaped_line, title_style))
        elif stripped.startswith(("## ", "### ")):
            elements.append(Paragraph(escaped_line, heading_style))
        else:
            elements.append(Paragraph(escaped_line, body_style))

    document.build(elements)
    return buffer.getvalue()


def build_pptx(markdown_text: str) -> bytes:
    """Turn Markdown sections into a title slide and grouped content slides."""
    from pptx import Presentation
    from pptx.enum.text import MSO_AUTO_SIZE
    from pptx.util import Inches, Pt

    title = "Lecture Study Pack"
    sections = []
    section_title = "Overview"
    section_items = []

    def add_section():
        if section_items:
            sections.append((section_title, section_items.copy()))
            section_items.clear()

    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = build_plain_text(stripped[2:]).strip()
        elif stripped.startswith("## "):
            add_section()
            section_title = build_plain_text(stripped[3:]).strip()
        elif stripped:
            is_bullet = stripped.startswith(("- ", "* "))
            clean = build_plain_text(stripped).strip()
            if is_bullet and clean.startswith("• "):
                clean = clean[2:]
            section_items.append((clean, is_bullet))
    add_section()

    presentation = Presentation()
    title_slide = presentation.slides.add_slide(presentation.slide_layouts[0])
    title_slide.shapes.title.text = title
    title_slide.placeholders[1].text = "Lecture study pack"

    for heading, items in sections:
        for start in range(0, len(items), 6):
            slide = presentation.slides.add_slide(presentation.slide_layouts[6])
            title_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(9), Inches(0.7))
            title_frame = title_box.text_frame
            title_frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
            title_paragraph = title_frame.paragraphs[0]
            title_paragraph.text = heading if start == 0 else f"{heading} (continued)"
            title_paragraph.font.size = Pt(26)
            title_paragraph.font.bold = True

            body_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.25), Inches(8.4), Inches(5.7))
            body_frame = body_box.text_frame
            body_frame.word_wrap = True
            body_frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
            for index, (text, is_bullet) in enumerate(items[start:start + 6]):
                paragraph = body_frame.paragraphs[0] if index == 0 else body_frame.add_paragraph()
                paragraph.text = f"• {text}" if is_bullet else text
                paragraph.font.size = Pt(18)

    buffer = io.BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


def save_export_files(
    export_dir: str,
    base: str,
    study_pack: str,
    plain_text: str,
    transcript: str,
    docx_bytes: bytes | None,
    pdf_bytes: bytes,
    pptx_bytes: bytes,
) -> list[str]:
    """Save available study-pack formats and the transcript into the export folder."""
    os.makedirs(export_dir, exist_ok=True)
    files = [
        (f"{base}_notes.md", study_pack.encode("utf-8")),
        (f"{base}_notes.txt", plain_text.encode("utf-8")),
        (f"{base}_transcript.txt", transcript.encode("utf-8")),
        (f"{base}_notes.pdf", pdf_bytes),
        (f"{base}_notes.pptx", pptx_bytes),
    ]
    if docx_bytes:
        files.append((f"{base}_notes.docx", docx_bytes))

    paths = []
    for filename, contents in files:
        path = os.path.join(export_dir, filename)
        with open(path, "wb") as output_file:
            output_file.write(contents)
        paths.append(path)
    return paths


# ----------------------------------------------------------------------------
# Sidebar settings
# ----------------------------------------------------------------------------
# Collect credentials and preferences that control transcription and note generation.
with st.sidebar:
    st.header("⚙️ Settings")
    st.text_input("Gemini API key", type="password", key="api_key",
                  help="Or set the GEMINI_API_KEY environment variable.")
    model_size = st.selectbox(
        "Whisper model", ["tiny", "base", "small", "medium"], index=1,
        help="Bigger = more accurate but slower. 'base' is a good start.",
    )
    language_name = st.selectbox("Lecture language", list(LANGUAGES.keys()))
    st.divider()
    st.subheader("📝 Notes options")
    subject = st.text_input("Subject / topic (optional)", placeholder="e.g. Operating Systems")
    note_style = st.selectbox("Note style", list(NOTE_STYLES.keys()))
    detail = st.slider("Detail level", 1, 3, 2)
    st.divider()
    if st.session_state.history:
        st.subheader("🕘 This session")
        for item in reversed(st.session_state.history[-5:]):
            st.caption(f"{item['time']} · {item['title']} · {item['words']} words")

# ----------------------------------------------------------------------------
# Header + input
# ----------------------------------------------------------------------------
# Let the learner choose a recording, audio upload, or an existing transcript.
st.title("🎓 AI Lecture Recording & Study Notes Generator")
st.caption("Record or upload a lecture, then get a transcript, study notes, flashcards, a quiz, and more.")

input_tab1, input_tab2, input_tab3 = st.tabs(["🎙️ Record", "📁 Upload audio", "📋 Paste transcript"])

audio = None
pasted = ""
with input_tab1:
    recorded = st.audio_input(
        "Record the lecture",
        key=f"recording_{st.session_state['recording_widget_version']}",
    )
    if recorded:
        audio = recorded
    st.button("Clear recording", key="clear_recording", on_click=clear_recording)
with input_tab2:
    uploaded = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "mp4", "ogg", "flac"])
    if uploaded:
        audio = uploaded
        st.audio(uploaded)
with input_tab3:
    pasted = st.text_area("Already have text? Paste it here to skip transcription", height=150)

lecture_title = st.text_input("Lecture title", value=subject or "", placeholder="e.g. Lecture 5: Deadlocks")

# Process the chosen input, generate the main notes, and record session history.
if st.button("🚀 Process lecture", type="primary", use_container_width=True):
    if not get_api_key():
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not audio and not pasted.strip():
        st.error("Please record, upload, or paste something first.")
    else:
        reset_outputs()
        if pasted.strip() and not audio:
            st.session_state.transcript = pasted.strip()
            st.session_state.timestamped = pasted.strip()
            st.session_state.segments = []
            st.session_state.duration = 0.0
            st.session_state.detected_lang = ""
        else:
            with st.spinner(f"Transcribing with Whisper ({model_size})... this can take a while."):
                try:
                    result = transcribe_audio(audio, model_size, LANGUAGES[language_name])
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    st.stop()
            segments = result.get("segments", [])
            st.session_state.transcript = result["text"].strip()
            st.session_state.segments = segments
            st.session_state.timestamped = "\n".join(
                f"[{fmt_time(seg['start'])}] {seg['text'].strip()}" for seg in segments
            )
            st.session_state.duration = segments[-1]["end"] if segments else 0.0
            st.session_state.detected_lang = result.get("language", "")

        st.session_state.title = lecture_title or subject or "Untitled lecture"

        if not st.session_state.transcript:
            st.error("No speech was detected. Try a clearer or longer recording.")
            st.stop()

        with st.spinner("Writing study notes..."):
            prompt = (
                f"Subject: {subject or 'not specified'}\n"
                f"Task: {NOTE_STYLES[note_style]}\n"
                f"Detail: {DETAIL_LEVELS[detail]}\n\n"
                f"Transcript:\n{transcript_for_prompt()}"
            )
            try:
                st.session_state.notes = ask_gemini(prompt, max_tokens=4000)
            except Exception as e:
                st.error(f"Note generation failed: {e}")
                st.stop()

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M"),
            "title": st.session_state.title,
            "words": len(st.session_state.transcript.split()),
        })
        st.success("Done! Explore the tabs below.")

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
# Show summary metrics and create tabs for reviewing and exporting the study pack.
if st.session_state.transcript:
    words = len(st.session_state.transcript.split())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Words", f"{words:,}")
    c2.metric("Duration", fmt_time(st.session_state.duration) if st.session_state.duration else "n/a")
    c3.metric("Reading time", f"{max(1, round(words / 200))} min")
    c4.metric("Language", st.session_state.detected_lang or "n/a")

    tab_notes, tab_transcript, tab_cards, tab_quiz, tab_ask, tab_export = st.tabs(
        ["📝 Notes", "📜 Transcript", "🃏 Flashcards", "❓ Quiz", "💬 Ask", "📥 Export"]
    )

    # ---------------- Notes + extras ----------------
    # Display the main notes and generate optional resources from the same transcript.
    with tab_notes:
        st.markdown(st.session_state.notes)
        st.divider()
        st.subheader("Extras")
        e1, e2, e3 = st.columns(3)
        extra_prompts = {
            "Key Terms Glossary": (
                "Create a glossary of the key terms from this lecture. Format: "
                "'**Term**: one-line definition'. 8-20 terms."
            ),
            "Likely Exam Questions": (
                "Write 8 likely exam questions (mix of short-answer and long-answer) "
                "based on this lecture, each followed by a 2-3 line model answer."
            ),
            "Action Items & Further Reading": (
                "List any homework, deadlines, or action items mentioned. Then suggest "
                "5 topics the student should read further on, with a one-line reason each."
            ),
        }
        for col, name in zip((e1, e2, e3), extra_prompts):
            if col.button(f"Generate: {name}", use_container_width=True):
                with st.spinner(f"Generating {name.lower()}..."):
                    try:
                        st.session_state.extras[name] = ask_gemini(
                            f"{extra_prompts[name]}\n\nTranscript:\n{transcript_for_prompt()}"
                        )
                    except Exception as e:
                        st.error(f"Failed: {e}")
        for name, content in st.session_state.extras.items():
            with st.expander(name, expanded=True):
                st.markdown(content)

    # ---------------- Transcript ----------------
    # Allow timestamped review and simple text search.
    with tab_transcript:
        show_ts = st.toggle("Show timestamps", value=bool(st.session_state.segments))
        text = st.session_state.timestamped if show_ts else st.session_state.transcript
        st.text_area("Transcript", text, height=400)
        search = st.text_input("🔍 Search transcript")
        if search:
            hits = [line for line in st.session_state.timestamped.splitlines()
                    if search.lower() in line.lower()]
            st.write(f"{len(hits)} match(es)")
            for line in hits[:50]:
                st.markdown(f"- {line}")

    # ---------------- Flashcards ----------------
    # Generate and display question-answer cards for recall practice.
    with tab_cards:
        n_cards = st.slider("Number of flashcards", 5, 25, 10)
        if st.button("Generate flashcards"):
            with st.spinner("Creating flashcards..."):
                prompt = (
                    f"Create {n_cards} flashcards from this lecture. Return ONLY a JSON "
                    'array like [{"question": "...", "answer": "..."}]. No other text.\n\n'
                    f"Transcript:\n{transcript_for_prompt()}"
                )
                try:
                    st.session_state.flashcards = extract_json(ask_gemini(prompt))
                except Exception as e:
                    st.error(f"Could not create flashcards: {e}")
        for i, card in enumerate(st.session_state.flashcards, 1):
            with st.expander(f"Card {i}: {card['question']}"):
                st.write(card["answer"])

    # ---------------- Quiz ----------------
    # Generate multiple-choice questions and score the learner's selections.
    with tab_quiz:
        n_q = st.slider("Number of questions", 3, 15, 5)
        if st.button("Generate quiz"):
            with st.spinner("Creating quiz..."):
                prompt = (
                    f"Create {n_q} multiple-choice questions from this lecture. Return ONLY "
                    'a JSON array like [{"question": "...", "options": ["a","b","c","d"], '
                    '"answer": "A", "explanation": "..."}]. "answer" must be one letter '
                    "A-D. Exactly 4 options each. No other text.\n\n"
                    f"Transcript:\n{transcript_for_prompt()}"
                )
                try:
                    st.session_state.quiz = extract_json(ask_gemini(prompt))
                    for k in list(st.session_state.keys()):
                        if k.startswith("quiz_q"):
                            del st.session_state[k]
                except Exception as e:
                    st.error(f"Could not create quiz: {e}")

        if st.session_state.quiz:
            for i, q in enumerate(st.session_state.quiz):
                st.markdown(f"**{i + 1}. {q['question']}**")
                st.radio(
                    "Choose one", [f"{l}. {o}" for l, o in zip("ABCD", q["options"])],
                    index=None, key=f"quiz_q{i}", label_visibility="collapsed",
                )
            if st.button("Check answers"):
                score = 0
                for i, q in enumerate(st.session_state.quiz):
                    choice = st.session_state.get(f"quiz_q{i}")
                    correct = choice is not None and choice.startswith(q["answer"])
                    score += correct
                    icon = "✅" if correct else "❌"
                    st.markdown(f"{icon} **Q{i + 1}** – Answer: **{q['answer']}**. {q.get('explanation', '')}")
                total = len(st.session_state.quiz)
                st.success(f"Your score: {score}/{total}")

    # ---------------- Ask questions ----------------
    # Answer follow-up questions using the lecture transcript as context.
    with tab_ask:
        st.write("Ask anything about this lecture and get an answer based on the transcript.")
        question = st.text_input("Your question", key="qa_input")
        if st.button("Ask") and question.strip():
            with st.spinner("Thinking..."):
                prompt = (
                    "Answer the student's question using only the lecture transcript. "
                    "If the transcript doesn't cover it, say so.\n\n"
                    f"Question: {question}\n\nTranscript:\n{transcript_for_prompt()}"
                )
                try:
                    st.session_state.qa.append((question, ask_gemini(prompt, max_tokens=1500)))
                except Exception as e:
                    st.error(f"Failed: {e}")
        for q, a in reversed(st.session_state.qa):
            st.markdown(f"**Q: {q}**")
            st.markdown(a)
            st.divider()

    # ---------------- Export ----------------
    # Build downloadable files and save copies locally when requested.
    with tab_export:
        pack = build_markdown()
        plain_text = build_plain_text(pack)
        base = re.sub(r"[^\w\-]+", "_", st.session_state.title or "lecture").strip("_") or "lecture"
        transcript = st.session_state.timestamped or st.session_state.transcript
        docx_bytes = build_docx(pack)
        pdf_bytes = build_pdf(pack)
        pptx_bytes = build_pptx(pack)

        if st.button("💾 Save study pack and transcript locally", use_container_width=True):
            export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
            try:
                saved_paths = save_export_files(
                    export_dir,
                    base,
                    pack,
                    plain_text,
                    transcript,
                    docx_bytes,
                    pdf_bytes,
                    pptx_bytes,
                )
            except OSError as error:
                st.error(f"Could not save the files locally: {error}")
            else:
                st.success(f"Files saved to: {export_dir}")
                for saved_path in saved_paths:
                    st.caption(saved_path)

        st.subheader("Study pack formats")
        st.download_button("⬇️ Study pack (Markdown)", pack, file_name=f"{base}_notes.md",
                           mime="text/markdown", use_container_width=True)
        st.download_button("⬇️ Study pack (Text)", plain_text, file_name=f"{base}_notes.txt",
                           mime="text/plain", use_container_width=True)
        st.download_button("⬇️ Study pack (PDF)", pdf_bytes, file_name=f"{base}_notes.pdf",
                           mime="application/pdf", use_container_width=True)
        st.download_button("⬇️ Study pack (PowerPoint)", pptx_bytes, file_name=f"{base}_notes.pptx",
                           mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                           use_container_width=True)
        if docx_bytes:
            st.download_button(
                "⬇️ Study pack (Word)", docx_bytes, file_name=f"{base}_notes.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        else:
            st.info("Want a Word export too? Run: pip install python-docx")
        st.subheader("Transcript")
        st.download_button("⬇️ Transcript with timestamps (TXT)",
                           transcript,
                           file_name=f"{base}_transcript.txt", use_container_width=True)
else:
    st.info("👆 Record, upload, or paste a lecture and click **Process lecture** to begin.")
