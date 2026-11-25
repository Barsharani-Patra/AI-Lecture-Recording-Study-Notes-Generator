# AI Lecture Recording and Study Notes Generator

A Streamlit application that accepts lecture audio or pasted transcripts, transcribes audio with Whisper, and uses Gemini to generate study notes and revision materials.

## Features

- Record audio, upload MP3/WAV/M4A/MP4/OGG/FLAC, or paste a transcript.
- Transcribe audio with selectable Whisper model sizes and lecture languages.
- Generate detailed, Cornell-style, concise, or exam-revision notes with Gemini.
- Create optional glossaries, exam questions, action items, flashcards, quizzes, and transcript-based Q&A.
- Review and search transcripts, including timestamps.
- Download study packs as Markdown, plain text, Word, PDF, or PowerPoint.
- Save generated files locally under `lecture-notes/exports`.

## Requirements

- Python 3.10 or newer.
- FFmpeg available on `PATH` for Whisper audio decoding.
- A Gemini API key. Enter it in the app sidebar or set `GEMINI_API_KEY`.

## Setup on Windows

From the project root, create the virtual environment and install Python dependencies:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install FFmpeg if needed:

```powershell
winget install --id Gyan.FFmpeg -e
```

Open a new terminal after installing FFmpeg so the updated `PATH` is loaded.

## Run

From the project root:

```powershell
Set-Location .\lecture-notes
..\venv\Scripts\python.exe -m streamlit run app.py
```

Then open `http://localhost:8501`.

## API key and privacy

Do not commit API keys. The app accepts a Gemini API key in its password field or from the `GEMINI_API_KEY` environment variable. Transcript text is sent to the Gemini API when generating notes and study aids. Only process lecture content you are authorized to use.

The temporary audio file used for transcription is removed after processing. Current lecture content is held in Streamlit session state; this project does not include a persistent database or user-account system.

## Project layout

- `lecture-notes/app.py` - Streamlit application.
- `lecture-notes/report/` - Project abstract, synopsis, report, and research paper PDFs.
- `app-launcher/` - Windows URL shortcut for the local app.

The launcher opens the local URL only while the Streamlit server is running.
