# 🧠 Automated Psychology Facts YouTube Shorts Generator

A fully automated Python system designed to create niche "Psychology Facts" YouTube Shorts (9:16) with zero manual intervention.

## 🚀 Features

- **Automated Topic Selection**: Picks from a built-in library of psychology facts (avoids repeats using `used_topics.json`).
- **Script Generation**: Conversational 45-60s scripts with strong hooks and curiosity loops.
- **Human-like Voiceover**: Uses `gTTS` (Google Text-to-Speech) for clear narration.
- **Dynamic Backgrounds**: Automatically fetches relevant royalty-free portrait videos from the **Pexels API**.
- **AI Subtitles**: Transcribes audio using **OpenAI Whisper** and generates high-contrast, burned-in subtitles.
- **Zero-Binary Setup**: Subtitles are rendered using `PIL` to remove the need for ImageMagick.
- **YouTube Optimized**: Outputs 1080x1920, 30fps MP4 files ready for upload.

## 🛠️ Prerequisites

- **Python 3.10+**
- **FFmpeg**: Provided automatically via `imageio-ffmpeg` (the script handles the path setup).

## 📥 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Akshzz-16/ai_youtube_vidGen.git
   cd ai_youtube_vidGen
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🔑 API Configuration

The system requires a **Pexels API Key** to download background videos.

1. **Local Setup**: Create a `.env` file in the root directory (copy from `.env.example`):
   ```bash
   PEXELS_API_KEY=your_actual_key_here
   OPENAI_API_KEY=your_optional_key_here
   ```
2. **Security**: The `.env` file is in `.gitignore` and will **NOT** be pushed to GitHub.
3. **Environment Variables**: For production/GitHub Actions, set these as Secrets/Environment variables on your hosting platform.

## 🎮 Usage

### 🧪 Test Mode
Run this to verify your setup and see the generation pipeline in action without rendering a full-length high-quality video:
```bash
python main.py --test
```

### 🎬 Generate a Short
Create a complete, upload-ready YouTube Short:
```bash
python main.py
```
*The final video will be saved in `/output/final_youtube_short.mp4`.*

## 📂 Project Structure

- `main.py`: The single-file orchestrator containing all logic.
- `requirements.txt`: Python package dependencies.
- `/data`: Stores history of used topics.
- `/assets`: Temporary storage for raw audio/video and SRT files.
- `/output`: Where your finished bangers live.

## ⚖️ License
MIT
