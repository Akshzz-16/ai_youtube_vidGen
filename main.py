import os
import json
import random
import os
import json
import random
import argparse
import sys
import logging
import imageio_ffmpeg
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup FFMPEG for MoviePy and Whisper
# This ensures "executable as-is" without system ffmpeg install
FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_BINARY
os.environ["FFMPEG_BINARY"] = FFMPEG_BINARY

# CRITICAL: Whisper expects 'ffmpeg' command. imageio provides 'ffmpeg-win-x86_64-vX.X.exe'
# We must create a 'ffmpeg.exe' copy/shim in the same folder if it doesn't exist.
ffmpeg_dir = os.path.dirname(FFMPEG_BINARY)
ffmpeg_shim = os.path.join(ffmpeg_dir, "ffmpeg.exe")

if not os.path.exists(ffmpeg_shim):
    try:
        import shutil
        shutil.copy(FFMPEG_BINARY, ffmpeg_shim)
        logger.info(f"Created ffmpeg.exe shim at {ffmpeg_shim}")
    except Exception as e:
        logger.error(f"Failed to create ffmpeg shim: {e}")

# Add the directory to PATH so 'ffmpeg' command works
os.environ["PATH"] += os.pathsep + ffmpeg_dir

# --- CONFIGURATION & ASSET MANAGEMENT ---

class Config:
    """Project configuration and API keys."""
    # Assets & Persistence
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    ASSETS_DIR = BASE_DIR / "assets"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Files
    USED_TOPICS_FILE = DATA_DIR / "used_topics.json"
    
    # API Keys (Loaded from Env for security)
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY") 
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Optional: fallback to templates if not provided

class AssetManager:
    """Manages directory creation and file persistence."""
    
    def __init__(self):
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Creates necessary directories if they don't exist."""
        for directory in [Config.DATA_DIR, Config.ASSETS_DIR, Config.OUTPUT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Verified directory: {directory}")

    def load_json(self, filepath: Path, default=None):
        """Loads JSON data from a file, returning default if not found."""
        if not filepath.exists():
            return default if default is not None else {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding {filepath}. Returning default.")
            return default if default is not None else {}

    def save_json(self, filepath: Path, data: any):
        """Saves data to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved data to {filepath}")


# --- CONTENT GENERATION ---

class TopicGenerator:
    """Handles selection of psychology topics and avoiding duplicates."""
    
    # Embedded Database of Psychology Facts
    EMBEDDED_FACTS = [
        {
            "id": "psy_001",
            "topic": "The Zeigarnik Effect",
            "fact": "The Zeigarnik Effect states that people remember uncompleted or interrupted tasks better than completed tasks.",
            "hook": "Ever wonder why you can't stop thinking about that one unfinished thing?",
            "keywords": ["brain", "thinking", "memory", "mind"]
        },
        {
            "id": "psy_002",
            "topic": "The Paradox of Choice",
            "fact": "Having too many choices can actually lead to anxiety and unhappiness, rather than freedom.",
            "hook": "Did you know that having more options might actually be making you miserable?",
            "keywords": ["stress", "anxiety", "choice", "shopping"]
        },
        {
            "id": "psy_003",
            "topic": "Mirror Neurons",
            "fact": "Mirror neurons are brain cells that fire both when we perform an action and when we observe someone else performing it, forming the basis of empathy.",
            "hook": "There's a specific reason why you cringe when you see someone else get hurt.",
            "keywords": ["empathy", "brain", "connection", "neuron"]
        },
        {
            "id": "psy_004",
            "topic": "Spotlight Effect",
            "fact": "The Spotlight Effect is the phenomenon where people tend to overestimate how much others notice aspects of their appearance or behavior.",
            "hook": "Stop worrying, because nobody is actually watching you as closely as you think.",
            "keywords": ["anxiety", "social", "crowd", "eyes"]
        },
        {
            "id": "psy_005",
            "topic": "Confirmation Bias",
            "fact": "Confirmation bias is the tendency to search for, interpret, and recall information in a way that confirms one's preexisting beliefs.",
            "hook": "Your brain is actively tricking you into believing you're always right.",
            "keywords": ["argument", "belief", "brain", "thinking"]
        },
         {
            "id": "psy_006",
            "topic": "Bystander Effect",
            "fact": "The Bystander Effect occurs when the presence of others discourages an individual from intervening in an emergency situation.",
            "hook": "If you're in trouble, you're actually safer with fewer people around.",
            "keywords": ["crowd", "people", "help", "emergency"]
        },
        {
            "id": "psy_007",
            "topic": "Halo Effect",
            "fact": "The Halo Effect is a cognitive bias where one positive trait, like physical attractiveness, makes us assume a person has other positive traits like intelligence.",
            "hook": "We automatically assume beautiful people are nicer and smarter, even when they aren't.",
            "keywords": ["beauty", "face", "attraction", "judgement"]
        }
    ]

    def __init__(self, asset_manager: AssetManager):
        self.asset_manager = asset_manager
        self.used_topics = self.asset_manager.load_json(Config.USED_TOPICS_FILE, default=[])

    def get_next_topic(self) -> Optional[Dict]:
        """Selects a topic that hasn't been used yet."""
        available = [f for f in self.EMBEDDED_FACTS if f['id'] not in self.used_topics]
        
        if not available:
            logger.warning("All embedded topics have been used! Resetting history or needing new topics.")
            # Implementation detail: For now, let's just pick a random one to ensure it keeps working
            # In a real scenario, we'd fetch from online or ask the user to clear the history
            return random.choice(self.EMBEDDED_FACTS)
        
        selected = random.choice(available)
        logger.info(f"Selected topic: {selected['topic']}")
        return selected

    def mark_topic_as_used(self, topic_id: str):
        """Marks a topic as used and persists the list."""
        if topic_id not in self.used_topics:
            self.used_topics.append(topic_id)
            self.asset_manager.save_json(Config.USED_TOPICS_FILE, self.used_topics)

class ScriptGenerator:
    """Generates a script from a given topic."""
    
    def __init__(self):
        pass # initialization if needed later

    def generate_script(self, topic_data: Dict) -> str:
        """
        Generates a script based on the topic data.
        Uses OpenAI if API key is present, otherwise uses templates.
        """
        if Config.OPENAI_API_KEY:
            return self._generate_with_llm(topic_data)
        else:
            return self._generate_with_template(topic_data)

    def _generate_with_template(self, data: Dict) -> str:
        """EXPANDS the fact into a full script using templates."""
        hook = data['hook']
        fact = data['fact']
        topic = data['topic']
        
        # Template structure for a 45-60s short
        # 1. Hook (provided)
        # 2. Intro/Bridge
        # 3. The Core Fact (provided, expanded)
        # 4. Explanation/Elaboration
        # 5. Outro/Loop
        
        bridges = [
            "It sounds crazy, but psychology says it's true.",
            "Here is a little psychological hack for you.",
            "Let me explain how your brain is playing tricks on you.",
            "This is known as the {topic}, and it explains a lot."
        ]
        
        explanations = [
            f"Essentially, {fact.lower()} This happens because our brains are wired to prioritize efficiency over accuracy.",
            f"Think about it. {fact} It controls more of your daily life than you realize.",
            f"In simple terms: {fact} It's a fundamental part of human nature."
        ]
        
        loops = [
            "So, the next time this happens, ask yourself...",
            "Now you know exactly what's happening when...",
            "And that is the answer to the question..."
        ]

        bridge = random.choice(bridges).format(topic=topic)
        explanation = random.choice(explanations)
        loop = random.choice(loops)
        
        # Combine
        script = f"{hook} {bridge} {explanation} {loop}"
        return script

    def _generate_with_llm(self, data: Dict) -> str:
        """Generates script using OpenAI (Placeholder for now)."""
        # In a real implementation, this would call openai.Completion.create
        # For this step, we'll fall back to template to keep it simple unless specifically requested
        logger.info("OpenAI Key detected, but falling back to template for this phase of testing.")
        return self._generate_with_template(data)

# --- MEDIA & ASSETS GENERATION ---

class VoiceGenerator:
    """Generates audio from text using gTTS."""
    
    def __init__(self):
        pass

    def generate_voice(self, script: str, output_path: Path):
        """Generates MP3 from script."""
        try:
            from gtts import gTTS
        except ImportError:
            logger.error("gTTS not installed. Run: pip install gTTS")
            return

        logger.info("Generating Audio with gTTS...")
        # Clean script of newlines for better TTS flow
        clean_script = script.replace("\n", " ")
        
        tts = gTTS(text=clean_script, lang='en', slow=False)
        tts.save(str(output_path))
        logger.info(f"Audio saved to {output_path}")

class MediaFetcher:
    """Fetches background stock video from Pexels."""
    
    PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
    
    def __init__(self):
        pass

    def download_video(self, query: str, output_path: Path):
        """Searches and downloads a video from Pexels."""
        if not Config.PEXELS_API_KEY:
            logger.warning("No Pexels API Key found. Skipping download.")
            return

        try:
            import requests
        except ImportError:
            logger.error("requests not installed. Run: pip install requests")
            return

        headers = {"Authorization": Config.PEXELS_API_KEY}
        params = {
            "query": query,
            "per_page": 1,
            "orientation": "portrait",
            "size": "medium" # Prefer lighter files
        }

        try:
            logger.info(f"Searching Pexels for: {query}")
            response = requests.get(self.PEXELS_SEARCH_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            if not data.get('videos'):
                logger.warning("No videos found for query.")
                return

            video_url = data['videos'][0]['video_files'][0]['link']
            
            # Find a suitable video file (prefer HD but not 4k to save bandwidth, or just take first)
            # Pexels returns multiple 'video_files'. Let's pick the one with width closest to 720 or 1080 
            # But for portrait, logic varies. Let's just grab the first 'link' which simplifies things or specific quality.
            # Actually, let's look for a direct link.
            for v_file in data['videos'][0]['video_files']:
                if v_file['quality'] == 'hd':
                    video_url = v_file['link']
                    break

            logger.info(f"Downloading video from: {video_url}")
            vid_response = requests.get(video_url, stream=True)
            vid_response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in vid_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Video saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to fetch video: {e}")

        except Exception as e:
            logger.error(f"Failed to fetch video: {e}")
            raise # Propagate error

# --- SUBTITLES & VIDEO EDITING ---

class SubtitleGenerator:
    """Generates subtitles/transcription using OpenAI Whisper."""
    
    def __init__(self):
        pass

    def generate_subtitles(self, audio_path: Path) -> List[Dict]:
        """Transcribes audio to get segments with timestamps. Exports SRT."""
        try:
            import whisper
            # FFMPEG path is already set globally
        except ImportError:
            logger.error("openai-whisper not installed.")
            raise

        logger.info("Loading Whisper model (base)... this may take a moment.")
        try:
            model = whisper.load_model("base")
            result = model.transcribe(str(audio_path), fp16=False)
            logger.info("Transcription complete.")
            
            # Export SRT (Requirement 2)
            srt_path = audio_path.with_suffix(".srt")
            self._save_srt(result['segments'], srt_path)
            logger.info(f"SRT saved to {srt_path}")
            
            return result['segments']
        except Exception as e:
            logger.error(f"Transciption failed: {e}")
            raise

    def _save_srt(self, segments: List[Dict], path: Path):
        """Saves segments to SRT format."""
        def format_timestamp(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

        with open(path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, start=1):
                start = format_timestamp(seg['start'])
                end = format_timestamp(seg['end'])
                text = seg['text'].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

class VideoEditor:
    """Combines assets into the final video using MoviePy."""
    
    def __init__(self):
        pass
    
    def _create_text_image(self, text: str, width: int = 900, fontsize: int = 70) -> str:
        """Creates a text image using PIL and returns the temp file path."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import textwrap
        except ImportError:
            logger.error("Pillow not installed.")
            raise

        # Create a dummy image to calculate text size
        # Estimate chars per line based on width and fontsize (approx)
        # Arial 70px ~ 40-50px wide? 900 / 40 = ~22 chars
        wrapper = textwrap.TextWrapper(width=22) 
        lines = wrapper.wrap(text)
        
        # Canvas size
        line_height = int(fontsize * 1.2)
        height = line_height * len(lines) + 20 # padding
        
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Load font - try to find Arial, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", fontsize)
        except IOError:
            try: 
                 # Linux/Mac fallback (generic)
                 font = ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
            except IOError:
                 font = ImageFont.load_default()
        
        # Draw Text with Stroke (simulated by drawing offsets)
        y_text = 10
        stroke_width = 4
        stroke_color = "black"
        text_color = "white"
        
        for line in lines:
            # Calculate text width to center
            # bbox = draw.textbbox((0, 0), line, font=font)
            # text_w = bbox[2] - bbox[0]
            # x_text = (width - text_w) / 2
            
            # Simple center calculation (approximate for default font, better for TrueType)
            text_w = draw.textlength(line, font=font)
            x_text = (width - text_w) / 2

            # Stroke
            for dx, dy in [(-stroke_width, 0), (stroke_width, 0), (0, -stroke_width), (0, stroke_width)]:
                 draw.text((x_text+dx, y_text+dy), line, font=font, fill=stroke_color)
            for dx, dy in [(-stroke_width, -stroke_width), (-stroke_width, stroke_width), (stroke_width, -stroke_width), (stroke_width, stroke_width)]: # Corners
                 draw.text((x_text+dx, y_text+dy), line, font=font, fill=stroke_color)

            # Main Text
            draw.text((x_text, y_text), line, font=font, fill=text_color)
            y_text += line_height

        # Save to temp
        temp_file = Config.ASSETS_DIR / f"temp_text_{random.randint(0, 100000)}.png"
        img.save(temp_file)
        return str(temp_file)

    def create_video(self, audio_path: Path, video_path: Path, subtitles: List[Dict], output_path: Path):
        """Creates the final 9:16 video strictly adhering to YouTube Shorts specs."""
        try:
            # MoviePy 2.0+ Import Compatibility
            from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, ColorClip, ImageClip, vfx
        except ImportError:
             logger.error("moviepy not installed or incompatible version.")
             raise

        logger.info("Starting Video Editing Process...")
        
        try:
            # 1. Load Audio
            if not audio_path.exists():
                logger.error(f"Audio file matching {audio_path} not found.")
                raise FileNotFoundError(f"Audio not found: {audio_path}")
            audio_clip = AudioFileClip(str(audio_path))
            duration = audio_clip.duration

            # 2. Load Video
            if video_path.exists():
                logger.info(f"Loading background video: {video_path}")
                video_clip = VideoFileClip(str(video_path))
                
                # Loop video if shorter than audio
                if video_clip.duration < duration:
                    # MoviePy 2.0: Use vfx.Loop
                    video_clip = video_clip.with_effects([vfx.Loop(duration=duration)])
                
                # Strict Trim to Audio Duration
                video_clip = video_clip.subclipped(0, duration)
            else:
                logger.warning("Background video not found. Using Placeholder ColorClip.")
                video_clip = ColorClip(size=(1080, 1920), color=(20, 20, 20), duration=duration)

            # 3. Strict 9:16 Crop/Resize (1080x1920)
            # Strategy: Resize to cover, then center crop.
            w, h = video_clip.size
            target_w, target_h = 1080, 1920
            target_ratio = target_w / target_h
            current_ratio = w / h
            
            if current_ratio > target_ratio:
                # Video is wider than 9:16 (e.g., 16:9 Landscape)
                # Resize by height to 1920, width will be > 1080
                new_h = target_h
                new_w = int(w * (target_h / h))
                video_clip = video_clip.resized(height=new_h)
                # Center Crop width
                x_center = new_w // 2
                video_clip = video_clip.cropped(x1=x_center - target_w // 2, y1=0, width=target_w, height=target_h)
            else:
                # Video is taller/narrower (rare) or matches
                # Resize by width to 1080, height will be >= 1920
                new_w = target_w
                new_h = int(h * (target_w / w))
                video_clip = video_clip.resized(width=new_w)
                # Center Crop height
                y_center = new_h // 2
                video_clip = video_clip.cropped(x1=0, y1=y_center - target_h // 2, width=target_w, height=target_h)
            
            # Double check size (precaution)
            video_clip = video_clip.resized(new_size=(1080, 1920))
            video_clip = video_clip.with_audio(audio_clip)

            # 4. Burn Subtitles (Using PIL)
            # Requires cleaning up temp files later
            subtitle_clips = []
            created_temp_files = []
            
            for segment in subtitles:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()
                
                # Generate Image for Text
                try:
                    img_path = self._create_text_image(text)
                    created_temp_files.append(img_path)
                    
                    img_clip = (ImageClip(img_path)
                                .with_start(start_time)
                                .with_end(end_time)
                                .with_position(('center', 1300)))
                    
                    subtitle_clips.append(img_clip)
                except Exception as ex:
                    logger.warning(f"Failed to create subtitle image: {ex}")

            # Compose
            final_video = CompositeVideoClip([video_clip] + subtitle_clips)
            
            # 5. Write File (Strict Specs)
            logger.info(f"Rendering video to {output_path}...")
            final_video.write_videofile(
                str(output_path), 
                codec='libx264', 
                audio_codec='aac', 
                fps=30, 
                threads=4, 
                preset='fast',
                logger=None
            )
            logger.info("Video rendering complete!")
            
            # Cleanup
            for p in created_temp_files:
                try:
                    os.remove(p)
                except:
                    pass

        except Exception as e:
            logger.error(f"Video editing failed: {e}")
            raise # Propagate error

            if "ImageMagick" in str(e):
                logger.error("ImageMagick is likely missing. Please install ImageMagick and configure MoviePy.")

# --- YOUTUBE UPLOAD (OPTIONAL) ---

class YouTubeUploader:
    """Handles uploading the video to YouTube (Placeholder/Stub)."""
    
    def __init__(self):
        pass

    def upload_video(self, video_path: Path, title: str, description: str, tags: List[str]):
        """Uploads video to YouTube. Requires OAuth setup."""
        if not Config.OPENAI_API_KEY: # Using a proxy check for now or just skipping by default
            # Real implementation requires google-auth-oauthlib and client_secrets.json
            logger.info("YouTube Upload skipped (Requires OAuth setup). To enable, implement full OAuth flow.")
            return

        logger.info(f"Mock Uploading {video_path} to YouTube...")
        logger.info(f"Title: {title}")
        logger.info(f"Tags: {tags}")
        # Placeholder for actual API call
        logger.info("Upload successful! (Simulation)")


# --- MAIN ORCHESTRATOR (FINAL) ---

def main():
    parser = argparse.ArgumentParser(description="YouTube Shorts Generator")
    parser.add_argument("--test", action="store_true", help="Run in test mode to verify components.")
    parser.add_argument("--upload", action="store_true", help="Upload to YouTube after generation.")
    args = parser.parse_args()

    # 1. Initialize
    assets = AssetManager()
    topic_gen = TopicGenerator(assets)
    script_gen = ScriptGenerator()
    voice_gen = VoiceGenerator()
    media_fetcher = MediaFetcher()
    subtitle_gen = SubtitleGenerator()
    video_editor = VideoEditor()
    uploader = YouTubeUploader()

    if args.test:
        logger.info("Running in TEST mode...")
        
        # Test Topic Selection
        logger.info("--- Testing Topic Generator ---")
        topic = topic_gen.get_next_topic()
        if not topic:
             logger.error("No topic found")
             return
        print(f"Topic: {topic['topic']}")
        
        # Test Script Generation
        logger.info("\n--- Testing Script Generator ---")
        script = script_gen.generate_script(topic)
        print(f"Script (Preview): {script[:50]}...")
        
        # Test Voice Generation
        logger.info("\n--- Testing Voice Generator ---")
        audio_path = Config.ASSETS_DIR / "test_audio.mp3"
        # Only regenerate if missing or force? For test let's regenerate
        voice_gen.generate_voice(script, audio_path)
        
        # Test Media Fetcher
        logger.info("\n--- Testing Media Fetcher ---")
        video_path = Config.ASSETS_DIR / "test_video.mp4"
        query = topic['keywords'][0] if topic.get('keywords') else "abstract"
        if not video_path.exists(): # Don't redownload in loop during dev if exists
            media_fetcher.download_video(query, video_path)
        
        # Test Subtitles
        logger.info("\n--- Testing Subtitle Generator ---")
        if audio_path.exists():
            subtitles = subtitle_gen.generate_subtitles(audio_path)
            print(f"Subtitles (First 2): {subtitles[:2]}")
        else:
            subtitles = []
            logger.warning("Audio path missing, skipping subtitles.")

        # Test Video Editor
        logger.info("\n--- Testing Video Editor ---")
        output_path = Config.OUTPUT_DIR / "test_short.mp4"
        # We will attempt to run it. If ImageMagick is missing, it will log error but not crash script hard.
        video_editor.create_video(audio_path, video_path, subtitles, output_path)
        
        logger.info("\nTEST MODE COMPLETE.")
        return

    # --- FULL EXECUTION FLOW ---
    logger.info("Starting Auto-Generation Process...")
    
    # 1. Select Topic
    topic = topic_gen.get_next_topic()
    if not topic:
        return
    
    # 2. Generate Script
    script = script_gen.generate_script(topic)
    
    # 3. Voice
    audio_path = Config.ASSETS_DIR / f"{topic['id']}.mp3"
    voice_gen.generate_voice(script, audio_path)
    
    # 4. Media
    video_path = Config.ASSETS_DIR / f"{topic['id']}_bg.mp4"
    if not video_path.exists():
        media_fetcher.download_video(topic['keywords'][0], video_path)
    
    # 5. Subtitles
    subtitles = subtitle_gen.generate_subtitles(audio_path)
    
    # 6. Edit
    output_path = Config.OUTPUT_DIR / "final_youtube_short.mp4"
    video_editor.create_video(audio_path, video_path, subtitles, output_path)
    
    # 7. Upload (Optional)
    if args.upload:
        title = f"{topic['topic']} #Shorts #Psychology"
        uploader.upload_video(output_path, title, script[:100], topic['keywords'])
    
    # 8. Finalize
    topic_gen.mark_topic_as_used(topic['id'])
    logger.info(f"DONE! Video saved to {output_path}")

if __name__ == "__main__":
    main()
