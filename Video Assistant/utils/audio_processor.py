import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ------------------------------------------------------------------
# CONFIGURATION: Target the folder where this script lives
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))

# Point pydub directly to the local .exe files inside the utils folder
AudioSegment.converter = os.path.join(UTILS_DIR, "ffmpeg.exe")
AudioSegment.ffprobe   = os.path.join(UTILS_DIR, "ffprobe.exe")
# ------------------------------------------------------------------

def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    
    # Path to your manually exported cookies file
    cookie_path = os.path.join(UTILS_DIR, "youtube_cookies.txt")
    
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "ffmpeg_location": UTILS_DIR, 
        
        # Uses the static exported file to bypass live-browser locking & folder lag
        "cookiefile": cookie_path if os.path.exists(cookie_path) else None,
        
        # 🔑 FIX 1: Tell the YouTube extractor to ignore the broken player clients
        "extractor_args": {
            "youtube": {
                "player_client": ["default", "-android_sdkless", "-ios_downgraded"]
            }
        },
        
        # 🔑 FIX 2: Send clean browser headers to appear like a web client
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        },

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        
        "quiet": False,
    }

    # Visual warning if you forgot to place the cookie file
    if not os.path.exists(cookie_path):
        print("\n⚠️ WARNING: 'youtube_cookies.txt' not found in utils folder.")
        print("If you still get an HTTP 403 error, please export your cookies to that path.\n")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(filename)
        filename = base + ".wav"
        
        # Safeguard if postprocessing altered the final naming scheme slightly
        if not os.path.exists(filename):
            actual_dir = os.path.dirname(filename)
            base_name = os.path.basename(base)
            for f in os.listdir(actual_dir):
                if f.startswith(base_name) and f.endswith(".wav"):
                    filename = os.path.join(actual_dir, f)
                    break
        
    return filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    
    # pydub uses the custom binary paths configured at the top
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16khz mono
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """Splits the wav audio file into segments of a specified minute duration."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000  # convert minutes to milliseconds

    chunks = []
    # Creating chunks of audio and saving them in the same directory
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start+chunk_ms]
        
        # Clean up naming structure so it reads cleanly as an extension
        chunk_path = f"{os.path.splitext(wav_path)[0]}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


# ==========================================
# EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # Target video URL
    target_url = "https://www.youtube.com/watch?v=nx-mGN2Fz5M"
    
    print("1. Downloading audio from YouTube...")
    downloaded_raw_wav = download_youtube_audio(target_url)
    print(f"Downloaded raw file: {downloaded_raw_wav}\n")

    print("2. Resampling audio to 16kHz Mono...")
    converted_file = convert_to_wav(downloaded_raw_wav)
    print(f"Converted file saved at: {converted_file}\n")

    print(f"3. Splitting processed audio into 10-minute chunks...")
    audio_chunks = chunk_audio(converted_file, chunk_minutes=10)
    
    print(f"\nSuccessfully generated {len(audio_chunks)} chunks:")
    for chunk in audio_chunks:
        print(f" -> {chunk}")


def process_input(url_or_path: str):
    print("1. Downloading audio from YouTube...")
    downloaded_raw_wav = download_youtube_audio(url_or_path)
    print(f"Downloaded raw file: {downloaded_raw_wav}\n")

    print("2. Resampling audio to 16kHz Mono...")
    # FIXED: Make sure you assign the output of convert_to_wav to the variable name your chunker expects!
    wav_path = convert_to_wav(downloaded_raw_wav) 
    print(f"Converted file saved at: {wav_path}\n")

    print(f"3. Chunking audio...")
    chunks = chunk_audio(wav_path) # Now wav_path exists and has a value!
    return chunks