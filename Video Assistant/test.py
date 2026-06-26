import os
import sys

# Safeguard: Tells Python to look for folders inside the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now your imports will match and resolve cleanly!
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions

def main():
    # 1. Provide your video source URL
    source_url = "https://www.youtube.com/watch?v=nx-mGN2Fz5M"
    
    print("=== STEP 1: Processing Audio (Downloading & Chunking) ===")
    # This calls your audio_processor.py logic to get 10-minute .wav segments
    audio_chunks = process_input(source_url)
    
    if not audio_chunks:
        print("Error: No audio chunks generated.")
        return

    print("\n=== STEP 2: Running Whisper Transcription locally ===")
    # This calls your faster-whisper code to stitch the chunks together
    full_transcript = transcribe_all(audio_chunks)
    print(f"\n--- Complete Transcript Preview ---\n{full_transcript[:500]}...\n")

    print("\n=== STEP 3: Generating Insights with AI ===")
    # Pass the full transcript to your core AI modules
    title = generate_title(full_transcript)
    summary = summarize(full_transcript)
    action_items = extract_action_items(full_transcript)
    decisions = extract_key_decisions(full_transcript)
    questions = extract_questions(full_transcript)

    print("\n==================================================")
    print(f"VIDEO ASSISTANT ANALYSIS FOR: {title}")
    print("==================================================")
    print(f"\n[SUMMARY]\n{summary}")
    print(f"\n[ACTION ITEMS]\n{action_items}")
    print(f"\n[KEY DECISIONS]\n{decisions}")
    print(f"\n[UNANSWERED QUESTIONS]\n{questions}")

if __name__ == "__main__":
    main()