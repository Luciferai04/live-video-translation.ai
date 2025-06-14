#!/usr/bin/env python3
import gradio as gr
import asyncio
import threading
import queue
import time
import numpy as np
import cv2
import pyaudio
from sonar_translator import SonarRealtimeTranslator, TranslationConfig
import logging
import os
import warnings
from datetime import datetime
import json
from typing import Tuple, Optional, Dict, Any

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
logging.basicConfig(level=logging.ERROR)

# Global state for the application
class VideoTranslatorApp:
    def __init__(self):
        self.is_running = False
        self.translator = None
        self.video_cap = None
        self.audio_queue = queue.Queue(maxsize=50)  # Reduced queue size for faster processing
        self.results_queue = queue.Queue(maxsize=20)
        self.video_thread = None
        self.processing_thread = None
        self.transcript_entries = []
        self.start_time = None
        self.initial_delay_done = False
        self.current_frame = None
        self.current_text = ""
        self.current_translations = {"hin": "", "ben": ""}
        self.update_counter = 0
        self.last_update_time = 0
        
    async def initialize_translator(self):
        """Initialize the Sonar translator"""
        if self.translator is None:
            config = TranslationConfig(
                target_langs=["hin", "ben"],
                device="cpu",
                chunk_duration=1.5  # Further reduced to 1.5 seconds for even faster processing
            )
            
            print("🔄 Loading Sonar models...")
            self.translator = SonarRealtimeTranslator(config)
            await self.translator.initialize_models()
            print("✅ Models loaded successfully!")
            
        return self.translator
    
    def extract_audio_from_video(self):
        """Extract audio from video frames and put into queue"""
        print("🎥 Starting video capture with audio extraction...")
        
        try:
            # Initialize video capture with macOS specific settings
            import platform
            if platform.system() == 'Darwin':  # macOS
                self.video_cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            else:
                self.video_cap = cv2.VideoCapture(0)
                
            if not self.video_cap.isOpened():
                print("❌ Failed to open camera, trying alternatives...")
                for i in range(3):
                    if platform.system() == 'Darwin':
                        self.video_cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                    else:
                        self.video_cap = cv2.VideoCapture(i)
                    if self.video_cap.isOpened():
                        print(f"✅ Camera {i} opened successfully")
                        break
                    self.video_cap.release()
                
                if not self.video_cap.isOpened():
                    print("❌ Could not open any camera")
                    return
            
            # Test frame capture
            ret, test_frame = self.video_cap.read()
            if not ret or test_frame is None:
                print("❌ Camera opened but cannot read frames")
                return
            else:
                print(f"✅ Camera working! Frame size: {test_frame.shape}")
            
            # Set video properties
            self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video_cap.set(cv2.CAP_PROP_FPS, 20)
            
            # Initialize audio capture
            audio = pyaudio.PyAudio()
            
            # Find audio input device
            input_device = None
            for i in range(audio.get_device_count()):
                device_info = audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    input_device = i
                    break
                    
            if input_device is None:
                print("❌ No audio input device found")
                return
                
            # Open audio stream
            audio_stream = audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=1024
            )
            
            print(f"🎤 Audio capture started on device {input_device}")
            self.start_time = time.time()
            frame_count = 0
            
            while self.is_running:
                # Capture video frame
                ret, frame = self.video_cap.read()
                if not ret or frame is None:
                    print("❌ Failed to read frame")
                    time.sleep(0.1)
                    continue
                    
                frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                # Add overlays to frame
                timestamp_text = f"Time: {elapsed_time:.1f}s"
                cv2.putText(frame, timestamp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Add status overlay
                if elapsed_time < 3:  # Reduced to 3 seconds
                    status_text = f"Buffering... {3-elapsed_time:.1f}s remaining"
                    color = (0, 255, 255)  # Yellow
                else:
                    status_text = "LIVE TRANSLATION"
                    color = (0, 255, 0)  # Green
                    
                cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Convert frame for display (BGR to RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = rgb_frame.copy()
                
                # Capture audio data
                try:
                    audio_data = audio_stream.read(1024, exception_on_overflow=False)
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                    
                    # Only process audio if there's a signal
                    if np.max(np.abs(audio_array)) > 0.001:
                        try:
                            self.audio_queue.put(audio_array, block=False)
                        except queue.Full:
                            # Remove old audio to make space
                            try:
                                self.audio_queue.get_nowait()
                                self.audio_queue.put(audio_array, block=False)
                            except queue.Empty:
                                pass
                except Exception as e:
                    if self.is_running:
                        print(f"Audio capture error: {e}")
                        
                # Control frame rate
                time.sleep(1/20)  # Target 20 FPS
                
        except Exception as e:
            print(f"Video capture error: {e}")
        finally:
            if self.video_cap:
                self.video_cap.release()
            if 'audio_stream' in locals():
                audio_stream.stop_stream()
                audio_stream.close()
            if 'audio' in locals():
                audio.terminate()
            print("🎥 Video capture stopped")
    
    def process_audio_continuously(self):
        """Process audio chunks for translation after initial delay"""
        if not self.translator:
            print("❌ No translator available")
            return
            
        print("🔄 Audio processing started (waiting for 5s initial delay)")
        audio_buffer = []
        target_samples = int(16000 * 1.5)  # Further reduced to 1.5 seconds for faster processing
        
        # Reduced initial delay to 3 seconds for faster startup
        while self.is_running:
            if self.start_time and (time.time() - self.start_time) >= 3:
                self.initial_delay_done = True
                print("✅ Initial delay completed, starting live translation")
                break
            time.sleep(0.1)  # Check even more frequently
        
        while self.is_running and self.initial_delay_done:
            try:
                # Collect audio samples
                while len(np.concatenate(audio_buffer) if audio_buffer else []) < target_samples:
                    if not self.is_running:
                        break
                    try:
                        data = self.audio_queue.get(timeout=1.0)
                        audio_buffer.append(data)
                    except queue.Empty:
                        continue
                        
                if audio_buffer and self.is_running:
                    # Process the audio chunk
                    full_audio = np.concatenate(audio_buffer)
                    chunk_to_process = full_audio[:target_samples]
                    
                    # Reduced overlap for faster processing
                    overlap_samples = int(16000 * 0.3)  # Reduced from 0.5 to 0.3 seconds
                    remaining_samples = len(full_audio) - target_samples + overlap_samples
                    if remaining_samples > 0:
                        audio_buffer = [full_audio[-remaining_samples:]]
                    else:
                        audio_buffer = []
                    
                    # Process with translator
                    def run_translation():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(
                                self.translator.process_audio_chunk(chunk_to_process)
                            )
                            
                            if result and result.get('original', '').strip():
                                # Add timestamp for transcript
                                elapsed_time = time.time() - self.start_time
                                result['timestamp'] = elapsed_time
                                result['formatted_time'] = f"{int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}"
                                
                                # Update current state
                                self.current_text = result['original']
                                self.current_translations = result.get('translations', {})
                                self.last_update_time = time.time()
                                self.update_counter += 1
                                
                                # Add to transcript
                                self.transcript_entries.append({
                                    'time': result['formatted_time'],
                                    'timestamp': elapsed_time,
                                    'text': result['original'],
                                    'translations': result.get('translations', {})
                                })
                                
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                print(f"[{timestamp}] 🔄 Video translation: '{result['original'][:30]}...' at {result['formatted_time']}")
                                
                        except Exception as e:
                            if "ScriptRunContext" not in str(e):
                                print(f"Translation error: {e}")
                        finally:
                            try:
                                loop.close()
                            except:
                                pass
                    
                    run_translation()
                    
            except Exception as e:
                if self.is_running:
                    print(f"Audio processing error: {e}")
                break
                
        print("🔄 Audio processing stopped")
    
    def start_translation(self):
        """Start the video translation process"""
        if not self.is_running:
            # Initialize translator
            async def init():
                await self.initialize_translator()
            
            try:
                asyncio.run(init())
                
                if self.translator:
                    self.is_running = True
                    self.initial_delay_done = False
                    self.transcript_entries = []
                    self.update_counter = 0
                    
                    # Start video capture thread
                    self.video_thread = threading.Thread(target=self.extract_audio_from_video)
                    self.video_thread.daemon = True
                    self.video_thread.start()
                    
                    # Start audio processing thread
                    self.processing_thread = threading.Thread(target=self.process_audio_continuously)
                    self.processing_thread.daemon = True
                    self.processing_thread.start()
                    
                    return "🎥 Video translation started! 3-second buffer, then live translation begins..."
                else:
                    return "❌ Failed to initialize translator"
                    
            except Exception as e:
                return f"❌ Initialization error: {e}"
        else:
            return "⚠️ Translation is already running!"
    
    def stop_translation(self):
        """Stop the video translation process"""
        if self.is_running:
            self.is_running = False
            
            # Wait for threads to finish
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=3)
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=3)
            
            # Clear queues
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            return "⏹️ Video translation stopped! Transcript is available below."
        else:
            return "⚠️ Translation is not running!"
    
    def save_transcript(self):
        """Save the complete transcript to file"""
        if self.transcript_entries:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_transcript_{timestamp}.json"
            
            transcript_data = {
                "session_info": {
                    "timestamp": datetime.now().isoformat(),
                    "total_entries": len(self.transcript_entries),
                    "duration": f"{len(self.transcript_entries) * 3}+ seconds"
                },
                "transcript": self.transcript_entries
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            
            # Also create a simple text version
            text_filename = f"video_transcript_{timestamp}.txt"
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write("VIDEO TRANSCRIPT\n")
                f.write("=" * 50 + "\n\n")
                
                for entry in self.transcript_entries:
                    f.write(f"[{entry['time']}] {entry['text']}\n")
                    for lang, translation in entry['translations'].items():
                        lang_name = "Hindi" if lang == "hin" else "Bengali"
                        f.write(f"  {lang_name}: {translation}\n")
                    f.write("\n")
            
            return f"📁 Transcript saved to {filename} and {text_filename}"
        else:
            return "⚠️ No transcript entries to save"
    
    def get_current_frame(self):
        """Get the current video frame"""
        if self.current_frame is not None:
            return self.current_frame
        else:
            # Return a placeholder image
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            if self.is_running:
                cv2.putText(placeholder, "Starting video feed...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(placeholder, "Click 'Start' to begin", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return placeholder
    
    def get_status_info(self):
        """Get current status information"""
        if self.is_running:
            if self.start_time:
                elapsed = time.time() - self.start_time
                elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
            else:
                elapsed_str = "00:00"
            
            if self.initial_delay_done:
                mode = "🎤 Live"
            else:
                mode = "⏳ Buffering"
                
            status = "🟢 Recording"
        else:
            elapsed_str = "00:00"
            mode = "⏸️ Stopped"
            status = "🔴 Stopped"
        
        return {
            "status": status,
            "elapsed": elapsed_str,
            "translations": self.update_counter,
            "mode": mode
        }
    
    def get_current_translations(self):
        """Get current translation text"""
        if self.current_text:
            timestamp_display = ""
            if self.last_update_time > 0:
                timestamp_display = f" (Updated: {datetime.fromtimestamp(self.last_update_time).strftime('%H:%M:%S')})"
            
            original = f"Original English{timestamp_display}:\n{self.current_text}"
            
            hindi = f"Hindi Translation:\n{self.current_translations.get('hin', '')}"
            bengali = f"Bengali Translation:\n{self.current_translations.get('ben', '')}"
            
            return original, hindi, bengali
        else:
            if self.is_running:
                if self.initial_delay_done:
                    msg = "🎤 Listening for speech in video..."
                else:
                    msg = "⏳ Buffering video (3 seconds)..."
            else:
                msg = "Click 'Start Video Translation' to begin"
            
            return msg, "", ""
    
    def get_transcript_summary(self):
        """Get transcript summary"""
        if self.transcript_entries:
            total_entries = len(self.transcript_entries)
            if self.transcript_entries:
                first_entry = self.transcript_entries[0]
                last_entry = self.transcript_entries[-1]
                duration = f"{first_entry['time']} - {last_entry['time']}"
            else:
                duration = "00:00 - 00:00"
            
            total_words = sum(len(entry['text'].split()) for entry in self.transcript_entries)
            
            # Get last few entries for display
            recent_entries = self.transcript_entries[-5:]  # Last 5 entries
            transcript_text = ""
            for entry in recent_entries:
                transcript_text += f"[{entry['time']}] {entry['text']}\n"
                for lang, translation in entry['translations'].items():
                    lang_name = "Hindi" if lang == "hin" else "Bengali"
                    transcript_text += f"  {lang_name}: {translation}\n"
                transcript_text += "\n"
            
            return f"Total Entries: {total_entries} | Duration: {duration} | Words: {total_words}", transcript_text
        else:
            return "No transcript entries yet", ""

# Create global app instance
app = VideoTranslatorApp()

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Live Video Translator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📹 Live Video Translator with Transcript
            *Real-time video translation with audio extraction using Meta's Sonar*
            """
        )
        
        # Control buttons
        with gr.Row():
            start_btn = gr.Button("🚀 Start Video Translation", variant="primary")
            stop_btn = gr.Button("⏹️ Stop Video", variant="secondary")
            save_btn = gr.Button("💾 Save Transcript", variant="secondary")
            clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
        
        # Status display
        with gr.Row():
            status_display = gr.Textbox(label="Status", interactive=False)
            elapsed_display = gr.Textbox(label="Elapsed Time", interactive=False)
            translations_display = gr.Number(label="Translations", interactive=False)
            mode_display = gr.Textbox(label="Mode", interactive=False)
        
        # Main content area
        with gr.Row():
            # Video feed (left column)
            with gr.Column(scale=2):
                gr.Markdown("## 📹 Live Video Feed")
                video_output = gr.Image(label="Live Video", streaming=True)
            
            # Translations (right column)
            with gr.Column(scale=1):
                gr.Markdown("## 📝 Live Translations")
                original_text = gr.Textbox(label="Original English", lines=4, interactive=False)
                hindi_text = gr.Textbox(label="Hindi Translation", lines=3, interactive=False)
                bengali_text = gr.Textbox(label="Bengali Translation", lines=3, interactive=False)
        
        # Transcript section
        gr.Markdown("## 📜 Video Transcript")
        transcript_summary = gr.Textbox(label="Transcript Summary", interactive=False)
        transcript_display = gr.Textbox(label="Recent Transcript Entries", lines=10, interactive=False)
        
        # Message display
        message_display = gr.Textbox(label="Messages", interactive=False)
        
        # Instructions
        gr.Markdown(
            """
            ## 📋 How to Use
            1. **Start**: Click "🚀 Start Video Translation"
            2. **Buffer Phase**: First 3 seconds are buffered (no translation)
            3. **Live Translation**: After 3 seconds, live translation begins
            4. **View Results**: Watch real-time translations in the right panel
            5. **Stop**: Click "⏹️ Stop Video" when done
            6. **Save**: Use "💾 Save Transcript" to export the complete transcript
            
            **Features:**
            - Real-time video feed with status overlay
            - Live audio extraction and translation
            - Complete transcript with timestamps
            - Export to JSON and text formats
            """
        )
        
        # Event handlers
        def start_handler():
            message = app.start_translation()
            return message
        
        def stop_handler():
            message = app.stop_translation()
            return message
        
        def save_handler():
            message = app.save_transcript()
            return message
        
        def clear_handler():
            app.current_text = ""
            app.current_translations = {"hin": "", "ben": ""}
            app.transcript_entries = []
            app.update_counter = 0
            return "🗑️ Cleared all data!"
        
        def update_displays():
            """Update all displays with current data"""
            # Get current frame
            frame = app.get_current_frame()
            
            # Get status info
            status_info = app.get_status_info()
            
            # Get translations
            original, hindi, bengali = app.get_current_translations()
            
            # Get transcript
            summary, transcript = app.get_transcript_summary()
            
            return (
                frame,
                status_info["status"],
                status_info["elapsed"],
                status_info["translations"],
                status_info["mode"],
                original,
                hindi,
                bengali,
                summary,
                transcript
            )
        
        # Button event handlers
        start_btn.click(start_handler, outputs=message_display)
        stop_btn.click(stop_handler, outputs=message_display)
        save_btn.click(save_handler, outputs=message_display)
        clear_btn.click(clear_handler, outputs=message_display)
        
        # Auto-update timer using gr.Timer
        timer = gr.Timer(value=0.2)  # Update every 200ms for more responsive UI
        timer.tick(
            update_displays,
            outputs=[
                video_output,
                status_display,
                elapsed_display,
                translations_display,
                mode_display,
                original_text,
                hindi_text,
                bengali_text,
                transcript_summary,
                transcript_display
            ]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

