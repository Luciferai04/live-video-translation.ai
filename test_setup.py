#!/usr/bin/env python3
"""
Test script for Sonar Real-time Translator

This script tests the basic functionality of the translation system
without requiring audio/video input.
"""

import asyncio
import numpy as np
import sys
import logging
from sonar_translator import SonarRealtimeTranslator, TranslationConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_basic_functionality():
    """Test basic translation functionality"""
    print("🎯 Testing Sonar Real-time Translator")
    print("=" * 40)
    
    try:
        # Create configuration (without Gemini API key for basic test)
        config = TranslationConfig(
            device="cpu",
            target_langs=["hin", "ben"],
            chunk_duration=2.0
        )
        
        print("🔧 Initializing translator...")
        translator = SonarRealtimeTranslator(config)
        
        print("📥 Loading Sonar models (this may take a few minutes)...")
        await translator.initialize_models()
        print("✅ Models loaded successfully!")
        
        # Test with synthetic audio (simulating silent audio)
        print("\n🎤 Testing audio processing...")
        sample_rate = 16000
        duration = 2.0
        
        # Create silent audio (in a real scenario, this would be speech)
        test_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        # Add some very quiet noise to simulate minimal audio input
        test_audio += np.random.normal(0, 0.001, test_audio.shape).astype(np.float32)
        
        print(f"Processing {duration}s of audio...")
        result = await translator.process_audio_chunk(test_audio)
        
        print("\n📋 Results:")
        print(f"Original text: '{result['original']}'")
        print(f"Translations: {result['translations']}")
        
        if not result['original']:
            print("⚠️  No speech detected in test audio (this is expected for silent audio)")
            print("💡 To test with real speech, use the full application with microphone input")
        
        # Test text translation directly
        print("\n🗺️ Testing direct text translation...")
        test_text = "Hello, how are you today?"
        print(f"Testing translation of: '{test_text}'")
        
        for lang in config.target_langs:
            translation = await translator.translate_text(test_text, lang)
            lang_name = {"hin": "Hindi", "ben": "Bengali"}.get(lang, lang)
            print(f"{lang_name}: {translation}")
        
        print("\n🧹 Cleaning up...")
        await translator.cleanup()
        
        print("\n✅ Test completed successfully!")
        print("🎉 Your Sonar Real-time Translator is ready to use!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Test error: {e}", exc_info=True)
        return False

async def test_dependencies():
    """Test if all required dependencies are available"""
    print("📋 Testing dependencies...")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("fairseq2", "fairseq2 (Sonar)"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("pyaudio", "PyAudio"),
        ("streamlit", "Streamlit"),
        ("google.generativeai", "Google Generative AI")
    ]
    
    missing_deps = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Missing!")
            missing_deps.append(name)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are available!")
    return True

def test_audio_devices():
    """Test audio device availability"""
    print("\n🎤 Testing audio devices...")
    
    try:
        import pyaudio
        
        audio = pyaudio.PyAudio()
        device_count = audio.get_device_count()
        
        print(f"Found {device_count} audio devices:")
        
        input_devices = []
        for i in range(device_count):
            device_info = audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append((i, device_info['name']))
                print(f"  ✅ Input Device {i}: {device_info['name']}")
        
        if not input_devices:
            print("⚠️  No audio input devices found!")
            return False
        
        audio.terminate()
        print(f"\n✅ Found {len(input_devices)} audio input device(s)")
        return True
        
    except Exception as e:
        print(f"❌ Audio device test failed: {e}")
        return False

def test_camera():
    """Test camera availability"""
    print("\n📹 Testing camera...")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera is working")
                cap.release()
                return True
            else:
                print("⚠️  Camera opened but cannot read frames")
                cap.release()
                return False
        else:
            print("⚠️  Cannot open camera (this is optional for audio-only translation)")
            return False
            
    except Exception as e:
        print(f"⚠️  Camera test failed: {e} (this is optional for audio-only translation)")
        return False

async def main():
    """Main test function"""
    print("🚀 Sonar Real-time Translator - Setup Test")
    print("=" * 50)
    
    # Test dependencies
    deps_ok = await test_dependencies()
    if not deps_ok:
        print("\n❌ Setup test failed - missing dependencies")
        sys.exit(1)
    
    # Test audio devices
    audio_ok = test_audio_devices()
    
    # Test camera (optional)
    camera_ok = test_camera()
    
    # Test basic functionality
    if deps_ok and audio_ok:
        print("\n" + "=" * 50)
        functionality_ok = await test_basic_functionality()
        
        if functionality_ok:
            print("\n" + "=" * 50)
            print("🎉 ALL TESTS PASSED!")
            print("\n📋 Next steps:")
            print("1. Get a Gemini API key from Google AI Studio")
            print("2. Run the web interface: streamlit run streamlit_app.py")
            print("3. Or run the CLI version: python realtime_video_translator.py")
            
            if not camera_ok:
                print("\n💡 Note: Camera not detected, but audio translation will still work!")
        else:
            print("\n❌ Setup test failed")
            sys.exit(1)
    else:
        print("\n❌ Setup test failed - check dependencies and audio devices")
        sys.exit(1)

if __name__ == "__main__":
    # Run the test
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

