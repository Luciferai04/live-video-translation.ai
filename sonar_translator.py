import asyncio
import torch
import torchaudio
import numpy as np
import cv2
import librosa
import soundfile as sf
import tempfile
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict
from sonar.inference_pipelines.speech import SpeechToTextModelPipeline
from sonar.inference_pipelines.text import TextToTextModelPipeline
import google.generativeai as genai
import logging
import warnings

# Suppress all warnings related to torch and torchaudio
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torchaudio.*")
warnings.filterwarnings("ignore", message=".*torch.*")
warnings.filterwarnings("ignore", message=".*fairseq2.*")

# Set environment variables
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging - only show errors to reduce noise
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Suppress specific loggers
for logger_name in ['torch', 'torchaudio', 'fairseq2', 'transformers', 'sonar']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

@dataclass
class TranslationConfig:
    """Configuration for the translation system"""
    source_lang: str = "eng"
    target_langs: List[str] = None
    sample_rate: int = 16000
    chunk_duration: float = 1.5  # seconds - reduced for faster processing
    overlap_duration: float = 0.2  # seconds - reduced overlap
    gemini_api_key: str = None
    device: str = "cpu"  # or "cuda" if GPU available
    gemini_timeout: float = 2.0  # seconds - timeout for Gemini API calls
    
    def __post_init__(self):
        if self.target_langs is None:
            self.target_langs = ["hin", "ben"]  # Hindi and Bengali

class SonarRealtimeTranslator:
    """Real-time video and audio translator using Sonar by Meta and Gemini"""
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self.speech_model = None
        self.text_model = None
        self.gemini_model = None
        
        # Audio processing buffers (removed AudioDecoder and Collater as they're internal to pipelines)
        
        # Buffers for real-time processing
        self.audio_buffer = np.array([])
        self.translation_cache = OrderedDict()  # LRU cache
        self.max_cache_size = 500  # Reduced cache size for faster lookup
        
        # Pre-allocate arrays for better performance
        self._target_length = int(config.sample_rate * config.chunk_duration)
        self._temp_tensor = torch.zeros(self._target_length, dtype=torch.float32, device=self.device)
        
    async def initialize_models(self):
        """Initialize all required models"""
        print("🔄 Loading Sonar models...")
        
        # Load Sonar Speech-to-Text pipeline
        try:
            print("   Loading speech-to-text pipeline...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.speech_model = SpeechToTextModelPipeline(
                    encoder="sonar_speech_encoder_eng",
                    decoder="text_sonar_basic_decoder",
                    tokenizer="text_sonar_basic_encoder",
                    device=self.device
                )
            print("   ✅ Speech-to-text pipeline loaded")
        except Exception as e:
            logger.error(f"Failed to load speech-to-text pipeline: {e}")
            raise RuntimeError("Could not load Sonar speech-to-text pipeline")
        
        # Load Sonar Text-to-Text pipeline
        try:
            print("   Loading text-to-text pipeline...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.text_model = TextToTextModelPipeline(
                    encoder="text_sonar_basic_encoder",
                    decoder="text_sonar_basic_decoder",
                    tokenizer="text_sonar_basic_encoder",
                    device=self.device
                )
            print("   ✅ Text-to-text pipeline loaded")
        except Exception as e:
            logger.error(f"Failed to load text-to-text pipeline: {e}")
            raise RuntimeError("Could not load Sonar text-to-text pipeline")
        
        # Initialize Gemini for enhanced translation
        if self.config.gemini_api_key:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    genai.configure(api_key=self.config.gemini_api_key)
                    self.gemini_model = genai.GenerativeModel('models/gemini-2.0-flash')
                print("   ✅ Gemini model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.gemini_model = None
        
        print("✅ All models loaded successfully!")
    
    def preprocess_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """Preprocess audio data for Sonar model - optimized version"""
        try:
            # Fast dtype check and conversion
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32, copy=False)
            
            # Fast normalization using vectorized operations
            abs_max = np.abs(audio_data).max()
            if abs_max > 0:
                audio_data = audio_data * (1.0 / abs_max)  # Faster than division
            
            # Fast tensor conversion - reuse pre-allocated tensor when possible
            current_length = len(audio_data)
            if current_length == self._target_length:
                # Perfect match - copy directly to pre-allocated tensor
                self._temp_tensor.copy_(torch.from_numpy(audio_data))
                audio_tensor = self._temp_tensor
            else:
                # Convert to tensor first
                audio_tensor = torch.from_numpy(audio_data).to(self.device)
                
                # Handle length mismatch
                if current_length < self._target_length:
                    # Pad with zeros - faster than functional.pad
                    padding = self._target_length - current_length
                    audio_tensor = torch.cat([audio_tensor, torch.zeros(padding, device=self.device)])
                else:
                    # Truncate
                    audio_tensor = audio_tensor[:self._target_length]
            
            return audio_tensor.unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise
    
    async def speech_to_text(self, audio_tensor: torch.Tensor) -> str:
        """Convert speech to text using Sonar"""
        if self.speech_model is None:
            logger.error("Speech model not initialized")
            return ""
        
        try:
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import tempfile
                import soundfile as sf
                import os
                
                # Convert tensor to numpy array for pipeline
                audio_array = audio_tensor.squeeze().cpu().numpy()
                
                # Create temporary audio file (required by pipeline) - optimized
                temp_filename = None
                try:
                    # Use faster temporary file creation
                    fd, temp_filename = tempfile.mkstemp(suffix='.wav')
                    os.close(fd)  # Close file descriptor immediately
                    
                    # Write audio file with minimal overhead
                    sf.write(temp_filename, audio_array, self.config.sample_rate, format='WAV', subtype='PCM_16')
                    
                    # Use pipeline predict method with correct target_lang
                    text_output = self.speech_model.predict([temp_filename], target_lang="eng_Latn")
                finally:
                    # Clean up temporary file
                    if temp_filename and os.path.exists(temp_filename):
                        try:
                            os.unlink(temp_filename)
                        except OSError:
                            pass
                
                if text_output and len(text_output) > 0:
                    result = text_output[0]
                    if isinstance(result, str):
                        return result.strip()
                    elif hasattr(result, 'text'):
                        return result.text.strip()
                    else:
                        return str(result).strip()
                else:
                    return ""
                
        except Exception as e:
            logger.error(f"Error in speech-to-text: {e}")
            return ""
    
    async def translate_text(self, text: str, target_lang: str) -> str:
        """Translate text using Sonar and enhance with Gemini"""
        try:
            if not text.strip():
                return ""
            
            # Check cache first (LRU implementation)
            cache_key = f"{text}_{target_lang}"
            if cache_key in self.translation_cache:
                # Move to end (most recently used)
                value = self.translation_cache.pop(cache_key)
                self.translation_cache[cache_key] = value
                return value
            
            # Sonar translation
            if self.text_model is None:
                logger.error("Text model not initialized")
                return text
            
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Map short language codes to full language-script codes for text models
                lang_code_map = {
                    "eng": "eng_Latn",
                    "hin": "hin_Deva", 
                    "ben": "ben_Beng"
                }
                
                source_lang_full = lang_code_map.get(self.config.source_lang, self.config.source_lang)
                target_lang_full = lang_code_map.get(target_lang, target_lang)
                
                # Use pipeline predict method with source and target languages
                sonar_output = self.text_model.predict([text], source_lang=source_lang_full, target_lang=target_lang_full)
                
                if sonar_output and len(sonar_output) > 0:
                    result = sonar_output[0]
                    if isinstance(result, str):
                        sonar_translation = result
                    elif hasattr(result, 'text'):
                        sonar_translation = result.text
                    else:
                        sonar_translation = str(result)
                else:
                    sonar_translation = text
            
            if not sonar_translation:
                logger.warning(f"Sonar translation returned empty result for: {text}")
                sonar_translation = text
            
            # Enhance with Gemini if available - with timeout and optimized prompts
            if self.gemini_model:
                try:
                    lang_names = {
                        "hin": "Hindi",
                        "ben": "Bengali"
                    }
                    
                    # Shorter, more efficient prompt
                    prompt = f"Improve this {lang_names.get(target_lang, target_lang)} translation: {sonar_translation}. Only return the improved text."
                    
                    # Use sync call with timeout
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Gemini API timeout")
                    
                    # Set up timeout
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(self.config.gemini_timeout))
                    
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            response = self.gemini_model.generate_content(prompt)
                            enhanced_translation = response.text.strip()
                    finally:
                        signal.alarm(0)  # Cancel timeout
                    
                    # Use enhanced translation if it's valid
                    if enhanced_translation and len(enhanced_translation) > 0:
                        final_translation = enhanced_translation
                    else:
                        final_translation = sonar_translation
                        
                except (TimeoutError, Exception) as e:
                    # Only log non-quota/timeout errors to reduce noise
                    if not any(x in str(e).lower() for x in ["quota", "rate limit", "timeout"]):
                        logger.warning(f"Gemini enhancement failed: {e}")
                    final_translation = sonar_translation
            else:
                final_translation = sonar_translation
            
            # Cache the result with LRU eviction
            if len(self.translation_cache) >= self.max_cache_size:
                # Remove oldest item
                self.translation_cache.popitem(last=False)
            self.translation_cache[cache_key] = final_translation
            
            return final_translation
            
        except Exception as e:
            logger.error(f"Error in text translation: {e}")
            return text  # Return original text as fallback
    
    async def process_audio_chunk(self, audio_chunk: np.ndarray) -> Dict[str, str]:
        """Process a single audio chunk and return translations"""
        try:
            # Preprocess audio
            audio_tensor = self.preprocess_audio(audio_chunk)
            
            # Speech to text
            transcribed_text = await self.speech_to_text(audio_tensor)
            
            if not transcribed_text.strip():
                return {"original": "", "translations": {}}
            
            # Translate to all target languages
            translations = {}
            for target_lang in self.config.target_langs:
                translation = await self.translate_text(transcribed_text, target_lang)
                translations[target_lang] = translation
            
            return {
                "original": transcribed_text,
                "translations": translations
            }
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {"original": "", "translations": {}}
    
    def add_audio_data(self, new_audio: np.ndarray):
        """Add new audio data to the buffer"""
        self.audio_buffer = np.concatenate([self.audio_buffer, new_audio])
        
        # Keep only recent audio (prevent memory overflow) - reduced buffer size
        max_buffer_length = int(self.config.sample_rate * 6)  # 6 seconds - smaller buffer
        if len(self.audio_buffer) > max_buffer_length:
            self.audio_buffer = self.audio_buffer[-max_buffer_length:]
    
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get the next audio chunk for processing"""
        chunk_samples = int(self.config.sample_rate * self.config.chunk_duration)
        
        if len(self.audio_buffer) >= chunk_samples:
            chunk = self.audio_buffer[:chunk_samples]
            
            # Remove processed samples (keeping overlap)
            overlap_samples = int(self.config.sample_rate * self.config.overlap_duration)
            self.audio_buffer = self.audio_buffer[chunk_samples - overlap_samples:]
            
            return chunk
        
        return None
    
    async def cleanup(self):
        """Cleanup resources"""
        self.translation_cache.clear()
        self.audio_buffer = np.array([])
        
        # Models are managed by pipelines, no explicit cleanup needed
        # The pipelines handle their own resource management
        
        logger.info("Cleanup completed")

# Example usage
if __name__ == "__main__":
    async def main():
        # Configuration
        config = TranslationConfig(
            gemini_api_key="YOUR_GEMINI_API_KEY",  # Replace with actual API key
            device="cpu",  # Change to "cuda" if GPU available
            target_langs=["hin", "ben"]
        )
        
        # Initialize translator
        translator = SonarRealtimeTranslator(config)
        await translator.initialize_models()
        
        # Example audio processing (you would replace this with real audio input)
        sample_rate = 16000
        duration = 3.0
        dummy_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
        
        result = await translator.process_audio_chunk(dummy_audio)
        print(f"Original: {result['original']}")
        for lang, translation in result['translations'].items():
            print(f"{lang}: {translation}")
        
        await translator.cleanup()
    
    # Run the example
    asyncio.run(main())

