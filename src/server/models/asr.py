import torch
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import os
import shutil
from config.logging_config import logger

class ASRManager:
    def __init__(self, default_language: str = "kn", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.default_language = default_language
        self.device = torch.device(device)
        self.model = None
        self.is_loaded = False
        self.model_language = {
            "kannada": "kn", "hindi": "hi", "malayalam": "ml", "assamese": "as", "bengali": "bn",
            "bodo": "brx", "dogri": "doi", "gujarati": "gu", "kashmiri": "ks", "konkani": "kok",
            "maithili": "mai", "manipuri": "mni", "marathi": "mr", "nepali": "ne", "odia": "or",
            "punjabi": "pa", "sanskrit": "sa", "santali": "sat", "sindhi": "sd", "tamil": "ta",
            "telugu": "te", "urdu": "ur"
        }
        self.config_models = {
            "as": "ai4bharat/indicconformer_stt_as_hybrid_rnnt_large",
            "bn": "ai4bharat/indicconformer_stt_bn_hybrid_rnnt_large",
            "brx": "ai4bharat/indicconformer_stt_brx_hybrid_rnnt_large",
            "doi": "ai4bharat/indicconformer_stt_doi_hybrid_rnnt_large",
            "gu": "ai4bharat/indicconformer_stt_gu_hybrid_rnnt_large",
            "hi": "ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large",
            "kn": "ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large",
            "ks": "ai4bharat/indicconformer_stt_ks_hybrid_rnnt_large",
            "kok": "ai4bharat/indicconformer_stt_kok_hybrid_rnnt_large",
            "mai": "ai4bharat/indicconformer_stt_mai_hybrid_rnnt_large",
            "ml": "ai4bharat/indicconformer_stt_ml_hybrid_rnnt_large",
            "mni": "ai4bharat/indicconformer_stt_mni_hybrid_rnnt_large",
            "mr": "ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large",
            "ne": "ai4bharat/indicconformer_stt_ne_hybrid_rnnt_large",
            "or": "ai4bharat/indicconformer_stt_or_hybrid_rnnt_large",
            "pa": "ai4bharat/indicconformer_stt_pa_hybrid_rnnt_large",
            "sa": "ai4bharat/indicconformer_stt_sa_hybrid_rnnt_large",
            "sat": "ai4bharat/indicconformer_stt_sat_hybrid_rnnt_large",
            "sd": "ai4bharat/indicconformer_stt_sd_hybrid_rnnt_large",
            "ta": "ai4bharat/indicconformer_stt_ta_hybrid_rnnt_large",
            "te": "ai4bharat/indicconformer_stt_te_hybrid_rnnt_large",
            "ur": "ai4bharat/indicconformer_stt_ur_hybrid_rnnt_large"
        }

    def unload(self):
        if self.is_loaded:
            # Delete the model and processor to free memory
            del self.model
            del self.processor
            # If using CUDA, clear the cache to free GPU memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.is_loaded = False
            logger.info(f"LLM {self.model_name} unloaded from {self.device}")
    def load(self, language_id: str = None):
        if not self.is_loaded or (language_id and language_id != self.default_language):
            model_name = self.config_models.get(language_id or self.default_language, self.config_models["kn"])
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name)
            self.model.freeze()  # Inference mode
            self.model = self.model.to(self.device)
            self.default_language = language_id or self.default_language
            self.is_loaded = True
            logger.info(f"ASR model {model_name} loaded for {self.default_language} on {self.device}")

    def split_audio(self, file_path: str, chunk_duration_ms: int = 15000) -> list[str]:
        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        if duration_ms <= chunk_duration_ms:
            return [file_path]
        
        num_chunks = (duration_ms + chunk_duration_ms - 1) // chunk_duration_ms
        chunks = [audio[i * chunk_duration_ms:min((i + 1) * chunk_duration_ms, duration_ms)] for i in range(num_chunks)]
        output_dir = "audio_chunks"
        os.makedirs(output_dir, exist_ok=True)
        chunk_file_paths = []
        for i, chunk in enumerate(chunks):
            chunk_file_path = os.path.join(output_dir, f"chunk_{i}.wav")
            chunk.export(chunk_file_path, format="wav")
            chunk_file_paths.append(chunk_file_path)
        return chunk_file_paths

    def cleanup(self):
        output_dir = "audio_chunks"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    def transcribe(self, file_paths: list[str], language_id: str) -> str:
        if not self.is_loaded or language_id != self.default_language:
            self.load(language_id)
        self.model.cur_decoder = "rnnt"
        transcriptions = []
        for file_path in file_paths:
            rnnt_texts = self.model.transcribe([file_path], batch_size=1, language_id=language_id)[0]
            text = rnnt_texts[0] if isinstance(rnnt_texts, list) and rnnt_texts else rnnt_texts
            transcriptions.append(text)
        return ' '.join(transcriptions)