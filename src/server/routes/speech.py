# routes/speech.py
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Query, Depends
from starlette.responses import StreamingResponse
import io
import tempfile
import torch
import torchaudio
import soundfile as sf
import numpy as np
from logging_config import logger
from models.schemas import TranscriptionResponse, ChatRequest
from config.constants import LANGUAGE_TO_SCRIPT
from utils.tts_utils import load_audio_from_url, synthesize_speech, SynthesizeRequest, KannadaSynthesizeRequest, EXAMPLES
from .chat import indic_chat
from core.dependencies import get_tts_manager, get_asr_manager, get_llm_manager, get_model_manager, get_settings

router = APIRouter(prefix="/v1", tags=["speech"])

@router.post("/audio/speech", response_class=StreamingResponse)
async def synthesize_kannada(
    request: KannadaSynthesizeRequest,
    tts_manager=Depends(get_tts_manager)
):
    if not tts_manager.model:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    kannada_example = next(ex for ex in EXAMPLES if ex["audio_name"] == "KAN_F (Happy)")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text to synthesize cannot be empty.")
    
    audio_buffer = synthesize_speech(
        tts_manager,
        text=request.text,
        ref_audio_name="KAN_F (Happy)",
        ref_text=kannada_example["ref_text"]
    )
    
    return StreamingResponse(
        audio_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=synthesized_kannada_speech.wav"}
    )

@router.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Query(...),
    asr_manager=Depends(get_asr_manager)
):
    if not asr_manager.model:
        raise HTTPException(status_code=503, detail="ASR model not loaded")
    valid_languages = list(asr_manager.model_language.keys())
    if language not in valid_languages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language. Supported languages: {', '.join(valid_languages)}"
        )
    try:
        wav, sr = torchaudio.load(file.file)
        wav = torch.mean(wav, dim=0, keepdim=True)
        target_sample_rate = 16000
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
            wav = resampler(wav)
        transcription_rnnt = asr_manager.model(wav, asr_manager.model_language[language], "rnnt")
        return TranscriptionResponse(text=transcription_rnnt)
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@router.post("/speech_to_speech")
async def speech_to_speech(
    request: Request,
    file: UploadFile = File(...),
    language: str = Query(...),
    tts_manager=Depends(get_tts_manager),
    asr_manager=Depends(get_asr_manager),
    llm_manager=Depends(get_llm_manager),
    model_manager=Depends(get_model_manager),
    settings=Depends(get_settings)
) -> StreamingResponse:
    if not tts_manager.model:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    valid_languages = list(asr_manager.model_language.keys())
    if language not in valid_languages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language. Supported languages: {', '.join(valid_languages)}"
        )
    transcription = await transcribe_audio(file, language, asr_manager)
    logger.debug(f"Transcribed text: {transcription.text}")

    chat_request = ChatRequest(
        prompt=transcription.text,
        src_lang=LANGUAGE_TO_SCRIPT.get(language, "kan_Knda"),
        tgt_lang=LANGUAGE_TO_SCRIPT.get(language, "kan_Knda")
    )
    processed_text = await indic_chat(request, chat_request, llm_manager, model_manager, settings)
    logger.debug(f"Processed text: {processed_text.response}")

    voice_request = KannadaSynthesizeRequest(text=processed_text.response)
    audio_response = await synthesize_kannada(voice_request, tts_manager)
    return audio_response