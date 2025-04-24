from time import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.responses import StreamingResponse
import io
from PIL import Image
import torchaudio
from logging_config import logger
from settings import ChatRequest, ChatResponse, TranslationRequest, TranslationResponse, TranscriptionResponse, KannadaSynthesizeRequest, device
from managers.llm_manager import LLMManager
from managers.tts_manager import TTSManager
from managers.asr_manager import ASRModelManager
from managers.translate_manager import TranslateManager, ModelManager
from utils.tts_utils import synthesize_speech, EXAMPLES
from utils.translation_utils import perform_internal_translation, SUPPORTED_LANGUAGES

# Global managers (set in main.py)
llm_manager: LLMManager = None
tts_manager: TTSManager = None
asr_manager: ASRModelManager = None
model_manager: ModelManager = None

def set_global_managers(llm: LLMManager, tts: TTSManager, asr: ASRModelManager, model: ModelManager):
    global llm_manager, tts_manager, asr_manager, model_manager
    llm_manager = llm
    tts_manager = tts
    asr_manager = asr
    model_manager = model
    logger.info("Global managers set successfully")

# Translation configs
translation_configs = []

# FastAPI App
app = FastAPI(
    title="Dhwani API",
    description="AI Chat API supporting Indian languages",
    version="1.0.0",
    redirect_slashes=False
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Timing Middleware
@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    start_time = time()
    response = await call_next(request)
    end_time = time()
    duration = end_time - start_time
    logger.info(f"Request to {request.url.path} took {duration:.3f} seconds")
    response.headers["X-Response-Time"] = f"{duration:.3f}"
    return response

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Dependency
def get_translate_manager(src_lang: str, tgt_lang: str) -> TranslateManager:
    return model_manager.get_model(src_lang, tgt_lang)

# Lifespan Event Handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    def load_all_models():
        logger.info("Entering load_all_models function")
        if llm_manager is None or tts_manager is None or asr_manager is None or model_manager is None:
            logger.error("One or more global managers are None")
            raise ValueError("Global managers not initialized")
        
        try:
            logger.info("Loading LLM model...")
            llm_manager.load()
            logger.info(f"LLM model {llm_manager.model_name} loaded successfully on cuda:0")

            logger.info("Loading TTS model...")
            tts_manager.load()
            logger.info(f"TTS model {tts_manager.repo_id} loaded successfully on cuda:0")

            logger.info("Loading ASR model...")
            asr_manager.load()
            logger.info("ASR model ai4bharat/indic-conformer-600m-multilingual loaded successfully on cuda:0")

            translation_tasks = [
                ('eng_Latn', 'kan_Knda', 'eng_indic'),
                ('kan_Knda', 'eng_Latn', 'indic_eng'),
                ('kan_Knda', 'hin_Deva', 'indic_indic'),
            ]
            
            logger.info(f"Translation configs: {translation_configs}")
            for config in translation_configs:
                src_lang = config["src_lang"]
                tgt_lang = config["tgt_lang"]
                key = model_manager._get_model_key(src_lang, tgt_lang)
                translation_tasks.append((src_lang, tgt_lang, key))

            for src_lang, tgt_lang, key in translation_tasks:
                logger.info(f"Loading translation model for {src_lang} -> {tgt_lang} (key: {key})...")
                try:
                    model_manager.load_model(src_lang, tgt_lang, key)
                    logger.info(f"Translation model for {key} loaded successfully on cuda:0")
                except Exception as e:
                    logger.error(f"Failed to load translation model for {key}: {str(e)}")
                    raise

            logger.info("All models loaded successfully during startup")
        except Exception as e:
            logger.error(f"Critical error loading models during startup: {str(e)}")
            raise

    logger.info("Starting FastAPI lifespan handler...")
    load_all_models()
    logger.info("Lifespan startup complete, yielding to FastAPI...")
    yield
    logger.info("Unloading LLM model during shutdown...")
    llm_manager.unload()
    logger.info("Server shutdown complete")

app.lifespan = lifespan

# API Endpoints
@app.post("/v1/audio/speech", response_class=StreamingResponse)
async def synthesize_kannada(request: KannadaSynthesizeRequest):
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

@app.get("/v1/health")
async def health_check():
    from settings import settings
    return {"status": "healthy", "model": settings.llm_model_name}

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/v1/unload_all_models")
async def unload_all_models():
    try:
        logger.info("Starting to unload all models...")
        llm_manager.unload()
        tts_manager.model = None
        asr_manager.model = None
        model_manager.models.clear()
        logger.info("All models unloaded successfully")
        return {"status": "success", "message": "All models unloaded"}
    except Exception as e:
        logger.error(f"Error unloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload models: {str(e)}")

@app.post("/v1/load_all_models")
async def load_all_models_endpoint():
    try:
        logger.info("Starting to reload all models...")
        load_all_models()
        logger.info("All models reloaded successfully")
        return {"status": "success", "message": "All models reloaded"}
    except Exception as e:
        logger.error(f"Error reloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload models: {str(e)}")

@app.post("/v1/translate", response_model=TranslationResponse)
async def translate_endpoint(request: TranslationRequest):
    logger.info(f"Received translation request: {request.dict()}")
    try:
        translations = await perform_internal_translation(
            sentences=request.sentences,
            src_lang=request.src_lang,
            tgt_lang=request.tgt_lang,
            model_manager=model_manager
        )
        logger.info(f"Translation successful: {translations}")
        return TranslationResponse(translations=translations)
    except Exception as e:
        logger.error(f"Unexpected error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/v1/chat", response_model=ChatResponse)
@limiter.limit("100/minute")
async def chat(request: Request, chat_request: ChatRequest):
    if not chat_request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    logger.info(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}, tgt_lang: {chat_request.tgt_lang}")
    
    EUROPEAN_LANGUAGES = {"deu_Latn", "fra_Latn", "nld_Latn", "spa_Latn", "ita_Latn", "por_Latn", "rus_Cyrl", "pol_Latn"}
    
    try:
        if chat_request.src_lang != "eng_Latn" and chat_request.src_lang not in EUROPEAN_LANGUAGES:
            translated_prompt = await perform_internal_translation(
                sentences=[chat_request.prompt],
                src_lang=chat_request.src_lang,
                tgt_lang="eng_Latn",
                model_manager=model_manager
            )
            prompt_to_process = translated_prompt[0]
            logger.info(f"Translated prompt to English: {prompt_to_process}")
        else:
            prompt_to_process = chat_request.prompt
            logger.info("Prompt in English or European language, no translation needed")

        from settings import settings
        response = await llm_manager.generate(prompt_to_process, settings.max_tokens)
        logger.info(f"Generated response: {response}")

        if chat_request.tgt_lang != "eng_Latn" and chat_request.tgt_lang not in EUROPEAN_LANGUAGES:
            translated_response = await perform_internal_translation(
                sentences=[response],
                src_lang="eng_Latn",
                tgt_lang=chat_request.tgt_lang,
                model_manager=model_manager
            )
            final_response = translated_response[0]
            logger.info(f"Translated response to {chat_request.tgt_lang}: {final_response}")
        else:
            final_response = response
            logger.info(f"Response in {chat_request.tgt_lang}, no translation needed")

        return ChatResponse(response=final_response)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/v1/visual_query/")
async def visual_query(
    file: UploadFile = File(...),
    query: str = Body(...),
    src_lang: str = Query("kan_Knda", enum=list(SUPPORTED_LANGUAGES)),
    tgt_lang: str = Query("kan_Knda", enum=list(SUPPORTED_LANGUAGES)),
):
    try:
        image = Image.open(file.file)
        if image.size == (0, 0):
            raise HTTPException(status_code=400, detail="Uploaded image is empty or invalid")

        if src_lang != "eng_Latn":
            translated_query = await perform_internal_translation(
                sentences=[query],
                src_lang=src_lang,
                tgt_lang="eng_Latn",
                model_manager=model_manager
            )
            query_to_process = translated_query[0]
            logger.info(f"Translated query to English: {query_to_process}")
        else:
            query_to_process = query
            logger.info("Query already in English, no translation needed")

        answer = await llm_manager.vision_query(image, query_to_process)
        logger.info("generated English answer: {answer}")

        if tgt_lang != "eng_Latn":
            translated_answer = await perform_internal_translation(
                sentences=[answer],
                src_lang="eng_Latn",
                tgt_lang=tgt_lang,
                model_manager=model_manager
            )
            final_answer = translated_answer[0]
            logger.info(f"Translated answer to {tgt_lang}: {final_answer}")
        else:
            final_answer = answer
            logger.info("Answer kept in English, no translation needed")

        return {"answer": final_answer}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/v1/chat_v2", response_model=ChatResponse)
@limiter.limit("100/minute")
async def chat_v2(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile = File(default=None),
    src_lang: str = Form("kan_Knda"),
    tgt_lang: str = Form("kan_Knda"),
):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if src_lang not in SUPPORTED_LANGUAGES or tgt_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language code. Supported codes: {', '.join(SUPPORTED_LANGUAGES)}")

    logger.info(f"Received prompt: {prompt}, src_lang: {src_lang}, tgt_lang: {tgt_lang}, Image provided: {image is not None}")

    try:
        if image:
            image_data = await image.read()
            if not image_data:
                raise HTTPException(status_code=400, detail="Uploaded image is empty")
            img = Image.open(io.BytesIO(image_data))

            if src_lang != "eng_Latn":
                translated_prompt = await perform_internal_translation(
                    sentences=[prompt],
                    src_lang=src_lang,
                    tgt_lang="eng_Latn",
                    model_manager=model_manager
                )
                prompt_to_process = translated_prompt[0]
                logger.info(f"Translated prompt to English: {prompt_to_process}")
            else:
                prompt_to_process = prompt
                logger.info("Prompt already in English, no translation needed")

            decoded = await llm_manager.chat_v2(img, prompt_to_process)
            logger.info(f"Generated English response: {decoded}")

            if tgt_lang != "eng_Latn":
                translated_response = await perform_internal_translation(
                    sentences=[decoded],
                    src_lang="eng_Latn",
                    tgt_lang=tgt_lang,
                    model_manager=model_manager
                )
                final_response = translated_response[0]
                logger.info(f"Translated response to {tgt_lang}: {final_response}")
            else:
                final_response = decoded
                logger.info("Response kept in English, no translation needed")
        else:
            if src_lang != "eng_Latn":
                translated_prompt = await perform_internal_translation(
                    sentences=[prompt],
                    src_lang=src_lang,
                    tgt_lang="eng_Latn",
                    model_manager=model_manager
                )
                prompt_to_process = translated_prompt[0]
                logger.info(f"Translated prompt to English: {prompt_to_process}")
            else:
                prompt_to_process = prompt
                logger.info("Prompt already in English, no translation needed")

            from settings import settings
            decoded = await llm_manager.generate(prompt_to_process, settings.max_tokens)
            logger.info(f"Generated English response: {decoded}")

            if tgt_lang != "eng_Latn":
                translated_response = await perform_internal_translation(
                    sentences=[decoded],
                    src_lang="eng_Latn",
                    tgt_lang=tgt_lang,
                    model_manager=model_manager
                )
                final_response = translated_response[0]
                logger.info(f"Translated response to {tgt_lang}: {final_response}")
            else:
                final_response = decoded
                logger.info("Response kept in English, no translation needed")

        return ChatResponse(response=final_response)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/v1/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...), language: str = Query(...)):
    if not asr_manager.model:
        raise HTTPException(status_code=503, detail="ASR model not loaded")
    if language not in asr_manager.model_language:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}. Supported languages: {list(asr_manager.model_language.keys())}")
    try:
        wav, sr = torchaudio.load(file.file)
        wav = torch.mean(wav, dim=0, keepdim=True).to(device)
        target_sample_rate = 16000
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate).to(device)
            wav = resampler(wav)
        transcription_rnnt = asr_manager.model(wav, asr_manager.model_language[language], "rnnt")
        return TranscriptionResponse(text=transcription_rnnt)
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/v1/speech_to_speech")
async def speech_to_speech(
    request: Request,
    file: UploadFile = File(...),
    language: str = Query(...),
) -> StreamingResponse:
    if not tts_manager.model:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    if language not in asr_manager.model_language:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}. Supported languages: {list(asr_manager.model_language.keys())}")
    transcription = await transcribe_audio(file, language)
    logger.info(f"Transcribed text: {transcription.text}")

    chat_request = ChatRequest(
        prompt=transcription.text,
        src_lang=LANGUAGE_TO_SCRIPT.get(language, "kan_Knda"),
        tgt_lang=LANGUAGE_TO_SCRIPT.get(language, "kan_Knda")
    )
    processed_text = await chat(request, chat_request)
    logger.info(f"Processed text: {processed_text.response}")

    voice_request = KannadaSynthesizeRequest(text=processed_text.response)
    audio_response = await synthesize_kannada(voice_request)
    return audio_response

LANGUAGE_TO_SCRIPT = {
    "kannada": "kan_Knda"
}