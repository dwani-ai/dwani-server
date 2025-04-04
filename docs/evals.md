ASR 

curl -X 'POST' \
  'https://slabstech-dhwani-internal-api-server.hf.space/transcribe/?language=kannada' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@kannada_sample_1.wav;type=audio/x-wav'


{
  "text": "ಕರ್ನಾಟಕ ದ ರಾಜಧಾನಿ ಯಾವುದು"
}



CHAT 

curl -X 'POST' \
  'https://slabstech-dhwani-internal-api-server.hf.space/v1/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "hello",
  "src_lang": "kan_Knda",
  "tgt_lang": "kan_Knda"
}'

{
  "response": "ನಮಸ್ತೇ! ಭಾರತಕ್ಕೆ, ವಿಶೇಷವಾಗಿ ಇಂದಿನ ಕರ್ನಾಟಕಕ್ಕೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿಯೊಂದಿಗೆ ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ? "
}



TTS

curl -X 'POST' \
  'https://slabstech-dhwani-internal-api-server.hf.space/v1/audio/speech' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": "ನಿಮ್ಮ ಇನ್‌ಪುಟ್ ಪಠ್ಯವನ್ನು ಇಲ್ಲಿ ಸೇರಿಸಿ",
  "voice": "Anu speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality",
  "model": "ai4bharat/indic-parler-tts",
  "response_format": "mp3",
  "speed": 1
}' -o test.mp3

-
Speech to speech

curl -X 'POST' \
  'https://slabstech-dhwani-internal-api-server.hf.space/v1/speech_to_speech?language=kannada' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@kannada_sample_1.wav;type=audio/x-wav' \
  -F 'voice=Anu speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality'  -o speech-output.mp3