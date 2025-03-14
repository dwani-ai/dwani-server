from transformers import AutoModel
import numpy as np
import soundfile as sf

# Load INF5 from Hugging Face
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# Generate speech
audio = model(
    "ಭರತನು ದಶರಥನ ಎರಡನೆಯ ಮಗ. ರಾಮನು ಸೀತಾ, ಲಕ್ಷ್ಮಣ ರೊಡನೆ ವನವಾಸಕ್ಕೆ ಹೊರಟಾಗ ಭರತ ಇರುವುದಿಲ್ಲ. ತನ್ನ ತಾಯಿಯೇ ರಾಮನನ್ನು ವನವಾಸಕ್ಕೆ ಕಳಿಸುವುದರ ಮೂಲಕ,",
    ref_audio_path="kannada_female_voice.wav",
    ref_text="ರಾಮ ರಾಮಾಯಣದ ನಾಯಕ. ರಾಮನನ್ನು ದೇವರ ಅವತಾರವೆಂದು ಚಿತ್ರಿಸಲಾಗಿದೆ. ರಾಮನು ಅಯೋಧ್ಯೆಯ ಸೂರ್ಯ ವಂಶದ ರಾಜನಾದ ದಶರಥನ ಹಿರಿಯ ಮಗ"
)

# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
sf.write("namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)

#    "ಭರತನು ದಶರಥನ ಎರಡನೆಯ ಮಗ. ರಾಮನು ಸೀತಾ, ಲಕ್ಷ್ಮಣ ರೊಡನೆ ವನವಾಸಕ್ಕೆ ಹೊರಟಾಗ ಭರತ ಇರುವುದಿಲ್ಲ. ತನ್ನ ತಾಯಿಯೇ ರಾಮನನ್ನು ವನವಾಸಕ್ಕೆ ಕಳಿಸುವುದರ ಮೂಲಕ, ತನ್ನ ತಂದೆ ದಶರಥನ ಸಾವಿಗೆ ಕಾರಣಳಾದ ವಿಷಯ ಭರತನಿಗೆ ನಂತರ ತಿಳಿಯುತ್ತದೆ. ಕೂಡಲೇ ತಾಯಿಯ ಮೇಲೆ ಕೋಪಗೊಂಡು ರಾಮನನ್ನು ಹುಡುಕಲು ಹೊರಡುತ್ತಾನೆ. ಭರತ ಎಷ್ಟೇ ವಿನಂತಿಸಿಕೊಂಡರೂ, ತನ್ನ ತಂದೆಗೆ ಕೊಟ್ಟ ಮಾತಿಗೆ ತಪ್ಪ್ಪಲು ಒಪ್ಪದ ರಾಮ ಭರತನೊಡನೆ ಹಿಂತಿರುಗಲು ಒಪ್ಪುವುದಿಲ್ಲ. ಆಗ ಭರತ ರಾಮನ ಪಾದುಕೆಗಳನ್ನು ಪಡೆದುಕೊಂಡು ಹಿಂತಿರುಗುತ್ತಾನೆ. ತಾನು ಸಿಂಹಾಸನದ ಮೇಲೆ ಕುಳಿತುಕೊಳ್ಳದೆ, ಅಣ್ಣನ ಪಾದುಕೆಗಳನ್ನೇ ಸಿಂಹಾಸನದ ಮೇಲಿಟ್ಟು, ರಾಮನ ಪರವಾಗಿ ರಾಜ್ಯದ ಆಡಳಿತವನ್ನು ನಿರ್ವಹಿಸುತ್ತಿರುತ್ತಾನೆ.",
