

curl -X 'POST' \
  'https://gaganyatri-translate-indic-server-cpu.hf.space/translate?src_lang=eng_Latn&tgt_lang=kan_Knda' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sentences": [
    "Hello, how are you?", "Good morning!"
  ],
  "src_lang": "eng_Latn",
  "tgt_lang": "kan_Knda"
}'


{
  "translations": [
    "ಹಲೋ, ಹೇಗಿದ್ದೀರಿ? ",
    "ಶುಭೋದಯ! "
  ]
}




curl -X 'POST' \
  'https://gaganyatri-translate-indic-server-cpu.hf.space/translate?src_lang=kan_Knda&tgt_lang=eng_Latn' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sentences": [
    "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?", "ಶುಭೋದಯ!"
  ],
  "src_lang": "kan_Knda",
  "tgt_lang": "eng_Latn"
}'


{
  "translations": [
    "Hello, how are you?",
    "Good morning!"
  ]
}



curl -X 'POST' \
  'https://gaganyatri-translate-indic-server-cpu.hf.space/translate?src_lang=kan_Knda&tgt_lang=hin_Deva' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sentences": [
    "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?", "ಶುಭೋದಯ!"
  ],
  "src_lang": "kan_Knda",
  "tgt_lang": "hin_Deva"
}'


{
  "translations": [
    "हैलो, कैसा लग रहा है? ",
    "गुड मॉर्निंग! "
  ]
}