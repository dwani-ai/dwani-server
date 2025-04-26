{
  "model": "gemma3-4b-it",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "what is your name ?"
        }
      ]
    }
  ],
  "temperature": 0.7,
  "max_tokens": 200
}



curl -X 'POST' \
  'http://example.com:example-port/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "gemma3-4b-it",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "what is your name ?"
        }
      ]
    }
  ],
  "temperature": 0.7,
  "max_tokens": 200
}'