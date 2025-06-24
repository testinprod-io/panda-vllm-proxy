#!/usr/bin/env bash
set -e

if [ -z "$MODEL_ID" ]; then
  echo "Error: Please set MODEL_ID environment variable."
  exit 1
fi

mkdir -p "$TOKENIZER_DIR"

for FILE in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json config.json chat_template.jinja; do
  curl -L "https://huggingface.co/$MODEL_ID/resolve/main/$FILE" -o "$TOKENIZER_DIR/$FILE"
done

if [ -n "$NEW_LENGTH" ]; then
  jq ".model_max_length = $NEW_LENGTH" "$TOKENIZER_DIR/tokenizer_config.json" > tmp && mv tmp "$TOKENIZER_DIR/tokenizer_config.json"
  echo "Override model_max_length: $NEW_LENGTH"
fi
