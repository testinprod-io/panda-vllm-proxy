#!/usr/bin/env bash
set -euo pipefail


required_vars=(
  PANDA_LLM_DOMAIN
  PANDA_LLM_CERT_DOMAINS
  PANDA_CERT_DIR
  PANDA_ACME_URL
  PANDA_CF_API_TOKEN
  PANDA_CF_ZONE_ID
  PANDA_APP_SERVER
  PANDA_APP_SERVER_TOKEN
)

for var in "${required_vars[@]}"; do
  if [ -z "${!var:-}" ]; then
    echo >&2 "Error: \$$var is not set"
    exit 1
  fi
done

env_list=$(printf '$%s ' "${required_vars[@]}")

envsubst "${env_list}" \
  < /etc/nginx/nginx.conf.template \
  > /etc/nginx/nginx.conf

envsubst "${env_list}" \
  < /etc/certbot/certbot.toml.template \
  > /etc/certbot/certbot.toml

echo "Generated config files from env variables"

mkdir -p ${PANDA_CERT_DIR}/live

PANDA_SSL_CERT_PATH=${PANDA_CERT_DIR}/live/cert.pem
PANDA_SSL_PUBKEY_PATH=${PANDA_CERT_DIR}/live/pubkey.pub
PANDA_SSL_KEY_PATH=${PANDA_CERT_DIR}/live/key.pem
PANDA_SSL_EC_KEY_PATH=${PANDA_CERT_DIR}/live/ec_key.pem

openssl ecparam -genkey -name prime256v1 -out "$PANDA_SSL_EC_KEY_PATH"
openssl pkcs8 -topk8 -nocrypt -in "$PANDA_SSL_EC_KEY_PATH" -out "$PANDA_SSL_KEY_PATH"

#openssl ecparam -genkey -name prime256v1 -out "$PANDA_SSL_KEY_PATH"
openssl pkey -in "$PANDA_SSL_KEY_PATH" -pubout -out "$PANDA_SSL_PUBKEY_PATH"

echo "Generated random keys"

HEX_KEY=$(openssl pkey -pubin -in "$PANDA_SSL_PUBKEY_PATH" -outform DER \
  | xxd -p -c 0 | sed -E 's/.*(04[0-9a-f]+)$/\1/')

HEX_KEY_HASH=$(echo -n $HEX_KEY | xxd -r -p | sha256sum | awk '{print $1}')

QUOTE_RESPONSE=$(curl --unix-socket /var/run/dstack.sock "http://localhost/GetQuote?report_data=0x${HEX_KEY_HASH}")
QUOTE_REGISTER_REQUEST=$(echo "$QUOTE_RESPONSE" \
  | jq --arg hex "$HEX_KEY" \
       'del(.report_data) | .public_key = $hex')

curl --fail -X POST "$PANDA_APP_SERVER/admin/tdxQuotes" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: $PANDA_APP_SERVER_TOKEN" \
     -d "$QUOTE_REGISTER_REQUEST"

echo "Registered attestation quote"

certbot renew --once -c /etc/certbot/certbot.toml

exec nginx -c /etc/nginx/nginx.conf -g 'daemon off;'
