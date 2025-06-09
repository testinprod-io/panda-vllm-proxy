FROM rust:1.86 AS builder
WORKDIR /app
#RUN git clone --depth 1 --branch v0.5.0 https://github.com/Dstack-TEE/dstack.git .
RUN git clone --depth 1 --branch tei/certbot-cli-support-multi-domains https://github.com/testinprod-io/dstack.git .
RUN cd certbot/cli && cargo build --release --target-dir /app/target

FROM nginx:1.27.5

RUN apt-get update && apt-get install gettext xxd jq -y

COPY --from=builder /app/target/release/certbot /usr/local/bin/certbot

COPY ./nginx.conf.template /etc/nginx/nginx.conf.template
COPY ./certbot.toml.template /etc/certbot/certbot.toml.template
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ADD https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

ENTRYPOINT ["/entrypoint.sh"]
