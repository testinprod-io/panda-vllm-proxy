FROM grafana/alloy:v1.9.1

RUN mkdir -p /etc/alloy

COPY config.alloy /etc/alloy/config.alloy

CMD ["run", "/etc/alloy/config.alloy"]

EXPOSE 12345
