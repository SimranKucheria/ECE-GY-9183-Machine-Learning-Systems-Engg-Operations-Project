FROM prom/prometheus:latest

COPY prometheus/prometheus.yml /etc/prometheus/prometheus.yml

EXPOSE 9090

CMD ["--config.file=/etc/prometheus/prometheus.yml"]
