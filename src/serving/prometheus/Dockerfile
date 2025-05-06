# Use the official Grafana image as the base
FROM prom/prometheus:latest

# Set up working directories for dashboards and provisioning
RUN mkdir -p /etc/prometheus/

# Copy provisioning info
COPY prometheus/prometheus.yml /etc/prometheus/

# Expose Grafana's default port
EXPOSE 9090

# Start Grafana
ENTRYPOINT ["/run.sh"]