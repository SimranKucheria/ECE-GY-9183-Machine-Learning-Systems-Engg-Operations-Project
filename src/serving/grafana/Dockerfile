# Use the official Grafana image as the base
FROM grafana/grafana:latest

# Set up working directories for dashboards and provisioning
RUN mkdir -p /var/lib/grafana/dashboards /etc/grafana/provisioning/dashboards /etc/grafana/provisioning/datasources

# Copy dashboards and provisioning configuration files into the image
COPY grafana/dashboards/ /var/lib/grafana/dashboards
COPY grafana/provisioning/dashboards/ /etc/grafana/provisioning/dashboards
COPY grafana/provisioning/datasources/ /etc/grafana/provisioning/datasources

# Set ownership to Grafana user
RUN chown -R grafana:grafana /etc/grafana/provisioning/dashboards /etc/grafana/provisioning/datasources

# Switch back to the Grafana user
USER grafana

# Expose Grafana's default port
EXPOSE 3000

# Start Grafana
ENTRYPOINT ["/run.sh"]