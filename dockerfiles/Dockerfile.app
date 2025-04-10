# Use the pre-built image with models as the base
FROM slabstech/dhwani-server-models:latest
WORKDIR /app

COPY dhwani_config.json .
# Copy application code
COPY . .

# Set up user
RUN useradd -ms /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 7860

# Start the server
CMD ["python", "/app/src/server/main.py", "--host", "0.0.0.0", "--port", "7860", "--config", "config_two"]