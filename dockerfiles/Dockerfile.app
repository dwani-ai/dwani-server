FROM slabstech/dhwani-server-base
WORKDIR /app

ENV HF_HOME=/data/huggingface
COPY . .

# Set up user
RUN useradd -ms /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser
ENV HF_HOME=/data/huggingface
# Expose port
EXPOSE 7860

# Start the server
CMD ["python", "/app/src/server/main.py", "--host", "0.0.0.0", "--port", "7860", "--config", "config_two"]