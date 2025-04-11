# Use official Python runtime as base image
FROM slabstech/dhwani-api-server-base

WORKDIR /app

# Copy application code
COPY . .


# Expose port from settings
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:7860/v1/health || exit 1

# Command to run the application
CMD ["python", "/app/src/server/main.py", "--host", "0.0.0.0", "--port", "7860"]