# GOV2DB Israeli Government Decisions Scraper
# Production-ready Docker image with Selenium, Python, and automated daily sync

FROM selenium/standalone-chrome:latest

# Switch to root for setup
USER root

# Install Python 3.11 and required system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3-venv \
    cron \
    tzdata \
    logrotate \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Use the existing seluser from selenium image (UID 1000) or create scraper with different UID
# selenium/standalone-chrome already has 'seluser' with UID 1000
RUN useradd -m -u 1001 -s /bin/bash scraper 2>/dev/null || true

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching optimization)
COPY requirements.txt .

# Install Python dependencies (setuptools needed for distutils on Python 3.14+)
RUN pip3 install --no-cache-dir setuptools && pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY bin/ ./bin/
COPY src/ ./src/
COPY setup.py .
COPY new_tags.md new_departments.md ./

# Install package in editable mode
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/healthcheck \
    && chown -R scraper:scraper /app

# Copy entrypoint, health check, and randomized sync scripts with execute permissions
COPY docker/docker-entrypoint.sh /usr/local/bin/
COPY docker/healthcheck.sh /usr/local/bin/
COPY docker/randomized_sync.sh /app/docker/
RUN chmod 755 /usr/local/bin/docker-entrypoint.sh /usr/local/bin/healthcheck.sh /app/docker/randomized_sync.sh

# Setup cron job for daily sync at 02:00 AM
# Using /etc/cron.d/ format - cron daemon reads these directly
COPY docker/crontab /etc/cron.d/gov2db-scraper
RUN chmod 0644 /etc/cron.d/gov2db-scraper

# Setup logrotate
COPY docker/logrotate.conf /etc/logrotate.d/gov2db
RUN chmod 0644 /etc/logrotate.d/gov2db

# Note: Running as root because cron daemon requires root permissions
# The sync script itself doesn't need root, but cron does

# Expose health check port (optional, for HTTP health endpoint)
EXPOSE 8080

# Environment defaults (override via docker-compose)
ENV TZ=Asia/Jerusalem \
    PYTHONUNBUFFERED=1 \
    SYNC_MODE=daily

# Health check
HEALTHCHECK --interval=1h --timeout=30s --start-period=5m --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Entry point
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["cron"]
