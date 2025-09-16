#!/bin/bash

# Content-Pal Complete Setup and Start Script
# This script handles installation and startup process

set -e  # Exit on any error

echo "Content-Pal Setup & Start"
echo "========================="

# Check if running from project root
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo "Error: Please run this script from the Content-Pal project root directory"
    exit 1
fi

echo "Checking prerequisites..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "Python $PYTHON_VERSION found"

# Check pipenv
if ! command -v pipenv &> /dev/null; then
    echo "Installing pipenv..."
    pip3 install pipenv
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is required but not installed"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Error: Docker Compose is required but not installed"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running. Please start Docker and try again."
    exit 1
fi

echo "Prerequisites check complete"

# Environment setup
echo ""
echo "Environment Configuration"
echo "========================"

if [ ! -f ".env" ]; then
    echo "Error: .env file not found"
    echo "Please copy .env_template to .env and configure your settings:"
    echo "  cp .env_template .env"
    echo "  # Edit .env file with your OPENAI_API_KEY and other settings"
    exit 1
fi

# Validate .env file
source .env
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "<openai_api_key_here>" ]; then
    echo "Error: Please set OPENAI_API_KEY in .env file"
    exit 1
fi
echo "Environment configuration validated"

# Python dependencies
echo ""
echo "Installing Dependencies"
echo "======================"

echo "Installing Python dependencies..."
pipenv install
pipenv install --dev

# Data file check
DATA_FILE="data/netflix_titles_enriched_full.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "Warning: Data file not found at $DATA_FILE"
    echo "Please ensure the Netflix dataset is available"
else
    echo "Data file found"
fi

# Stop any existing services
echo ""
echo "Starting Services"
echo "================"

echo "Stopping any existing services..."
./stop_all.sh 2>/dev/null || true
sleep 3

echo "Starting Docker services..."
docker-compose up -d

echo "Waiting for services to initialize..."
sleep 15

# Database initialization
echo "Checking database initialization..."

# Check if database tables already exist using Python
DB_CHECK_RESULT=$(pipenv run python -c "
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

try:
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        database=os.getenv('POSTGRES_DB', 'content_pal'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        sslmode='prefer',
        gssencmode='disable'
    )
    with conn.cursor() as cur:
        cur.execute(\"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'conversations')\")
        exists = cur.fetchone()[0]
        print('exists' if exists else 'not_exists')
    conn.close()
except Exception:
    print('not_exists')
" 2>/dev/null)

if [ "$DB_CHECK_RESULT" = "exists" ]; then
    echo "Database tables already exist, skipping initialization"
else
    echo "Initializing database tables..."
    pipenv run python -m src.modules.workflow.db_prep
fi

# Wait for API to be ready
echo "Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:${APP_PORT:-5001}/health &>/dev/null; then
        echo "API is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Warning: API may not be fully ready yet"
    fi
    sleep 2
done

# Start Streamlit
echo "Starting Streamlit UI..."
nohup pipenv run streamlit run src/modules/ui/streamlit_app.py --server.port=8501 --server.headless=true > streamlit.log 2>&1 &
STREAMLIT_PID=$!
sleep 5

echo ""
echo "Setup Complete!"
echo "==============="
echo ""
echo "Access URLs:"
echo "  Streamlit UI:    http://localhost:8501"
echo "  Grafana:         http://localhost:3000"
echo "  API Backend:     http://localhost:${APP_PORT:-5001}"
echo ""
echo "Useful Commands:"
echo "  View API logs:        docker-compose logs -f app"
echo "  View Streamlit logs:  tail -f streamlit.log"
echo "  Stop all services:    ./stop_all.sh"
echo "  Test API:             pipenv run python -m src.modules.workflow.test"
echo ""
echo "Open http://localhost:8501 to start using Content-Pal"

# Save PID for stop script
echo $STREAMLIT_PID > .streamlit_pid