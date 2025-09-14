#!/bin/bash

echo "Stopping Content-Pal Services..."
echo "=================================="

echo "Stopping Docker services..."
docker-compose down

echo "Stopping Streamlit..."
pkill -f "pipenv run streamlit" 2>/dev/null || echo "Streamlit not running"

if [ -f streamlit.log ]; then
    rm streamlit.log
    echo "Cleaned up streamlit.log"
fi

echo ""
echo " ## All services stopped!"