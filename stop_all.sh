#!/bin/bash

echo "Stopping Content-Pal Services..."
echo "=================================="

echo "Stopping Docker services..."
docker-compose down

echo "Stopping Streamlit..."
if [ -f .streamlit_pid ]; then
    STREAMLIT_PID=$(cat .streamlit_pid)
    if kill $STREAMLIT_PID 2>/dev/null; then
        echo "Stopped Streamlit (PID: $STREAMLIT_PID)"
    fi
    rm .streamlit_pid
else
    pkill -f "pipenv run streamlit" 2>/dev/null || echo "Streamlit not running"
fi

if [ -f streamlit.log ]; then
    rm streamlit.log
    echo "Cleaned up streamlit.log"
fi

echo ""
echo "All services stopped!"