#!/bin/bash

echo "Starting Content-Pal Services..."
echo "=================================="

if [ ! -f .env ]; then
    echo " -- [.env] file not found!"
    echo "Please copy .env_template to .env and configure your settings."
    exit 1
fi

source .env

echo " Services to start:"
echo "  • OpenSearch (search engine)"
echo "  • PostgreSQL (database)"  
echo "  • Grafana (monitoring)"
echo "  • Flask API (backend)"
echo ""

echo "Starting services..."
docker-compose up -d

echo "Waiting for services to start..."
sleep 10

echo "Service Status:"
echo "=================="

services=("postgres" "grafana" "app")
for service in "${services[@]}"; do
    if docker-compose ps $service | grep -q "Up"; then
        echo "$service: Running"
    else
        echo "$service: Not running"
    fi
done

echo "Starting Streamlit UI locally..."
nohup pipenv run streamlit run src/modules/ui/streamlit_app.py --server.port=8501 --server.headless=true > streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "Streamlit PID: $STREAMLIT_PID"

sleep 5

echo ""
echo "Access URLs:"
echo "==============="
echo " ** Streamlit UI:    http://localhost:8501"
echo " ** Grafana:         http://localhost:3000  (admin/password from .env)"
echo " ** API Backend:     http://localhost:5001"
echo ""
echo " Tips:"
echo "  • Use 'docker-compose logs app' to see Flask API logs"
echo "  • Use 'docker-compose stop' to stop Docker services"
echo "  • Use 'docker-compose down' to stop and remove containers"
echo "  • Streamlit logs: tail -f streamlit.log"
echo "  • To stop Streamlit: kill $STREAMLIT_PID"
echo ""
echo " ++ All services started! Open http://localhost:8501 to use Content-Pal"
