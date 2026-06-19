#!/bin/bash
echo "🚀 Starting all IR services..."

uvicorn services.preprocessing_service.main:app    --port 8001 --reload &
uvicorn services.indexing_service.main:app         --port 8002 --reload &
uvicorn services.retrieval_service.main:app        --port 8003 --reload &
uvicorn services.ranking_service.main:app          --port 8004 --reload &
uvicorn services.query_refinement_service.main:app --port 8005 --reload &
uvicorn services.api_gateway.main:app              --port 8000 --reload &

echo "✅ All services started!"
echo "   Gateway  → http://localhost:8000"
echo "   API Docs → http://localhost:8000/docs"
