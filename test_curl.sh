#!/bin/bash
# Simple curl-based API test script
# Usage: ./test_curl.sh [API_URL]
# Example: ./test_curl.sh https://your-app.onrender.com

API_URL="${1:-http://localhost:8000}"

echo "============================================================"
echo "Testing API at: $API_URL"
echo "============================================================"

echo -e "\n1. Testing root endpoint..."
curl -s "$API_URL/" | jq '.' || curl -s "$API_URL/"

echo -e "\n\n2. Testing health endpoint..."
curl -s "$API_URL/health" | jq '.' || curl -s "$API_URL/health"

echo -e "\n\n3. Testing model info endpoint..."
curl -s "$API_URL/model-info" | jq '.' || curl -s "$API_URL/model-info"

echo -e "\n\n4. Testing text generation..."
curl -s -X POST "$API_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "max_length": 100,
    "temperature": 0.7
  }' | jq '.' || curl -s -X POST "$API_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "max_length": 100,
    "temperature": 0.7
  }'

echo -e "\n\n============================================================"
echo "Test completed!"
echo "============================================================"
