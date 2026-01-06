#!/bin/bash
# Start all services

set -e

echo "======================================"
echo "BTC ML System - Starting Services"
echo "======================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found"
    echo "Please copy .env.example to .env and configure"
    exit 1
fi

# Create directories if they don't exist
mkdir -p logs models/active models/archive data

# Ask for isolation mode
echo ""
echo "Choose deployment mode:"
echo "  1) FULL ISOLATION (no ports exposed) - RECOMMENDED"
echo "  2) WITH DATABASE PORT (for development/monitoring)"
echo ""
read -p "Enter choice [1-2] (default: 1): " choice
choice=${choice:-1}

if [ "$choice" = "1" ]; then
    COMPOSE_FILE="docker-compose.isolated.yml"
    echo "Using FULLY ISOLATED mode (no external ports)"
else
    COMPOSE_FILE="docker-compose.yml"
    echo "Using DEVELOPMENT mode (database port exposed)"
fi

echo ""
echo "Building Docker images..."
docker-compose -f $COMPOSE_FILE build

echo ""
echo "Starting services..."
docker-compose -f $COMPOSE_FILE up -d

echo ""
echo "======================================"
echo "Services started!"
echo "======================================"
echo ""
echo "View logs:"
echo "  docker-compose -f $COMPOSE_FILE logs -f"
echo ""
echo "Check status:"
echo "  docker-compose -f $COMPOSE_FILE ps"
echo ""

if [ "$choice" = "1" ]; then
    echo "Database access (isolated mode):"
    echo "  docker exec btc-ml-db psql -U mluser -d btc_ml"
else
    echo "Database access (exposed mode):"
    echo "  psql -h localhost -p 5432 -U mluser -d btc_ml"
fi

echo ""
echo "Stop services:"
echo "  docker-compose -f $COMPOSE_FILE down"
echo ""