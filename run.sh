#!/bin/bash
# TennisPreview Run Script

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    TennisPreview v1.0.0                        ║${NC}"
echo -e "${GREEN}║           Tennis Match Analysis & Value Betting               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Determine python interpreter
if [ -d "venv" ] && [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
    echo -e "${GREEN}Using virtual environment${NC}"
else
    PYTHON="python3"
    echo -e "${YELLOW}No venv found - using system python3${NC}"
fi

# Install missing dependencies if needed
if ! $PYTHON -c "import pandas, numpy, loguru, feedparser, tqdm, yaml, dotenv, aiohttp" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    if [ -d "venv" ]; then
        venv/bin/pip install -r requirements.txt -q
    else
        pip3 install --break-system-packages -r requirements.txt -q 2>/dev/null || \
        pip3 install -r requirements.txt -q
    fi
fi

# Check .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}No .env file found. Creating from example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}API keys are optional: the program works in demo mode without them.${NC}"
    echo -e "${YELLOW}Edit .env and add THEODDSAPI_KEY / API_FOOTBALL_KEY for live odds.${NC}"
fi

# Load environment (ignore errors if keys missing)
set +e
source .env 2>/dev/null
set -e

# Run the pipeline
echo -e "${GREEN}Starting TennisPreview pipeline...${NC}"
echo ""

$PYTHON -m src.main "$@"

echo ""
echo -e "${GREEN}Done!${NC}"