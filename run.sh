#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== NLP-ASP System Launcher ===${NC}"

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Fix any potential import path issues
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Install SpaCy model if needed
if ! python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    echo -e "${YELLOW}Installing SpaCy language model...${NC}"
    python -m spacy download en_core_web_sm
fi

# Prompt for document to process if not provided
if [ "$#" -eq 0 ]; then
    echo -e "${YELLOW}Enter the path to a document to process (default: input.txt):${NC}"
    read DOC_PATH
    DOC_PATH=${DOC_PATH:-input.txt}
    
    # Run the CLI tool
    echo -e "${GREEN}Processing document: ${DOC_PATH}${NC}"
    python cli.py process "$DOC_PATH"
else
    # Run with given arguments
    echo -e "${GREEN}Running command: python cli.py $@${NC}"
    python cli.py "$@"
fi

echo -e "${GREEN}Done!${NC}" 