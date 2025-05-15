#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== NLP-ASP System Installer ===${NC}"

# Install basic requirements
echo -e "${YELLOW}Installing basic dependencies...${NC}"
pip install spacy clingo matplotlib networkx numpy

# Download SpaCy model
echo -e "${YELLOW}Downloading SpaCy language model...${NC}"
python -m spacy download en_core_web_sm

echo -e "${GREEN}Installation complete!${NC}"
echo -e "${YELLOW}Run ./run.sh to start using the system${NC}" 