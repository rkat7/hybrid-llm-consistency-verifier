#!/bin/bash
# Script to run the NLP-ASP system

# Set color variables
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== NLP-ASP System Runner ===${NC}"

# Function to display usage information
show_usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  ./run.sh [command] [options]"
    echo
    echo -e "${YELLOW}Commands:${NC}"
    echo -e "  ${CYAN}simple${NC} <input_file> [rules_file]       Run simple example (default: uses src/domain_rules/domain_rules.lp)"
    echo -e "  ${CYAN}end-to-end${NC} <input_file> [rules_file]   Run end-to-end test (default: uses src/domain_rules/fixed_rules.lp)"
    echo -e "  ${CYAN}advanced${NC} <input_file> [rules_file]     Run advanced end-to-end test (default: uses src/domain_rules/advanced_domain_rules.lp)"
    echo -e "  ${CYAN}batch${NC} <input_dir> [rules_file]         Run batch tests on all .txt files in a directory"
    echo -e "  ${CYAN}tests${NC} [test_suite]                     Run test suites (basic, advanced, or full)"
    echo -e "  ${CYAN}help${NC}                                   Show this help message"
    echo
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  ./run.sh simple data/example_inputs/input.txt"
    echo -e "  ./run.sh end-to-end data/example_inputs/clearer_input.txt"
    echo -e "  ./run.sh advanced data/example_inputs/clearer_input.txt src/domain_rules/advanced_domain_rules.lp"
    echo -e "  ./run.sh batch tests/scenarios"
    echo -e "  ./run.sh tests basic"
}

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Command processing
case "$1" in
    simple)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Input file is required${NC}"
            show_usage
            exit 1
        fi
        
        input_file="$2"
        rules_file="${3:-src/domain_rules/domain_rules.lp}"
        
        echo -e "${GREEN}Running simple example...${NC}"
        python src/simple_example.py "$input_file" --rules "$rules_file"
        ;;
        
    end-to-end)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Input file is required${NC}"
            show_usage
            exit 1
        fi
        
        input_file="$2"
        rules_file="${3:-src/domain_rules/fixed_rules.lp}"
        
        echo -e "${GREEN}Running end-to-end test...${NC}"
        python src/end_to_end_test.py "$input_file" --rules "$rules_file" --output results/end_to_end
        ;;
        
    advanced)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Input file is required${NC}"
            show_usage
            exit 1
        fi
        
        input_file="$2"
        rules_file="${3:-src/domain_rules/advanced_domain_rules.lp}"
        
        echo -e "${GREEN}Running advanced end-to-end test...${NC}"
        python src/extended_end_to_end_test.py "$input_file" --rules "$rules_file" --output results/advanced --verbose
        ;;
        
    batch)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Input directory is required${NC}"
            show_usage
            exit 1
        fi
        
        input_dir="$2"
        rules_file="${3:-src/domain_rules/advanced_domain_rules.lp}"
        
        echo -e "${GREEN}Running batch tests...${NC}"
        python src/extended_end_to_end_test.py "$input_dir" --batch --rules "$rules_file" --output results/batch
        ;;
        
    tests)
        test_suite="${2:-basic}"
        
        case "$test_suite" in
            basic)
                echo -e "${GREEN}Running basic tests...${NC}"
                bash tests/scripts/test_examples.sh
                ;;
                
            advanced)
                echo -e "${GREEN}Running advanced tests...${NC}"
                bash tests/scripts/test_advanced_scenarios.sh
                ;;
                
            full)
                echo -e "${GREEN}Running full test suite...${NC}"
                bash tests/scripts/test_pipeline_full.sh
                ;;
                
            *)
                echo -e "${YELLOW}Error: Unknown test suite: $test_suite${NC}"
                show_usage
                exit 1
                ;;
        esac
        ;;
        
    help|--help|-h)
        show_usage
        ;;
        
    *)
        echo -e "${YELLOW}Error: Unknown command: $1${NC}"
        show_usage
        exit 1
        ;;
esac 