# NLP-ASP Quick Start Guide

This guide provides simplified instructions to get the system running quickly.

## Installation

1. Run the installation script to install basic dependencies:
```
./install.sh
```

This will install the core requirements:
- spacy
- clingo
- matplotlib
- networkx
- numpy

2. The SpaCy language model will also be downloaded automatically:
```
python -m spacy download en_core_web_sm
```

## Basic Usage

There are three main ways to use the system:

### 1. Simple Example Script

For a quick demo of the core functionality:

```
python simple_example.py input.txt
```

This will:
- Extract facts from input.txt
- Check consistency against domain rules
- Generate repair suggestions if inconsistencies are found

### 2. Command Line Interface

For more options and functionality:

```
./run.sh process input.txt
```

Additional options:
```
# Generate rules from a document
./run.sh generate-rules input.txt -n 5

# See all available commands
./run.sh --help
```

### 3. Advanced API Usage

For programmatic use in your own Python code:

```python
from enhanced_extractor import EnhancedFactExtractor
from advanced_solver import AdvancedSolver

# Extract facts
extractor = EnhancedFactExtractor()
facts = extractor.process_document("input.txt")

# Check consistency
solver = AdvancedSolver(rules_file="domain_rules.lp")
result = solver.check_consistency(facts)

# Generate repairs if needed
if result["status"] == "INCONSISTENT":
    repairs = solver.suggest_repairs(facts)
```

### 4. Processing Large Files

For large documents like books or long articles:

```
python process_large_file.py corpus/austen-emma.txt 5
```

This will:
- Extract only the first 5 paragraphs from the file
- Process this smaller subset to avoid memory or performance issues
- Create a temporary file with the extracted content
- Run the standard fact extraction and consistency checking

## Troubleshooting

### Missing Dependencies

If you encounter import errors, run the install script:
```
./install.sh
```

### Python Path Issues

If you see "module not found" errors, set the Python path:
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Advanced Features

Some advanced features (like neural coreference resolution) are optional. To enable them, you can try installing additional packages:
```
pip install transformers torch
```

## Example Files

- `input.txt`: Sample input document
- `domain_rules.lp`: Basic logical rules in ASP format
- `facts.lp`: Generated facts from example text 