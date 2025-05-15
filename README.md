# NLP-ASP System

A system that combines Natural Language Processing (NLP) with Answer Set Programming (ASP) for logical reasoning over text.

## Project Structure

```
hybrid-llm-consistency-verifier/
├── src/                  # Source code
│   ├── extractors/       # Fact extraction modules
│   ├── solvers/          # ASP solvers and reasoning modules
│   ├── domain_rules/     # ASP rule definitions
│   ├── utils/            # Utility functions
│   ├── app/              # Application code (CLI, pipeline, tasks)
│   ├── extensions/       # Extension modules
│   ├── runner.py         # Main runner script
│   └── simple_example.py # Simple example usage
├── tests/                # Test files
│   ├── scenarios/        # Test scenario files
│   └── scripts/          # Test scripts
├── data/                 # Data files
│   ├── example_inputs/   # Example input texts
│   └── corpus/           # Larger text corpus
├── docs/                 # Documentation
└── results/              # Test results
```

## Quick Start

1. **Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. **Run a simple example**
```bash
python src/simple_example.py data/example_inputs/input.txt
```

3. **Run tests**
```bash
# Basic tests
bash tests/scripts/test_examples.sh

# Advanced tests
bash tests/scripts/test_advanced_scenarios.sh

# Full pipeline tests
bash tests/scripts/test_pipeline_full.sh
```

## Features

- **Fact Extraction**: Extract structured facts from natural language text
- **Consistency Checking**: Verify logical consistency of extracted facts
- **Repair Suggestions**: Generate repair suggestions for inconsistent facts
- **Advanced Domain Rules**: Define complex domain rules in ASP
- **Comprehensive Testing**: Test various scenarios and rule violations

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [Basic Testing Guide](docs/README_TESTING.md)
- [Advanced Testing Guide](docs/README_ADVANCED_TESTING.md)

## Extended Usage

### End-to-End Processing

```bash
python src/end_to_end_test.py data/example_inputs/input.txt
```

### Advanced Processing with Domain Rules

```bash
python src/extended_end_to_end_test.py data/example_inputs/input.txt --rules src/domain_rules/advanced_domain_rules.lp
```

### Batch Processing

```bash
python src/extended_end_to_end_test.py tests/scenarios --batch --output results/batch_results
```

## License

This project is open source and available under the MIT License. 
