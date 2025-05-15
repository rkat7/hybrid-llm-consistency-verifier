# NLP-ASP System Testing Guide

This document explains how to test the NLP-ASP system using the provided testing scripts.

## Overview

Three testing scripts are provided:
1. `end_to_end_test.py` - Python script for integrated testing
2. `test_examples.sh` - Simple shell script to test multiple examples
3. `test_pipeline_full.sh` - Comprehensive test script for full system testing

## Basic Testing

To run a single test with the end-to-end script:

```bash
python end_to_end_test.py clearer_input.txt
```

This will:
1. Extract facts from the input text
2. Check consistency against the rules (default: stronger_rules.lp)
3. Suggest repairs if inconsistencies are found

## Options for end_to_end_test.py

```
usage: end_to_end_test.py [-h] [--rules RULES] [--output OUTPUT] [--method {integrated,separate}] input_file

Run end-to-end test of NLP-ASP system

positional arguments:
  input_file            Input text file to process

options:
  -h, --help            show this help message and exit
  --rules RULES, -r RULES
                        Rules file (default: stronger_rules.lp)
  --output OUTPUT, -o OUTPUT
                        Output directory for results (default: results)
  --method {integrated,separate}, -m {integrated,separate}
                        Run method: integrated (in one process) or separate (individual scripts)
```

## Running Multiple Tests

For a quick run of multiple predefined examples:

```bash
./test_examples.sh
```

This script runs 4 different test cases:
1. Consistent example
2. Inconsistent example (John is the same as May)
3. Using fixed_rules.lp (less strict rules)
4. Custom example

## Comprehensive Testing

For a full system test with all components:

```bash
./test_pipeline_full.sh
```

This comprehensive test runs 9 test cases:
- Basic tests with different extractors and rule sets
- Advanced tests with the end-to-end script
- Custom tests with special cases

Results are saved in the `test_results` directory.

## Custom Testing

To run a test with your own input:

1. Create a text file with statements (e.g., `my_input.txt`)
2. Run the test:
```bash
python end_to_end_test.py my_input.txt --rules rules_for_custom_example.lp
```

## Available Rule Sets

1. `fixed_rules.lp` - Basic rules that define predicates and some constraints
2. `stronger_rules.lp` - Rules that explicitly forbid John and May to be the same person
3. `rules_for_custom_example.lp` - Custom rules with marriage relationships and identity constraints

## Extractors

1. `custom_extractor.py` - Basic fact extractor
2. `update_custom_extractor.py` - Enhanced extractor with more patterns

## Testing Output

Test results are saved in the specified output directory (or `results` by default) and include:
- Extracted facts
- Consistency check results
- Repair suggestions (if applicable) 