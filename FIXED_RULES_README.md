# Fixed Rules and Repair System

This README describes the solution to the recursion error that occurred when using the NLP-ASP system with generated rules.

## Problem

When trying to use the generated rules (generated_rules_20250514_235643.lp) with the simple_example.py, a recursion error occurred in the advanced_solver.py. The error was happening in the `_find_minimal_inconsistent_subset` method which entered an infinite recursion loop when trying to check consistency.

The primary issues were:
1. The rules had a self-referential loop where checking them would cause infinite recursion
2. The rule `:- not { may(X) } = 4.` expected exactly 4 entities to be "may", but our input only mentioned one
3. The recursive explanation-finding algorithm in advanced_solver.py didn't have a proper base case

## Solution

We created several alternative components to fix the issues:

1. **Fixed Rules**: We created clearer rules in `fixed_rules.lp` and `stronger_rules.lp` that avoid self-referential loops.

2. **Custom Extractor**: We created `custom_extractor.py` that specifically extracts facts in the format needed for our rules, including the important "same_as" relationship.

3. **Simplified Checking**: We created `simple_check.py` that directly checks fact consistency against rules without the recursive explanation algorithm.

4. **Simple Repair**: We created `simple_repair.py` that suggests repairs by testing the removal of each fact to see if it resolves inconsistencies.

## How to Use

### Extracting Facts from Text
```
python custom_extractor.py clearer_input.txt
```
This creates an `extracted_facts.lp` file with the extracted facts.

### Checking Consistency
```
python simple_check.py extracted_facts.lp stronger_rules.lp
```
This will check if the facts are consistent with the rules.

### Repairing Inconsistencies
```
python simple_repair.py extracted_facts.lp stronger_rules.lp
```
This will suggest repairs to resolve any inconsistencies.

## Rule Sets

### Fixed Rules (fixed_rules.lp)
Basic rules that define predicates and some constraints.

### Stronger Rules (stronger_rules.lp)
Rules that explicitly forbid John and May to be the same person, creating a clear inconsistency for demonstration.

## Example Files

- `clearer_input.txt`: Clear text input to extract facts from
- `extracted_facts.lp`: Sample facts extracted from the input
- `sample_facts.lp`: Hand-crafted example facts

## Future Improvements

1. Update advanced_solver.py to handle recursive explanations without stack overflow
2. Integrate more robust rule generation that avoids problematic constructs
3. Add better visualization of fact relationships and conflicts 