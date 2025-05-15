# Advanced NLP-ASP Testing Suite

This document explains the advanced testing capabilities added to the NLP-ASP system.

## New Components

1. **Advanced Domain Rules:**
   - `advanced_domain_rules.lp` - Comprehensive rules including gender, family relationships, locations, and occupations

2. **Extended Extractors:**
   - `extended_custom_extractor.py` - Enhanced fact extractor supporting all advanced domain predicates

3. **Advanced Testing Scripts:**
   - `extended_end_to_end_test.py` - Improved end-to-end testing with batch capabilities and analysis
   - `test_advanced_scenarios.py` - Script to test specific domain constraint scenarios

4. **Test Data:**
   - Various test scenarios organized in the `test_data/` directory

## Advanced Domain Rules

The advanced domain rules include:

- **Basic Predicates:** person, male, female
- **Family Relationships:** parent_of, child_of, related
- **Marriage Rules:** married, married_to, can_marry
- **Identity Rules:** same_as (with reflexivity, symmetry, transitivity)
- **Location Rules:** lives_in (with constraints for families)
- **Occupation Rules:** occupation
- **Age Rules:** minor

### Example Constraints:

- A person cannot be both male and female
- A person cannot be their own parent
- A minor must live with their parents
- Married couples must live in the same location
- Certain people cannot be the same (e.g., john and jay)
- Related people cannot marry each other

## Test Scenarios

The test data is organized by domain:

1. **Gender:** Testing gender-related constraints
   - `mixed_gender.txt` - Tests the constraint that a person can't be both male and female

2. **Family:** Testing family relationship constraints
   - `parent_child_cycle.txt` - Tests cycles in parent-child relationships
   - `forbidden_relation.txt` - Tests relationship constraints
   - `complex_family.txt` - Tests a consistent complex family structure

3. **Identity:** Testing identity constraints
   - `forbidden_identity.txt` - Tests forbidden identity relationships

4. **Location:** Testing location constraints
   - `location_conflict.txt` - Tests the constraint that couples must live together

5. **Occupation:** Testing occupation information
   - `occupation_test.txt` - Tests occupation facts

## How to Run Advanced Tests

### Running Individual Tests

To run a test on a single scenario:

```bash
python extended_end_to_end_test.py test_data/gender/mixed_gender.txt --rules advanced_domain_rules.lp
```

With detailed analysis:
```bash
python extended_end_to_end_test.py test_data/gender/mixed_gender.txt --rules advanced_domain_rules.lp --verbose
```

### Running Batch Tests

To run tests on all scenarios in a directory:

```bash
python extended_end_to_end_test.py test_data --batch --rules advanced_domain_rules.lp
```

Output will be saved to a directory (default: `batch_results`)

### Running All Advanced Tests

To run all test scenarios with detailed output:

```bash
./test_advanced_scenarios.sh
```

This script will test all scenarios and save results to `advanced_results/`.

## Advanced Testing Features

1. **Fact Analysis:**
   - Automatic categorization of facts by type
   - Statistics on fact distribution

2. **Inconsistency Detection:**
   - Identification of constraint violations
   - Explanation of which rules are violated

3. **Repair Suggestions:**
   - Suggestions for resolving inconsistencies
   - Multiple repair options when available

4. **JSON Reports:**
   - Structured output for programmatic analysis
   - Summary of all test results

## Extending the Tests

To add new test scenarios:

1. Create a new text file with the scenario description
2. Place it in the appropriate subdirectory of `test_data/`
3. Run the batch tests to include your new scenario

To add new domain rules:

1. Edit `advanced_domain_rules.lp` to add new predicates and constraints
2. Update `extended_custom_extractor.py` to recognize new patterns
3. Create test cases for the new constraints 