#!/bin/bash
# Script to run multiple test examples through the NLP-ASP system

mkdir -p results/test_outputs

echo "===================================="
echo "RUNNING MULTIPLE EXAMPLE TESTS"
echo "===================================="

# Test 1: Consistent example
echo -e "\n\n==== TEST 1: CONSISTENT EXAMPLE ===="
python src/end_to_end_test.py data/example_inputs/sample_input.txt --output results/test_outputs/test1

# Test 2: Inconsistent example (John is the same as May)
echo -e "\n\n==== TEST 2: INCONSISTENT EXAMPLE ===="
python src/end_to_end_test.py data/example_inputs/clearer_input.txt --output results/test_outputs/test2

# Test 3: Using fixed_rules.lp (less strict rules)
echo -e "\n\n==== TEST 3: USING FIXED RULES ===="
python src/end_to_end_test.py data/example_inputs/clearer_input.txt --rules src/domain_rules/fixed_rules.lp --output results/test_outputs/test3

# Test 4: Custom example
echo -e "\n\n==== TEST 4: CUSTOM EXAMPLE ===="
echo "Creating custom example..."
cat > results/test_outputs/custom_example.txt << EOF
John is a person.
May is a person.
Jay is a person.
John is married to May.
Jay is the same person as John.
EOF

python src/end_to_end_test.py results/test_outputs/custom_example.txt --output results/test_outputs/test4

echo -e "\n\n===================================="
echo "ALL TESTS COMPLETED"
echo "Results saved in results/test_outputs directory"
echo "====================================" 