#!/bin/bash
# Script to run multiple test examples through the NLP-ASP system

mkdir -p test_outputs

echo "===================================="
echo "RUNNING MULTIPLE EXAMPLE TESTS"
echo "===================================="

# Test 1: Consistent example
echo -e "\n\n==== TEST 1: CONSISTENT EXAMPLE ===="
python end_to_end_test.py sample_input.txt --output test_outputs/test1

# Test 2: Inconsistent example (John is the same as May)
echo -e "\n\n==== TEST 2: INCONSISTENT EXAMPLE ===="
python end_to_end_test.py clearer_input.txt --output test_outputs/test2

# Test 3: Using fixed_rules.lp (less strict rules)
echo -e "\n\n==== TEST 3: USING FIXED RULES ===="
python end_to_end_test.py clearer_input.txt --rules fixed_rules.lp --output test_outputs/test3

# Test 4: Custom example
echo -e "\n\n==== TEST 4: CUSTOM EXAMPLE ===="
echo "Creating custom example..."
cat > test_outputs/custom_example.txt << EOF
John is a person.
May is a person.
Jay is a person.
John is married to May.
Jay is the same person as John.
EOF

python end_to_end_test.py test_outputs/custom_example.txt --output test_outputs/test4

echo -e "\n\n===================================="
echo "ALL TESTS COMPLETED"
echo "Results saved in test_outputs directory"
echo "====================================" 