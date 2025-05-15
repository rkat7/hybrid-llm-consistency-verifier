#!/bin/bash
# Comprehensive test script for the NLP-ASP system

# Create test directories
mkdir -p test_results/basic
mkdir -p test_results/advanced
mkdir -p test_results/custom

echo "============================================================"
echo "            COMPREHENSIVE NLP-ASP SYSTEM TEST               "
echo "============================================================"

# Run basic tests with different extractors and rule sets
echo -e "\n\n### BASIC TESTS ###"
echo "----------------------------------------------------------------"

# Test 1: Basic consistent example with custom_extractor
echo -e "\n--- Test 1: Basic consistent example (custom_extractor) ---"
python custom_extractor.py sample_input.txt
python simple_check.py extracted_facts.lp fixed_rules.lp
cp extracted_facts.lp test_results/basic/test1_facts.lp

# Test 2: Basic consistent example with update_custom_extractor
echo -e "\n--- Test 2: Basic consistent example (update_custom_extractor) ---"
python update_custom_extractor.py sample_input.txt
python simple_check.py extracted_facts.lp fixed_rules.lp
cp extracted_facts.lp test_results/basic/test2_facts.lp

# Test 3: Inconsistent example with stronger rules
echo -e "\n--- Test 3: Inconsistent example with stronger rules ---"
python custom_extractor.py clearer_input.txt
python simple_check.py extracted_facts.lp stronger_rules.lp
python simple_repair.py extracted_facts.lp stronger_rules.lp
cp extracted_facts.lp test_results/basic/test3_facts.lp

# Run advanced tests with the end-to-end script
echo -e "\n\n### ADVANCED TESTS ###"
echo "----------------------------------------------------------------"

# Test 4: End-to-end with consistent example
echo -e "\n--- Test 4: End-to-end consistent example ---"
python end_to_end_test.py sample_input.txt --output test_results/advanced/test4

# Test 5: End-to-end with inconsistent example
echo -e "\n--- Test 5: End-to-end inconsistent example ---"
python end_to_end_test.py clearer_input.txt --output test_results/advanced/test5

# Test 6: End-to-end with separate process method
echo -e "\n--- Test 6: End-to-end with separate processes ---"
python end_to_end_test.py clearer_input.txt --method separate --output test_results/advanced/test6

# Run custom tests with special cases
echo -e "\n\n### CUSTOM TESTS ###"
echo "----------------------------------------------------------------"

# Create a custom example with marriage relationship
echo -e "\n--- Creating custom example 1 ---"
cat > test_results/custom/custom_example1.txt << EOF
John is a person.
May is a person.
John is married to May.
EOF

echo -e "\n--- Test 7: Custom example with marriage relationship ---"
python update_custom_extractor.py test_results/custom/custom_example1.txt
python simple_check.py extracted_facts.lp rules_for_custom_example.lp
cp extracted_facts.lp test_results/custom/test7_facts.lp

# Create a custom example with jay and john same person
echo -e "\n--- Creating custom example 2 ---"
cat > test_results/custom/custom_example2.txt << EOF
John is a person.
May is a person.
Jay is a person.
John is married to May.
Jay is the same person as John.
EOF

echo -e "\n--- Test 8: Custom example with Jay same as John (should be inconsistent) ---"
python update_custom_extractor.py test_results/custom/custom_example2.txt
python simple_check.py extracted_facts.lp rules_for_custom_example.lp
python simple_repair.py extracted_facts.lp rules_for_custom_example.lp
cp extracted_facts.lp test_results/custom/test8_facts.lp

# Run a final end-to-end test with the custom rules
echo -e "\n--- Test 9: End-to-end with custom example and rules ---"
python end_to_end_test.py test_results/custom/custom_example2.txt --rules rules_for_custom_example.lp --output test_results/custom/test9

echo -e "\n\n============================================================"
echo "              ALL TESTS COMPLETED                        "
echo "              Results saved in test_results directory    "
echo "============================================================"

# Summary display
echo -e "\n\nTEST SUMMARY:"
echo "----------------------------------------------------------------"
echo "Total tests run: 9"
echo "Test results saved in test_results directory"
echo "See individual test output directories for detailed results" 