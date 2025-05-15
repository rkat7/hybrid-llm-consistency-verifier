#!/bin/bash
# Advanced test script using the updated domain rules and complex scenarios

# Create results directories
mkdir -p results/advanced/gender
mkdir -p results/advanced/family
mkdir -p results/advanced/location
mkdir -p results/advanced/identity
mkdir -p results/advanced/occupation
mkdir -p results/advanced/combined

echo "============================================================"
echo "           ADVANCED NLP-ASP SYSTEM TEST SUITE               "
echo "============================================================"

# Test 1: Gender inconsistency
echo -e "\n\n==== TEST 1: GENDER INCONSISTENCY ===="
echo "Testing conflict where a person can't be both male and female"
python src/extractors/extended_custom_extractor.py tests/scenarios/gender/mixed_gender.txt
python src/solvers/simple_check.py extracted_facts.lp src/domain_rules/advanced_domain_rules.lp
python src/solvers/simple_repair.py extracted_facts.lp src/domain_rules/advanced_domain_rules.lp
cp extracted_facts.lp results/advanced/gender/mixed_gender_facts.lp

# Test 2: Family parent-child cycle
echo -e "\n\n==== TEST 2: PARENT-CHILD CYCLE ===="
echo "Testing cycle in parent-child relationships"
python src/extractors/extended_custom_extractor.py tests/scenarios/family/parent_child_cycle.txt
python src/solvers/simple_check.py extracted_facts.lp src/domain_rules/advanced_domain_rules.lp
python src/solvers/simple_repair.py extracted_facts.lp src/domain_rules/advanced_domain_rules.lp
cp extracted_facts.lp results/advanced/family/parent_child_cycle_facts.lp

# Test 3: Family forbidden relationship
echo -e "\n\n==== TEST 3: FAMILY RELATIONSHIPS ===="
echo "Testing related people structure"
python src/extractors/extended_custom_extractor.py tests/scenarios/family/forbidden_relation.txt
python src/solvers/simple_check.py extracted_facts.lp src/domain_rules/advanced_domain_rules.lp
cp extracted_facts.lp results/advanced/family/forbidden_relation_facts.lp

# Test 4: Identity constraint
echo -e "\n\n==== TEST 4: FORBIDDEN IDENTITY ===="
echo "Testing identity constraint (John can't be Jay)"
python src/extractors/extended_custom_extractor.py tests/scenarios/identity/forbidden_identity.txt
python src/solvers/simple_check.py extracted_facts.lp src/domain_rules/advanced_domain_rules.lp
python src/solvers/simple_repair.py extracted_facts.lp src/domain_rules/advanced_domain_rules.lp
cp extracted_facts.lp results/advanced/identity/forbidden_identity_facts.lp

# Test 5: Location conflicts
echo -e "\n\n==== TEST 5: LOCATION CONFLICTS ===="
echo "Testing location constraints (couples living in different places)"
python src/extractors/extended_custom_extractor.py tests/scenarios/location/location_conflict.txt
python src/solvers/simple_check.py extracted_facts.lp src/domain_rules/advanced_domain_rules.lp
python src/solvers/simple_repair.py extracted_facts.lp src/domain_rules/advanced_domain_rules.lp
cp extracted_facts.lp results/advanced/location/location_conflict_facts.lp

# Test 6: Occupation test
echo -e "\n\n==== TEST 6: OCCUPATION TEST ===="
echo "Testing occupation information"
python src/extractors/extended_custom_extractor.py tests/scenarios/occupation/occupation_test.txt
python src/solvers/simple_check.py extracted_facts.lp src/domain_rules/advanced_domain_rules.lp
cp extracted_facts.lp results/advanced/occupation/occupation_facts.lp

# Test 7: Complex family (consistent)
echo -e "\n\n==== TEST 7: COMPLEX FAMILY STRUCTURE (CONSISTENT) ===="
echo "Testing complex but consistent family structure"
python src/extractors/extended_custom_extractor.py tests/scenarios/family/complex_family.txt
python src/solvers/simple_check.py extracted_facts.lp src/domain_rules/advanced_domain_rules.lp
cp extracted_facts.lp results/advanced/family/complex_family_facts.lp

# Test 8: Combined end-to-end test with complex data
echo -e "\n\n==== TEST 8: END-TO-END WITH COMPLEX DATA ===="
python src/end_to_end_test.py tests/scenarios/location/location_conflict.txt --rules src/domain_rules/advanced_domain_rules.lp --output results/advanced/combined/location_test

# Test 9: End-to-end with gender inconsistency
echo -e "\n\n==== TEST 9: END-TO-END WITH GENDER INCONSISTENCY ===="
python src/end_to_end_test.py tests/scenarios/gender/mixed_gender.txt --rules src/domain_rules/advanced_domain_rules.lp --output results/advanced/combined/gender_test

echo -e "\n\n============================================================"
echo "                 ADVANCED TESTS COMPLETED                    "
echo "        Results saved in results/advanced directory          "
echo "============================================================"

# Print summary
echo -e "\n\nADVANCED TEST SUMMARY:"
echo "----------------------------------------------------------------"
echo "9 advanced tests completed with the following scenarios:"
echo "1. Gender inconsistency (person can't be both male and female)"
echo "2. Parent-child cycle (person can't be their own ancestor)"
echo "3. Family relationships (related people constraints)"
echo "4. Forbidden identity (certain people can't be the same)"
echo "5. Location conflicts (couples must live together, minors with parents)"
echo "6. Occupation information (testing job roles)"
echo "7. Complex but consistent family structure"
echo "8-9. End-to-end tests with complex data"
echo ""
echo "Check results/advanced directory for detailed output files" 