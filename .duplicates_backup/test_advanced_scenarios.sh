#!/bin/bash
# Advanced test script using the updated domain rules and complex scenarios

# Create results directories
mkdir -p advanced_results/gender
mkdir -p advanced_results/family
mkdir -p advanced_results/location
mkdir -p advanced_results/identity
mkdir -p advanced_results/occupation
mkdir -p advanced_results/combined

echo "============================================================"
echo "           ADVANCED NLP-ASP SYSTEM TEST SUITE               "
echo "============================================================"

# Test 1: Gender inconsistency
echo -e "\n\n==== TEST 1: GENDER INCONSISTENCY ===="
echo "Testing conflict where a person can't be both male and female"
python extended_custom_extractor.py test_data/gender/mixed_gender.txt
python simple_check.py extracted_facts.lp advanced_domain_rules.lp
python simple_repair.py extracted_facts.lp advanced_domain_rules.lp
cp extracted_facts.lp advanced_results/gender/mixed_gender_facts.lp

# Test 2: Family parent-child cycle
echo -e "\n\n==== TEST 2: PARENT-CHILD CYCLE ===="
echo "Testing cycle in parent-child relationships"
python extended_custom_extractor.py test_data/family/parent_child_cycle.txt
python simple_check.py extracted_facts.lp advanced_domain_rules.lp
python simple_repair.py extracted_facts.lp advanced_domain_rules.lp
cp extracted_facts.lp advanced_results/family/parent_child_cycle_facts.lp

# Test 3: Family forbidden relationship
echo -e "\n\n==== TEST 3: FAMILY RELATIONSHIPS ===="
echo "Testing related people structure"
python extended_custom_extractor.py test_data/family/forbidden_relation.txt
python simple_check.py extracted_facts.lp advanced_domain_rules.lp
cp extracted_facts.lp advanced_results/family/forbidden_relation_facts.lp

# Test 4: Identity constraint
echo -e "\n\n==== TEST 4: FORBIDDEN IDENTITY ===="
echo "Testing identity constraint (John can't be Jay)"
python extended_custom_extractor.py test_data/identity/forbidden_identity.txt
python simple_check.py extracted_facts.lp advanced_domain_rules.lp
python simple_repair.py extracted_facts.lp advanced_domain_rules.lp
cp extracted_facts.lp advanced_results/identity/forbidden_identity_facts.lp

# Test 5: Location conflicts
echo -e "\n\n==== TEST 5: LOCATION CONFLICTS ===="
echo "Testing location constraints (couples living in different places)"
python extended_custom_extractor.py test_data/location/location_conflict.txt
python simple_check.py extracted_facts.lp advanced_domain_rules.lp
python simple_repair.py extracted_facts.lp advanced_domain_rules.lp
cp extracted_facts.lp advanced_results/location/location_conflict_facts.lp

# Test 6: Occupation test
echo -e "\n\n==== TEST 6: OCCUPATION TEST ===="
echo "Testing occupation information"
python extended_custom_extractor.py test_data/occupation/occupation_test.txt
python simple_check.py extracted_facts.lp advanced_domain_rules.lp
cp extracted_facts.lp advanced_results/occupation/occupation_facts.lp

# Test 7: Complex family (consistent)
echo -e "\n\n==== TEST 7: COMPLEX FAMILY STRUCTURE (CONSISTENT) ===="
echo "Testing complex but consistent family structure"
python extended_custom_extractor.py test_data/family/complex_family.txt
python simple_check.py extracted_facts.lp advanced_domain_rules.lp
cp extracted_facts.lp advanced_results/family/complex_family_facts.lp

# Test 8: Combined end-to-end test with complex data
echo -e "\n\n==== TEST 8: END-TO-END WITH COMPLEX DATA ===="
python end_to_end_test.py test_data/location/location_conflict.txt --rules advanced_domain_rules.lp --output advanced_results/combined/location_test

# Test 9: End-to-end with gender inconsistency
echo -e "\n\n==== TEST 9: END-TO-END WITH GENDER INCONSISTENCY ===="
python end_to_end_test.py test_data/gender/mixed_gender.txt --rules advanced_domain_rules.lp --output advanced_results/combined/gender_test

echo -e "\n\n============================================================"
echo "                 ADVANCED TESTS COMPLETED                    "
echo "        Results saved in advanced_results directory          "
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
echo "Check advanced_results directory for detailed output files" 