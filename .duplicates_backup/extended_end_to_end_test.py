#!/usr/bin/env python3
import sys
import argparse
import subprocess
import os
import json
from extended_custom_extractor import extract_facts
from simple_check import check_consistency
from simple_repair import suggest_repairs

def run_test(input_file, rules_file="advanced_domain_rules.lp", output_dir="results", verbose=False):
    """Run end-to-end test of the NLP-ASP system with advanced rules"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"RUNNING ADVANCED END-TO-END TEST")
    print(f"Input file: {input_file}")
    print(f"Rules file: {rules_file}")
    print("=" * 60)
    
    # Step 1: Extract facts
    print("\n\n1. EXTRACTING FACTS")
    print("-" * 60)
    facts = extract_facts(input_file)
    
    # Write facts to a file in the output directory
    facts_file = os.path.join(output_dir, "extracted_facts.lp")
    with open(facts_file, 'w') as f:
        for fact in facts:
            f.write(fact + "\n")
    
    print(f"Extracted {len(facts)} facts from {input_file}:")
    for fact in facts:
        print(f"  {fact}")
    print(f"Facts written to {facts_file}")
    
    # Step 2: Check consistency
    print("\n\n2. CHECKING CONSISTENCY")
    print("-" * 60)
    is_consistent = check_consistency(facts_file, rules_file)
    
    # Step 3: If inconsistent, suggest repairs
    repairs = []
    if not is_consistent:
        print("\n\n3. SUGGESTING REPAIRS")
        print("-" * 60)
        repairs = suggest_repairs(facts_file, rules_file, return_repairs=True)
    
    # Step 4: Output summary and save results
    print("\n\n4. SAVING RESULTS")
    print("-" * 60)
    
    # Save analysis results to a JSON file
    results = {
        "input_file": input_file,
        "rules_file": rules_file,
        "total_facts": len(facts),
        "is_consistent": is_consistent,
        "fact_types": analyze_facts(facts),
        "repairs": [{
            "removed_fact": repair["removed_fact"],
            "explanation": repair["explanation"]
        } for repair in repairs] if repairs else []
    }
    
    # Write results to file
    results_file = os.path.join(output_dir, "analysis_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    if verbose:
        print_analysis(results)
    
    print("\n" + "=" * 60)
    print("ADVANCED END-TO-END TEST COMPLETE")
    print("=" * 60)
    
    return is_consistent, results

def analyze_facts(facts):
    """Analyze the types of facts in the input"""
    analysis = {
        "person": 0,
        "gender": {
            "male": 0,
            "female": 0
        },
        "family": {
            "parent_of": 0,
            "child_of": 0
        },
        "marriage": {
            "married": 0,
            "married_to": 0
        },
        "identity": {
            "same_as": 0
        },
        "location": {
            "lives_in": 0
        },
        "occupation": 0,
        "other": 0
    }
    
    for fact in facts:
        if fact.startswith("person("):
            analysis["person"] += 1
        elif fact.startswith("male("):
            analysis["gender"]["male"] += 1
        elif fact.startswith("female("):
            analysis["gender"]["female"] += 1
        elif fact.startswith("parent_of("):
            analysis["family"]["parent_of"] += 1
        elif fact.startswith("child_of("):
            analysis["family"]["child_of"] += 1
        elif fact.startswith("married("):
            analysis["marriage"]["married"] += 1
        elif fact.startswith("married_to("):
            analysis["marriage"]["married_to"] += 1
        elif fact.startswith("same_as("):
            analysis["identity"]["same_as"] += 1
        elif fact.startswith("lives_in("):
            analysis["location"]["lives_in"] += 1
        elif fact.startswith("occupation("):
            analysis["occupation"] += 1
        else:
            analysis["other"] += 1
    
    return analysis

def print_analysis(results):
    """Print detailed analysis of the facts and results"""
    print("\n\nDETAILED ANALYSIS:")
    print("-" * 60)
    print(f"Total facts: {results['total_facts']}")
    print(f"Consistency: {'✅ Consistent' if results['is_consistent'] else '❌ Inconsistent'}")
    
    fact_types = results['fact_types']
    print("\nFact types:")
    print(f"  Person declarations: {fact_types['person']}")
    print(f"  Gender facts: {fact_types['gender']['male']} male, {fact_types['gender']['female']} female")
    print(f"  Family relations: {fact_types['family']['parent_of']} parent-of, {fact_types['family']['child_of']} child-of")
    print(f"  Marriage relations: {fact_types['marriage']['married']} married, {fact_types['marriage']['married_to']} married-to")
    print(f"  Identity relations: {fact_types['identity']['same_as']} same-as")
    print(f"  Location facts: {fact_types['location']['lives_in']} lives-in")
    print(f"  Occupation facts: {fact_types['occupation']}")
    print(f"  Other facts: {fact_types['other']}")
    
    if not results['is_consistent'] and results['repairs']:
        print("\nRepair suggestions:")
        for i, repair in enumerate(results['repairs'], 1):
            print(f"  {i}. Remove: {repair['removed_fact']}")
            print(f"     Explanation: {repair['explanation']}")

def run_batch_tests(test_dir, rules_file="advanced_domain_rules.lp", output_dir="batch_results"):
    """Run tests on all .txt files in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"RUNNING BATCH TESTS FROM {test_dir}")
    print("=" * 60)
    
    results = {}
    
    # Find all .txt files in the directory and its subdirectories
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.txt'):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, test_dir)
                test_name = os.path.splitext(rel_path)[0].replace('/', '_')
                
                print(f"\n\nTesting: {rel_path}")
                test_output_dir = os.path.join(output_dir, test_name)
                
                # Run the test
                is_consistent, test_results = run_test(
                    filepath, 
                    rules_file, 
                    test_output_dir,
                    verbose=False
                )
                
                # Store the result
                results[rel_path] = {
                    "is_consistent": is_consistent,
                    "output_dir": test_output_dir,
                    "fact_count": test_results["total_facts"],
                    "repair_count": len(test_results["repairs"])
                }
    
    # Write batch summary
    summary_file = os.path.join(output_dir, "batch_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print batch summary
    print("\n\n" + "=" * 60)
    print("BATCH TEST SUMMARY")
    print("=" * 60)
    print(f"Tested {len(results)} files:")
    
    consistent_count = sum(1 for r in results.values() if r["is_consistent"])
    inconsistent_count = len(results) - consistent_count
    
    print(f"  ✅ Consistent: {consistent_count}")
    print(f"  ❌ Inconsistent: {inconsistent_count}")
    print(f"Results saved in {output_dir}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run advanced end-to-end test of NLP-ASP system")
    parser.add_argument("input", help="Input text file or directory to process")
    parser.add_argument("--rules", "-r", default="advanced_domain_rules.lp", 
                        help="Rules file (default: advanced_domain_rules.lp)")
    parser.add_argument("--output", "-o", default="results",
                        help="Output directory for results (default: results)")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Run batch tests on all .txt files in the input directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed analysis")
    
    args = parser.parse_args()
    
    if args.batch:
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            sys.exit(1)
        run_batch_tests(args.input, args.rules, args.output)
    else:
        if not os.path.isfile(args.input):
            print(f"Error: {args.input} is not a file")
            sys.exit(1)
        run_test(args.input, args.rules, args.output, args.verbose)

if __name__ == "__main__":
    main() 