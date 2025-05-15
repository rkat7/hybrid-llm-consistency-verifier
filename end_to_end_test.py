#!/usr/bin/env python3
import sys
import argparse
import subprocess
import os
from custom_extractor import extract_facts
from simple_check import check_consistency
from simple_repair import suggest_repairs

def run_test(input_file, rules_file="stronger_rules.lp", output_dir="results"):
    """Run end-to-end test of the NLP-ASP system"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"RUNNING END-TO-END TEST")
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
    if not is_consistent:
        print("\n\n3. SUGGESTING REPAIRS")
        print("-" * 60)
        suggest_repairs(facts_file, rules_file)
    
    print("\n" + "=" * 60)
    print("END-TO-END TEST COMPLETE")
    print("=" * 60)
    
    return is_consistent

def run_full_pipeline(input_file, rules_file="stronger_rules.lp"):
    """Execute the full pipeline using different scripts"""
    print("=" * 60)
    print(f"RUNNING FULL PIPELINE")
    print(f"Input file: {input_file}")
    print(f"Rules file: {rules_file}")
    print("=" * 60)
    
    # Step 1: Extract facts
    print("\n\n1. EXTRACTING FACTS")
    print("-" * 60)
    subprocess.run(["python", "custom_extractor.py", input_file])
    
    # Step 2: Check consistency
    print("\n\n2. CHECKING CONSISTENCY")
    print("-" * 60)
    result = subprocess.run(["python", "simple_check.py", "extracted_facts.lp", rules_file])
    
    # Step 3: If check failed, suggest repairs
    if result.returncode != 0:
        print("\n\n3. SUGGESTING REPAIRS")
        print("-" * 60)
        subprocess.run(["python", "simple_repair.py", "extracted_facts.lp", rules_file])
    
    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPLETE")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Run end-to-end test of NLP-ASP system")
    parser.add_argument("input_file", help="Input text file to process")
    parser.add_argument("--rules", "-r", default="stronger_rules.lp", 
                        help="Rules file (default: stronger_rules.lp)")
    parser.add_argument("--output", "-o", default="results",
                        help="Output directory for results (default: results)")
    parser.add_argument("--method", "-m", choices=["integrated", "separate"], default="integrated",
                        help="Run method: integrated (in one process) or separate (individual scripts)")
    
    args = parser.parse_args()
    
    if args.method == "integrated":
        run_test(args.input_file, args.rules, args.output)
    else:
        run_full_pipeline(args.input_file, args.rules)

if __name__ == "__main__":
    main() 