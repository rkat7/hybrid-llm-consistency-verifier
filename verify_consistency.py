#!/usr/bin/env python3
from advanced_solver import AdvancedSolver
import argparse

def main():
    parser = argparse.ArgumentParser(description="Verify consistency of facts against rules")
    parser.add_argument("facts_file", help="File containing facts")
    parser.add_argument("--rules", default="fixed_rules.lp", help="Rules file (default: fixed_rules.lp)")
    args = parser.parse_args()
    
    # Read facts
    with open(args.facts_file, 'r') as f:
        facts = [line.strip() for line in f if line.strip()]
    
    # Print facts
    print(f"Loaded {len(facts)} facts from {args.facts_file}:")
    for fact in facts:
        print(f"  {fact}")
    
    # Check consistency
    print("\nChecking consistency against rules...")
    solver = AdvancedSolver(rules_file=args.rules)
    result = solver.check_consistency(facts)
    
    if result["status"] == "CONSISTENT":
        print("\n✅ Facts are consistent with domain rules")
    else:
        print("\n❌ Facts are inconsistent with domain rules")
        
        if result["explanation"]:
            print("\nExplanation:")
            print(result["explanation"]["description"])
            
            # Generate repair suggestions
            print("\nGenerating repair suggestions...")
            repairs = solver.suggest_repairs(facts)
            
            print(f"Found {len(repairs)} possible repairs:")
            for i, repair in enumerate(repairs, 1):
                print(f"\nRepair option {i}:")
                print(f"Explanation: {repair.explanation}")
                print("Removed facts:")
                for fact in repair.removed_facts:
                    print(f"  - {fact}")
                if repair.added_facts:
                    print("Added facts:")
                    for fact in repair.added_facts:
                        print(f"  + {fact}")

if __name__ == "__main__":
    main() 