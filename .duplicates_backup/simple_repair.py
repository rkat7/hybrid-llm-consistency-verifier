#!/usr/bin/env python3
from clingo import Control
import sys
import tempfile
import os

def check_consistency(facts, rules_file):
    """
    Check if a set of facts is consistent with rules
    """
    # Create a clingo control object
    ctl = Control(["--warn=none"])
    
    # Load rules
    ctl.load(rules_file)
    
    # Create a temporary file with facts
    facts_content = "\n".join(facts)
    fd, temp_file = tempfile.mkstemp(suffix=".lp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(facts_content)
        
        # Add facts
        ctl.add("base", [], f'#include "{temp_file}".')
        
        # Ground the program
        ctl.ground([("base", [])])
        
        # Solve
        result = ctl.solve()
        
        # Check result
        return result.satisfiable
    finally:
        # Clean up
        os.unlink(temp_file)

def suggest_repairs(facts_file, rules_file, return_repairs=False):
    """
    Suggest repairs to make facts consistent with rules
    """
    # Read facts
    with open(facts_file, 'r') as f:
        facts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(facts)} facts from {facts_file}:")
    for fact in facts:
        print(f"  {fact}")
    
    # Check if facts are consistent
    if check_consistency(facts, rules_file):
        print("\n✅ Facts are already consistent with domain rules")
        return [] if return_repairs else None
    
    print("\n❌ Facts are inconsistent with domain rules")
    
    # Try removing each fact to see if it resolves the inconsistency
    print("\nGenerating repair suggestions:")
    repair_options = []
    
    for i, fact in enumerate(facts):
        # Create a new list without this fact
        repaired_facts = facts.copy()
        repaired_facts.pop(i)
        
        # Check if this resolves the inconsistency
        if check_consistency(repaired_facts, rules_file):
            repair_options.append({
                "removed_fact": fact,
                "explanation": f"Removing the fact '{fact}' resolves the inconsistency."
            })
    
    # Print repair options
    if repair_options:
        print(f"\nFound {len(repair_options)} possible repairs:")
        for i, repair in enumerate(repair_options, 1):
            print(f"\nRepair option {i}:")
            print(f"Explanation: {repair['explanation']}")
            print("Removed fact:")
            print(f"  - {repair['removed_fact']}")
    else:
        print("\nNo simple repairs found. The inconsistency may require more complex changes.")
    
    return repair_options if return_repairs else None

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simple_repair.py <facts_file> <rules_file>")
        sys.exit(1)
    
    facts_file = sys.argv[1]
    rules_file = sys.argv[2]
    
    suggest_repairs(facts_file, rules_file) 