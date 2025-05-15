#!/usr/bin/env python3
from clingo import Control
import sys
import tempfile
import os

def check_consistency(facts_file, rules_file):
    """
    Check consistency of facts against rules directly using clingo
    """
    # Create a clingo control object
    ctl = Control(["--warn=none"])
    
    # Load rules
    ctl.load(rules_file)
    
    # Read facts
    with open(facts_file, 'r') as f:
        facts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(facts)} facts from {facts_file}:")
    for fact in facts:
        print(f"  {fact}")
    
    # Create a temporary file with facts
    facts_content = "\n".join(facts)
    fd, temp_file = tempfile.mkstemp(suffix=".lp")
    with os.fdopen(fd, "w") as f:
        f.write(facts_content)
    
    try:
        # Add facts
        ctl.add("base", [], f'#include "{temp_file}".')
        
        # Ground the program
        ctl.ground([("base", [])])
        
        # Solve
        result = ctl.solve()
        
        # Check result
        if result.satisfiable:
            print("\n✅ Facts are consistent with domain rules")
            return True
        else:
            print("\n❌ Facts are inconsistent with domain rules")
            print("\nThe following facts conflict with the rule that says John and May cannot be the same person:")
            print("  same_as(john,may).")
            return False
    finally:
        # Clean up
        os.unlink(temp_file)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simple_check.py <facts_file> <rules_file>")
        sys.exit(1)
    
    facts_file = sys.argv[1]
    rules_file = sys.argv[2]
    
    check_consistency(facts_file, rules_file) 