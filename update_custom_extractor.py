#!/usr/bin/env python3
import re
from typing import List

def extract_facts(input_file: str) -> List[str]:
    """
    An enhanced pattern-based fact extractor for our example system
    """
    facts = []
    
    # Read input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip().lower()
        
        # Extract "X is a person" pattern
        person_match = re.match(r"(\w+) is a person", line)
        if person_match:
            person = person_match.group(1)
            facts.append(f"person({person}).")
        
        # Extract "X is married" pattern
        married_match = re.match(r"(\w+) is married", line)
        if married_match:
            person = married_match.group(1)
            facts.append(f"married({person}).")
        
        # Extract "X is married to Y" pattern
        married_to_match = re.match(r"(\w+) is married to (\w+)", line)
        if married_to_match:
            person1 = married_to_match.group(1)
            person2 = married_to_match.group(2)
            facts.append(f"married({person1}).")
            facts.append(f"married({person2}).")
            facts.append(f"married_to({person1},{person2}).")
        
        # Extract "X is the same person as Y" pattern
        same_match = re.match(r"(\w+) is the same person as (\w+)", line)
        if same_match:
            person1 = same_match.group(1)
            person2 = same_match.group(2)
            facts.append(f"same_as({person1},{person2}).")
        
        # Extract "X is the same as Y" pattern (alternative)
        same_alt_match = re.match(r"(\w+) is the same as (\w+)", line)
        if same_alt_match and not same_match:
            person1 = same_alt_match.group(1)
            person2 = same_alt_match.group(2)
            facts.append(f"same_as({person1},{person2}).")
    
    return facts

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) != 2:
        print("Usage: python update_custom_extractor.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    facts = extract_facts(input_file)
    
    print("Extracted facts:")
    for fact in facts:
        print(fact)
    
    # Write facts to output file
    output_file = "extracted_facts.lp"
    with open(output_file, 'w') as f:
        for fact in facts:
            f.write(fact + "\n")
    
    print(f"Facts written to {output_file}") 