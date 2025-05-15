#!/usr/bin/env python3
import re
from typing import List

def extract_facts(input_file: str) -> List[str]:
    """
    An extended pattern-based fact extractor for the advanced domain rules
    """
    facts = []
    entities = set()  # Track all entities
    
    # Read input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip().lower()
        
        # Basic identity
        # Extract "X is a person" pattern
        person_match = re.match(r"(\w+) is a person", line)
        if person_match:
            person = person_match.group(1)
            facts.append(f"person({person}).")
            entities.add(person)
        
        # Gender
        # Extract "X is male" pattern
        male_match = re.match(r"(\w+) is male", line)
        if male_match:
            person = male_match.group(1)
            facts.append(f"male({person}).")
            entities.add(person)
        
        # Extract "X is female" pattern
        female_match = re.match(r"(\w+) is female", line)
        if female_match:
            person = female_match.group(1)
            facts.append(f"female({person}).")
            entities.add(person)
        
        # Marriage relationships
        # Extract "X is married" pattern
        married_match = re.match(r"(\w+) is married", line)
        if married_match:
            person = married_match.group(1)
            facts.append(f"married({person}).")
            entities.add(person)
        
        # Extract "X is married to Y" pattern
        married_to_match = re.match(r"(\w+) is married to (\w+)", line)
        if married_to_match:
            person1 = married_to_match.group(1)
            person2 = married_to_match.group(2)
            facts.append(f"married_to({person1},{person2}).")
            entities.add(person1)
            entities.add(person2)
        
        # Family relationships
        # Extract "X is the parent of Y" pattern
        parent_match = re.match(r"(\w+) is the parent of (\w+)", line)
        if parent_match:
            parent = parent_match.group(1)
            child = parent_match.group(2)
            facts.append(f"parent_of({parent},{child}).")
            entities.add(parent)
            entities.add(child)
        
        # Extract "X is the child of Y" pattern
        child_match = re.match(r"(\w+) is the child of (\w+)", line)
        if child_match:
            child = child_match.group(1)
            parent = child_match.group(2)
            facts.append(f"child_of({child},{parent}).")
            entities.add(child)
            entities.add(parent)
        
        # Identity relationships
        # Extract "X is the same person as Y" pattern
        same_match = re.match(r"(\w+) is the same person as (\w+)", line)
        if same_match:
            person1 = same_match.group(1)
            person2 = same_match.group(2)
            facts.append(f"same_as({person1},{person2}).")
            entities.add(person1)
            entities.add(person2)
        
        # Extract "X is the same as Y" pattern (alternative)
        same_alt_match = re.match(r"(\w+) is the same as (\w+)", line)
        if same_alt_match and not same_match:
            person1 = same_alt_match.group(1)
            person2 = same_alt_match.group(2)
            facts.append(f"same_as({person1},{person2}).")
            entities.add(person1)
            entities.add(person2)
        
        # Location and occupation
        # Extract "X lives in Y" pattern
        lives_in_match = re.match(r"(\w+) lives in (\w+)", line)
        if lives_in_match:
            person = lives_in_match.group(1)
            location = lives_in_match.group(2)
            facts.append(f"lives_in({person},{location}).")
            entities.add(person)
        
        # Extract "X works as a Y" pattern
        occupation_match = re.match(r"(\w+) works as a (\w+)", line)
        if occupation_match:
            person = occupation_match.group(1)
            job = occupation_match.group(2)
            facts.append(f"occupation({person},{job}).")
            entities.add(person)
        
        # Extract "X is a minor" pattern
        minor_match = re.match(r"(\w+) is a minor", line)
        if minor_match:
            person = minor_match.group(1)
            facts.append(f"minor({person}).")
            entities.add(person)
    
    # Ensure all referenced entities are declared as persons
    for entity in entities:
        person_fact = f"person({entity})."
        if person_fact not in facts:
            facts.append(person_fact)
    
    return facts

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) != 2:
        print("Usage: python extended_custom_extractor.py <input_file>")
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