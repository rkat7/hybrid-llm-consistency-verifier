#!/usr/bin/env python3
import sys
import json
import re
import datetime
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_rules(input_file, num_rules=5, category="auto_generated"):
    """Generate sample rules based on input file content"""
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract entities from the text (very basic extraction)
        entities = []
        for line in content.split('\n'):
            # Look for patterns like "X is Y"
            match = re.search(r'(\w+) is (\w+)', line)
            if match:
                entities.append((match.group(1).lower(), match.group(2).lower()))
        
        # Generate rules based on extracted entities
        rules = []
        
        # Rule type 1: Entity cannot be both X and Y (contradictory attributes)
        attributes = [entity[1] for entity in entities]
        for i in range(min(num_rules, len(attributes))):
            attr1 = attributes[i]
            # Pick a different attribute for contradiction
            other_attrs = [a for a in attributes if a != attr1]
            if other_attrs:
                attr2 = random.choice(other_attrs)
                rule = {
                    "content": f":- {attr1}(X), {attr2}(X).",
                    "description": f"An entity cannot be both {attr1} and {attr2} at the same time."
                }
                rules.append(rule)
        
        # Rule type 2: If X is Y then X is also Z (implication)
        if len(rules) < num_rules and len(attributes) > 1:
            attr1 = random.choice(attributes)
            attr2 = random.choice([a for a in attributes if a != attr1])
            rule = {
                "content": f"{attr2}(X) :- {attr1}(X).",
                "description": f"If an entity is {attr1}, then it is also {attr2}."
            }
            rules.append(rule)
        
        # Rule type 3: Count constraint
        if len(rules) < num_rules:
            entity_types = list(set([entity[1] for entity in entities]))
            if entity_types:
                entity_type = random.choice(entity_types)
                count = random.randint(2, 5)
                rule = {
                    "content": f":- not {{ {entity_type}(X) }} = {count}.",
                    "description": f"There must be exactly {count} entities that are {entity_type}."
                }
                rules.append(rule)
        
        # Fill remaining rules with generic constraints
        generic_rules = [
            {
                "content": ":- person(X), animal(X).",
                "description": "An entity cannot be both a person and an animal."
            },
            {
                "content": "adult(X) :- person(X), not child(X).",
                "description": "A person is an adult if they are not a child."
            },
            {
                "content": ":- parent(X, Y), child(X).",
                "description": "A child cannot be a parent."
            },
            {
                "content": "sibling(X, Y) :- parent(Z, X), parent(Z, Y), X != Y.",
                "description": "Two different people with the same parent are siblings."
            },
            {
                "content": ":- married(X, Y), not person(X).",
                "description": "Only persons can be married."
            }
        ]
        
        while len(rules) < num_rules and generic_rules:
            rules.append(generic_rules.pop(0))
        
        # Write rules to file
        current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"generated_rules_{current_date}.lp"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"% Auto-generated rules from {input_file}\n")
            f.write(f"% Generated on: {datetime.datetime.now().isoformat()}\n")
            f.write(f"% Category: {category}\n\n")
            
            for i, rule in enumerate(rules, 1):
                f.write(f"% Rule {i}: {rule['description']}\n")
                f.write(f"{rule['content']}\n\n")
            
            # Add show directives
            f.write("% Show directives\n")
            predicates = set()
            for rule in rules:
                # Extract predicates from rule content
                matches = re.findall(r'([a-zA-Z_]+)\([^)]*\)', rule['content'])
                predicates.update(matches)
            
            for predicate in predicates:
                if predicate not in ["not"]:  # Skip special keywords
                    f.write(f"#show {predicate}/1.\n")
        
        print(f"Generated {len(rules)} rules and saved to {output_file}")
        print("\nGenerated rules:")
        for i, rule in enumerate(rules, 1):
            print(f"\nRule {i}:")
            print(f"Content: {rule['content']}")
            print(f"Description: {rule['description']}")
        
        return output_file, rules
        
    except Exception as e:
        logger.error(f"Error generating rules: {e}")
        return None, []

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_rules.py <input_file> [num_rules] [category]")
        print("Example: python generate_rules.py input.txt 5 custom_rules")
        sys.exit(1)
    
    input_file = sys.argv[1]
    num_rules = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    category = sys.argv[3] if len(sys.argv) > 3 else "auto_generated"
    
    output_file, rules = generate_sample_rules(input_file, num_rules, category)
    if not output_file:
        sys.exit(1)
    
    print(f"\nTo use these rules, add them to your domain_rules.lp file or use them directly:")
    print(f"python simple_example.py {input_file} --rules {output_file}")
    
    sys.exit(0) 