#!/usr/bin/env python3
import logging
from enhanced_extractor import EnhancedFactExtractor
from advanced_solver import AdvancedSolver
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(input_file, max_paragraphs=5):
    """Process a large file by analyzing only a portion"""
    print(f"Processing file: {input_file} (max {max_paragraphs} paragraphs)")
    
    # Read only the first part of the file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into paragraphs and limit to max_paragraphs
    paragraphs = content.split('\n\n')
    limited_content = '\n\n'.join(paragraphs[:max_paragraphs])
    
    # Write to a temporary file
    temp_file = f"temp_{input_file.split('/')[-1]}"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(limited_content)
    
    print(f"Created temporary file with {max_paragraphs} paragraphs: {temp_file}")
    
    # Step 1: Extract facts
    print("\n=== Step 1: Extracting facts ===")
    extractor = EnhancedFactExtractor(model_name="en_core_web_sm")
    facts = extractor.process_document(temp_file)
    
    print(f"Extracted {len(facts)} facts:")
    for i, fact in enumerate(facts, 1):
        print(f"  {i}. {fact}")
    
    # Step 2: Check consistency
    print("\n=== Step 2: Checking consistency ===")
    solver = AdvancedSolver(rules_file="domain_rules.lp")
    result = solver.check_consistency(facts)
    
    if result["status"] == "CONSISTENT":
        print("✅ Facts are consistent with domain rules")
    else:
        print("❌ Facts are inconsistent with domain rules")
        
        if result["explanation"]:
            print("\nExplanation:")
            print(result["explanation"]["description"])
            
            # Step 3: Generate repair suggestions
            print("\n=== Step 3: Generating repair suggestions ===")
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
    
    print("\nDone!")
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_large_file.py <input_file> [max_paragraphs]")
        print("Example: python process_large_file.py corpus/austen-emma.txt 5")
        sys.exit(1)
    
    input_file = sys.argv[1]
    max_paragraphs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    sys.exit(main(input_file, max_paragraphs)) 