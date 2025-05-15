#!/usr/bin/env python3
import logging
import sys
import argparse
import os

# Add parent directory to sys.path if running from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractors.enhanced_extractor import EnhancedFactExtractor
from src.solvers.advanced_solver import AdvancedSolver, Repair

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(input_file, rules_file="src/domain_rules/domain_rules.lp"):
    """Run a simple example of fact extraction and consistency checking"""
    print(f"Processing file: {input_file}")
    print(f"Using rules from: {rules_file}")
    
    # Step 1: Extract facts
    print("\n=== Step 1: Extracting facts ===")
    extractor = EnhancedFactExtractor(model_name="en_core_web_sm")
    facts = extractor.process_document(input_file)
    
    print(f"Extracted {len(facts)} facts:")
    for i, fact in enumerate(facts, 1):
        print(f"  {i}. {fact}")
    
    # Step 2: Check consistency
    print("\n=== Step 2: Checking consistency ===")
    solver = AdvancedSolver(rules_file=rules_file)
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
    parser = argparse.ArgumentParser(description="Process a document and check consistency with rules")
    parser.add_argument("input_file", help="Path to input document")
    parser.add_argument("--rules", "-r", dest="rules_file", default="src/domain_rules/domain_rules.lp",
                        help="Path to rules file (default: src/domain_rules/domain_rules.lp)")
    
    args = parser.parse_args()
    
    sys.exit(main(args.input_file, args.rules_file)) 