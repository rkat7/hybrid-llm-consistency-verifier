#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import json
import time
from typing import Dict, List, Optional

# Import our pipeline components
from pipeline import Pipeline, PipelineTask

def setup_logging(verbose: bool = False):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('nlp_asp.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from file or use defaults"""
    default_config = {
        "document_processing": {
            "spacy_model": "en_core_web_sm",  # Use small model by default for speed
            "use_gpu": False
        },
        "knowledge_update": {
            "db_path": "knowledge.db"
        },
        "consistency_check": {
            "rules_file": "domain_rules.lp",
            "repair_template": "repair.lp",
            "debug": False
        },
        "rule_generation": {
            "rules_dir": "rules",
            "llm_model": "gpt-3.5-turbo"
        }
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults
                for section, settings in loaded_config.items():
                    if section in default_config:
                        default_config[section].update(settings)
                    else:
                        default_config[section] = settings
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
            logging.warning("Using default configuration")
    
    return default_config

def process_document(args, config: Dict):
    """Process a document"""
    pipeline = Pipeline(config)
    pipeline.start(num_workers=args.workers)
    
    try:
        logging.info(f"Processing document: {args.doc_path}")
        task_id = pipeline.create_document_task(args.doc_path, generate_rules=args.generate_rules)
        
        logging.info(f"Task ID: {task_id}")
        logging.info(f"Waiting for processing to complete (timeout: {args.timeout}s)...")
        
        result = pipeline.wait_for_task(task_id, timeout=args.timeout)
        
        if result:
            if result['status'] == 'completed':
                print(f"\n✅ Document processed successfully")
                
                # Show extracted facts
                if 'facts' in result['output_data']:
                    facts = result['output_data']['facts']
                    print(f"\n📊 Extracted {len(facts)} facts:")
                    for i, fact in enumerate(facts[:10]):  # Show first 10
                        print(f"  {i+1}. {fact}")
                    if len(facts) > 10:
                        print(f"  ... and {len(facts) - 10} more facts")
                
                # Show knowledge graph stats
                if 'num_entities' in result['output_data'] and 'num_relations' in result['output_data']:
                    print(f"\n🔄 Knowledge graph updated:")
                    print(f"  • Entities: {result['output_data']['num_entities']}")
                    print(f"  • Relations: {result['output_data']['num_relations']}")
                
                # Show consistency results
                if 'consistency_status' in result['output_data']:
                    status = result['output_data']['consistency_status']
                    if status == "CONSISTENT":
                        print(f"\n✅ Facts are consistent with domain rules")
                    else:
                        print(f"\n❌ Facts are inconsistent with domain rules")
                        
                        if 'explanation' in result['output_data']:
                            explanation = result['output_data']['explanation']
                            print("\n📋 Explanation:")
                            print(f"  {explanation['description']}")
                        
                        if 'repairs' in result['output_data']:
                            repairs = result['output_data']['repairs']
                            print(f"\n🔧 {len(repairs)} repair suggestions:")
                            for i, repair in enumerate(repairs):
                                print(f"\n  Repair {i+1}: {repair['explanation']}")
                                print("    Removed facts:")
                                for fact in repair['removed_facts']:
                                    print(f"      - {fact}")
                                if repair['added_facts']:
                                    print("    Added facts:")
                                    for fact in repair['added_facts']:
                                        print(f"      + {fact}")
                
                # Show generated rules
                if 'rules' in result['output_data']:
                    rules = result['output_data']['rules']
                    print(f"\n📜 Generated {len(rules)} rules:")
                    for i, rule in enumerate(rules):
                        print(f"\n  Rule {i+1}: {rule['content']}")
                        print(f"  Description: {rule['description']}")
            
            elif result['status'] == 'failed':
                print(f"\n❌ Processing failed: {result['error']}")
                return 1
            else:
                print(f"\n⚠️ Processing status: {result['status']}")
                return 1
        else:
            print("\n⚠️ Processing timed out")
            return 1
    
    finally:
        pipeline.stop()
    
    return 0

def generate_rules(args, config: Dict):
    """Generate rules from a document"""
    pipeline = Pipeline(config)
    pipeline.start(num_workers=1)
    
    try:
        logging.info(f"Generating rules from document: {args.doc_path}")
        task_id = pipeline.create_rule_generation_task(
            args.doc_path, 
            num_rules=args.num_rules,
            category=args.category
        )
        
        logging.info(f"Task ID: {task_id}")
        logging.info(f"Waiting for processing to complete (timeout: {args.timeout}s)...")
        
        result = pipeline.wait_for_task(task_id, timeout=args.timeout)
        
        if result:
            if result['status'] == 'completed':
                print(f"\n✅ Rules generated successfully")
                
                # Show generated rules
                if 'rules' in result['output_data']:
                    rules = result['output_data']['rules']
                    print(f"\n📜 Generated {len(rules)} rules:")
                    for i, rule in enumerate(rules):
                        print(f"\n  Rule {i+1}: {rule['content']}")
                        print(f"  Description: {rule['description']}")
                
                # Show output file
                if 'rules_file' in result['output_data']:
                    print(f"\n💾 Rules saved to: {result['output_data']['rules_file']}")
            
            elif result['status'] == 'failed':
                print(f"\n❌ Rule generation failed: {result['error']}")
                return 1
            else:
                print(f"\n⚠️ Processing status: {result['status']}")
                return 1
        else:
            print("\n⚠️ Processing timed out")
            return 1
    
    finally:
        pipeline.stop()
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="NLP-ASP Pipeline CLI")
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('-c', '--config', type=str, help='Path to configuration file')
    parser.add_argument('-w', '--workers', type=int, default=2, help='Number of worker threads')
    parser.add_argument('-t', '--timeout', type=int, default=120, help='Processing timeout in seconds')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process document command
    process_parser = subparsers.add_parser('process', help='Process a document')
    process_parser.add_argument('doc_path', type=str, help='Path to document file')
    process_parser.add_argument('-g', '--generate-rules', action='store_true', help='Generate rules from document')
    
    # Generate rules command
    rule_parser = subparsers.add_parser('generate-rules', help='Generate rules from a document')
    rule_parser.add_argument('doc_path', type=str, help='Path to document file')
    rule_parser.add_argument('-n', '--num-rules', type=int, default=5, help='Number of rules to generate')
    rule_parser.add_argument('-cat', '--category', type=str, default='auto_generated', help='Category for generated rules')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute command
    if args.command == 'process':
        return process_document(args, config)
    elif args.command == 'generate-rules':
        return generate_rules(args, config)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 