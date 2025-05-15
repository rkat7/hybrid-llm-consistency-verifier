#!/usr/bin/env python3
import os
import json
import hashlib
import datetime
import openai
from typing import List, Dict, Optional, Union, Set

class Rule:
    def __init__(self, 
                content: str, 
                description: str = "", 
                category: str = "general", 
                source: str = "manual",
                confidence: float = 1.0,
                version: str = "1.0",
                rule_id: Optional[str] = None):
        self.content = content
        self.description = description
        self.category = category
        self.source = source  # manual, auto-generated, verified, etc.
        self.confidence = confidence  # 0.0 to 1.0
        self.version = version
        self.created_at = datetime.datetime.now().isoformat()
        # Generate a hash-based ID if not provided
        self.rule_id = rule_id or self._generate_id()
        self.metadata = {}
        
    def _generate_id(self) -> str:
        """Generate a unique ID for the rule based on its content"""
        return hashlib.md5(f"{self.content}_{self.created_at}".encode()).hexdigest()
    
    def to_asp(self) -> str:
        """Convert to ASP format with metadata as comments"""
        asp_comments = f"% Rule ID: {self.rule_id}\n"
        asp_comments += f"% Description: {self.description}\n"
        asp_comments += f"% Category: {self.category}\n"
        asp_comments += f"% Source: {self.source}\n"
        asp_comments += f"% Confidence: {self.confidence}\n"
        asp_comments += f"% Version: {self.version}\n"
        
        return f"{asp_comments}{self.content}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "rule_id": self.rule_id,
            "content": self.content,
            "description": self.description,
            "category": self.category,
            "source": self.source,
            "confidence": self.confidence,
            "version": self.version,
            "created_at": self.created_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Rule':
        """Create Rule object from dictionary"""
        rule = cls(
            content=data["content"],
            description=data.get("description", ""),
            category=data.get("category", "general"),
            source=data.get("source", "manual"),
            confidence=data.get("confidence", 1.0),
            version=data.get("version", "1.0"),
            rule_id=data.get("rule_id")
        )
        rule.created_at = data.get("created_at", rule.created_at)
        rule.metadata = data.get("metadata", {})
        return rule

class RuleManager:
    def __init__(self, 
                rules_dir: str = "rules",
                openai_api_key: Optional[str] = None,
                llm_model: str = "gpt-3.5-turbo"):
        self.rules_dir = rules_dir
        self.rules: Dict[str, Rule] = {}
        self.categories: Set[str] = set()
        self.index_path = os.path.join(rules_dir, "index.json")
        
        # Set up LLM integration
        if openai_api_key:
            openai.api_key = openai_api_key
        self.llm_model = llm_model
        
        # Create rules directory if it doesn't exist
        os.makedirs(rules_dir, exist_ok=True)
        
        # Load existing rules
        self._load_rules()
    
    def _load_rules(self):
        """Load rules from the rules directory"""
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                index = json.load(f)
                
            for rule_file in index.get("rules", []):
                rule_path = os.path.join(self.rules_dir, rule_file)
                if os.path.exists(rule_path):
                    with open(rule_path, 'r') as f:
                        rule_data = json.load(f)
                        rule = Rule.from_dict(rule_data)
                        self.rules[rule.rule_id] = rule
                        self.categories.add(rule.category)
    
    def _save_index(self):
        """Save the rule index"""
        index = {
            "rules": [f"{rule_id}.json" for rule_id in self.rules.keys()],
            "categories": list(self.categories),
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        with open(self.index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _save_rule(self, rule: Rule):
        """Save a rule to file"""
        rule_path = os.path.join(self.rules_dir, f"{rule.rule_id}.json")
        with open(rule_path, 'w') as f:
            json.dump(rule.to_dict(), f, indent=2)
    
    def add_rule(self, rule: Rule) -> str:
        """Add a new rule to the manager"""
        self.rules[rule.rule_id] = rule
        self.categories.add(rule.category)
        self._save_rule(rule)
        self._save_index()
        return rule.rule_id
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by ID"""
        return self.rules.get(rule_id)
    
    def update_rule(self, rule_id: str, **kwargs) -> Optional[Rule]:
        """Update a rule"""
        if rule_id in self.rules:
            rule_dict = self.rules[rule_id].to_dict()
            for key, value in kwargs.items():
                if key in rule_dict:
                    rule_dict[key] = value
            
            # Create a new rule version
            current_version = self.rules[rule_id].version
            try:
                major, minor = current_version.split('.')
                new_version = f"{major}.{int(minor) + 1}"
            except:
                new_version = "1.1"  # Fallback
            
            rule_dict["version"] = new_version
            
            updated_rule = Rule.from_dict(rule_dict)
            self.rules[rule_id] = updated_rule
            self._save_rule(updated_rule)
            self._save_index()
            return updated_rule
        return None
    
    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule"""
        if rule_id in self.rules:
            rule_path = os.path.join(self.rules_dir, f"{rule_id}.json")
            if os.path.exists(rule_path):
                os.remove(rule_path)
            del self.rules[rule_id]
            self._save_index()
            return True
        return False
    
    def get_rules_by_category(self, category: str) -> List[Rule]:
        """Get all rules in a category"""
        return [rule for rule in self.rules.values() if rule.category == category]
    
    def export_asp_rules(self, output_path: str, 
                        categories: Optional[List[str]] = None,
                        min_confidence: float = 0.0) -> int:
        """Export rules to an ASP file"""
        if categories is None:
            # Use all categories
            rules_to_export = [rule for rule in self.rules.values() 
                              if rule.confidence >= min_confidence]
        else:
            rules_to_export = [rule for rule in self.rules.values() 
                              if rule.category in categories and rule.confidence >= min_confidence]
        
        # Sort rules by category and confidence
        rules_to_export.sort(key=lambda r: (r.category, -r.confidence))
        
        with open(output_path, 'w') as f:
            f.write("% Auto-generated ASP rules\n")
            f.write(f"% Generated on: {datetime.datetime.now().isoformat()}\n\n")
            
            current_category = None
            for rule in rules_to_export:
                if rule.category != current_category:
                    current_category = rule.category
                    f.write(f"\n% === {current_category.upper()} RULES ===\n\n")
                
                f.write(f"{rule.to_asp()}\n\n")
        
        return len(rules_to_export)
    
    async def generate_rules_from_text(self, 
                               text: str, 
                               num_rules: int = 5, 
                               category: str = "auto_generated",
                               existing_rules: Optional[List[str]] = None) -> List[Rule]:
        """Generate new rules from text using LLM"""
        if existing_rules is None:
            # Get all existing rules content
            existing_rules = [rule.content for rule in self.rules.values()]
        
        # Prepare prompt
        prompt = [
            {"role": "system", "content": 
             "You are an expert in Answer Set Programming (ASP) and logical reasoning. "
             "Your task is to generate meaningful ASP rules and constraints based on the provided text. "
             "Rules should be syntactically correct ASP code and capture logical relationships or constraints."
            },
            {"role": "user", "content": 
             f"Based on the following text, generate {num_rules} ASP rules or constraints that would be useful "
             f"for catching logical inconsistencies. Each rule should have a brief description explaining its purpose.\n\n"
             f"TEXT: {text}\n\n"
             f"EXISTING RULES:\n" + "\n".join(existing_rules) + "\n\n"
             f"Generate {num_rules} NEW rules in the following JSON format: "
             f"[{{'content': 'ASP rule code here', 'description': 'Brief explanation'}}]"
            }
        ]
        
        # Call GPT API
        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=prompt,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Parse the response
        output = response.choices[0].message.content
        try:
            # Extract JSON part if response contains explanatory text
            if "[" in output and "]" in output:
                json_part = output[output.find("["):output.rfind("]")+1]
                rules_data = json.loads(json_part)
            else:
                rules_data = json.loads(output)
            
            # Create Rule objects
            new_rules = []
            for data in rules_data:
                rule = Rule(
                    content=data["content"],
                    description=data["description"],
                    category=category,
                    source="llm_generated",
                    confidence=0.8  # Default confidence for LLM-generated rules
                )
                # Add the rule to the manager
                self.add_rule(rule)
                new_rules.append(rule)
            
            return new_rules
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            # Try to extract rule content and descriptions directly
            lines = output.split("\n")
            new_rules = []
            current_content = None
            current_description = None
            
            for line in lines:
                line = line.strip()
                if line.startswith("%") or not line:
                    continue
                
                if ":-" in line or line.endswith("."):
                    # This looks like ASP rule content
                    if current_content is not None and current_content != line:
                        # Save previous rule
                        rule = Rule(
                            content=current_content,
                            description=current_description or "",
                            category=category,
                            source="llm_generated",
                            confidence=0.7  # Lower confidence for parsed rules
                        )
                        self.add_rule(rule)
                        new_rules.append(rule)
                    
                    current_content = line
                    current_description = None
                elif current_content and not current_description:
                    # This might be a description
                    current_description = line
            
            # Add the last rule if any
            if current_content:
                rule = Rule(
                    content=current_content,
                    description=current_description or "",
                    category=category,
                    source="llm_generated",
                    confidence=0.7
                )
                self.add_rule(rule)
                new_rules.append(rule)
            
            return new_rules
    
    def verify_rule(self, rule_id: str, examples: List[Dict[str, str]]) -> Dict:
        """Verify a rule against examples and update its confidence"""
        rule = self.get_rule(rule_id)
        if not rule:
            return {"success": False, "error": "Rule not found"}
        
        # Verify rule with positive and negative examples
        results = {
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        # Implement verification logic based on examples
        # Each example should have "facts", "expected" (consistent/inconsistent)
        
        # Update rule confidence based on verification results
        if results["passed"] + results["failed"] > 0:
            confidence = results["passed"] / (results["passed"] + results["failed"])
            self.update_rule(rule_id, confidence=confidence)
            results["new_confidence"] = confidence
        
        return {"success": True, "results": results}

if __name__ == "__main__":
    # Example usage
    manager = RuleManager()
    
    # Create some example rules
    rule1 = Rule(
        content="married(X) :- spouse(X, Y).",
        description="If X has a spouse Y, then X is married",
        category="relationships"
    )
    
    rule2 = Rule(
        content=":- married(X), single(X).",
        description="A person cannot be both married and single",
        category="relationships"
    )
    
    rule3 = Rule(
        content="parent(X, Y) :- father(X, Y).",
        description="If X is the father of Y, then X is a parent of Y",
        category="family"
    )
    
    # Add rules to manager
    manager.add_rule(rule1)
    manager.add_rule(rule2)
    manager.add_rule(rule3)
    
    # Export rules to ASP file
    num_exported = manager.export_asp_rules("domain_rules_extended.lp")
    print(f"Exported {num_exported} rules to domain_rules_extended.lp") 