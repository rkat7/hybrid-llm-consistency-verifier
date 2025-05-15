#!/usr/bin/env python3
import os
import tempfile
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple, Optional, Union
from clingo import Control, Function, Number, String, Symbol, parse_term
from clingo.symbol import SymbolType

class Explanation:
    """Class to represent an explanation for an inconsistency"""
    def __init__(self, facts: List[str], rules: List[str], description: str = ""):
        self.facts = facts
        self.rules = rules
        self.description = description
    
    def to_dict(self) -> Dict:
        return {
            "facts": self.facts,
            "rules": self.rules,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Explanation':
        return cls(
            facts=data.get("facts", []),
            rules=data.get("rules", []),
            description=data.get("description", "")
        )

class Repair:
    """Class to represent a repair suggestion"""
    def __init__(self, 
                original_facts: List[str], 
                repaired_facts: List[str],
                removed_facts: List[str] = None,
                added_facts: List[str] = None,
                confidence: float = 1.0,
                explanation: str = ""):
        self.original_facts = original_facts
        self.repaired_facts = repaired_facts
        self.removed_facts = removed_facts or []
        self.added_facts = added_facts or []
        self.confidence = confidence
        self.explanation = explanation
    
    def to_dict(self) -> Dict:
        return {
            "original_facts": self.original_facts,
            "repaired_facts": self.repaired_facts,
            "removed_facts": self.removed_facts,
            "added_facts": self.added_facts,
            "confidence": self.confidence,
            "explanation": self.explanation
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Repair':
        return cls(
            original_facts=data.get("original_facts", []),
            repaired_facts=data.get("repaired_facts", []),
            removed_facts=data.get("removed_facts", []),
            added_facts=data.get("added_facts", []),
            confidence=data.get("confidence", 1.0),
            explanation=data.get("explanation", "")
        )

class AdvancedSolver:
    def __init__(self, 
                rules_file: str,
                repair_template_file: Optional[str] = None,
                debug: bool = False):
        self.rules_file = rules_file
        self.repair_template_file = repair_template_file
        self.debug = debug
        self.fact_graph = nx.DiGraph()
        
        # Load repair template if provided
        self.repair_template = None
        if repair_template_file and os.path.exists(repair_template_file):
            with open(repair_template_file, 'r') as f:
                self.repair_template = f.read()
    
    def _create_temp_file(self, content: str) -> str:
        """Create a temporary file with given content"""
        fd, path = tempfile.mkstemp(suffix=".lp")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path
    
    def _parse_fact(self, fact: str) -> Tuple[str, List[str]]:
        """Parse a fact into predicate and arguments"""
        fact = fact.strip()
        if fact.endswith("."):
            fact = fact[:-1]  # Remove trailing dot
            
        if "(" not in fact:
            return fact, []
            
        predicate = fact[:fact.find("(")]
        args_str = fact[fact.find("(")+1:fact.rfind(")")]
        args = [arg.strip() for arg in args_str.split(",")]
        
        return predicate, args
    
    def _build_fact_graph(self, facts: List[str]):
        """Build a graph of relationships between facts"""
        self.fact_graph = nx.DiGraph()
        
        # Add all facts as nodes
        for fact in facts:
            pred, args = self._parse_fact(fact)
            self.fact_graph.add_node(fact, predicate=pred, arguments=args)
        
        # Connect facts with shared arguments
        fact_args = {}  # Maps arguments to facts that use them
        
        for fact in facts:
            pred, args = self._parse_fact(fact)
            
            for arg in args:
                if arg not in fact_args:
                    fact_args[arg] = []
                fact_args[arg].append(fact)
        
        # Create edges between facts sharing arguments
        for arg, related_facts in fact_args.items():
            for i in range(len(related_facts)):
                for j in range(i+1, len(related_facts)):
                    fact1 = related_facts[i]
                    fact2 = related_facts[j]
                    # Add bidirectional edges
                    self.fact_graph.add_edge(fact1, fact2, shared_arg=arg)
                    self.fact_graph.add_edge(fact2, fact1, shared_arg=arg)
    
    def _visualize_fact_graph(self, output_path: str = "fact_graph.png"):
        """Visualize the fact graph"""
        if len(self.fact_graph) == 0:
            return
            
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(self.fact_graph)
        
        # Draw nodes with different colors based on predicate
        predicates = set(nx.get_node_attributes(self.fact_graph, 'predicate').values())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(predicates)))
        color_map = {pred: colors[i] for i, pred in enumerate(predicates)}
        
        for pred in predicates:
            pred_nodes = [n for n, d in self.fact_graph.nodes(data=True) if d.get('predicate') == pred]
            nx.draw_networkx_nodes(self.fact_graph, pos, 
                                  nodelist=pred_nodes,
                                  node_color=[color_map[pred]],
                                  node_size=500,
                                  alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.fact_graph, pos, width=1.0, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(self.fact_graph, pos, font_size=8)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def check_consistency(self, facts: List[str]) -> Dict:
        """Check consistency of facts against rules"""
        # Prepare control object
        ctl = Control(["--warn=none"])
        
        # Add rules
        ctl.load(self.rules_file)
        
        # Add facts to a temporary file
        facts_content = "\n".join(facts)
        facts_file = self._create_temp_file(facts_content)
        
        try:
            # Add facts
            ctl.add("base", [], f'#include "{facts_file}".')
            
            # Ground and solve
            ctl.ground([("base", [])])
            result = ctl.solve()
            
            # Build the fact relationship graph
            self._build_fact_graph(facts)
            
            if self.debug:
                self._visualize_fact_graph()
            
            if result.satisfiable:
                return {
                    "status": "CONSISTENT",
                    "explanation": None
                }
            else:
                # Find explanation for inconsistency
                explanation = self._explain_inconsistency(facts)
                return {
                    "status": "INCONSISTENT",
                    "explanation": explanation.to_dict() if explanation else None
                }
        finally:
            # Clean up temporary file
            os.remove(facts_file)
    
    def _explain_inconsistency(self, facts: List[str]) -> Optional[Explanation]:
        """Find an explanation for the inconsistency"""
        explanation = None
        
        # Try to identify minimal inconsistent subset
        inconsistent_facts = self._find_minimal_inconsistent_subset(facts)
        
        if inconsistent_facts:
            # Identify rules that likely caused the inconsistency
            rules = self._identify_relevant_rules(inconsistent_facts)
            
            # Generate natural language explanation
            description = self._generate_explanation_text(inconsistent_facts, rules)
            
            explanation = Explanation(
                facts=inconsistent_facts,
                rules=rules,
                description=description
            )
        
        return explanation
    
    def _find_minimal_inconsistent_subset(self, facts: List[str]) -> List[str]:
        """Use a binary search approach to find a minimal inconsistent subset"""
        if not facts:
            return []
            
        # If the entire set is consistent, there's no inconsistency
        if self.check_consistency(facts)["status"] == "CONSISTENT":
            return []
        
        # If there's only one fact, it's the culprit (with rules)
        if len(facts) == 1:
            return facts
        
        # Try to find a minimal inconsistent subset using binary search
        mid = len(facts) // 2
        first_half = facts[:mid]
        second_half = facts[mid:]
        
        # Check if first half is inconsistent
        first_half_result = self.check_consistency(first_half)
        if first_half_result["status"] == "INCONSISTENT":
            return self._find_minimal_inconsistent_subset(first_half)
        
        # Check if second half is inconsistent
        second_half_result = self.check_consistency(second_half)
        if second_half_result["status"] == "INCONSISTENT":
            return self._find_minimal_inconsistent_subset(second_half)
        
        # If neither half is inconsistent alone, the inconsistency spans both halves
        # Try to find the minimal set across both halves
        return self._find_cross_inconsistency(first_half, second_half)
    
    def _find_cross_inconsistency(self, facts1: List[str], facts2: List[str]) -> List[str]:
        """Find the minimal inconsistent subset that spans across two sets of facts"""
        # Try removing facts from the first set
        for i, fact in enumerate(facts1):
            reduced = facts1[:i] + facts1[i+1:] + facts2
            if self.check_consistency(reduced)["status"] == "INCONSISTENT":
                return self._find_minimal_inconsistent_subset(reduced)
        
        # Try removing facts from the second set
        for i, fact in enumerate(facts2):
            reduced = facts1 + facts2[:i] + facts2[i+1:]
            if self.check_consistency(reduced)["status"] == "INCONSISTENT":
                return self._find_minimal_inconsistent_subset(reduced)
        
        # If no reduction works, return the combined set
        return facts1 + facts2
    
    def _identify_relevant_rules(self, facts: List[str]) -> List[str]:
        """Identify the rules most likely involved in the inconsistency"""
        relevant_rules = []
        
        # Parse facts to get predicates
        predicates = set()
        for fact in facts:
            pred, _ = self._parse_fact(fact)
            predicates.add(pred)
        
        # Read rules file and filter rules that involve the predicates
        with open(self.rules_file, 'r') as f:
            lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip comments and empty lines
                if line.startswith("%") or not line:
                    i += 1
                    continue
                
                # Check if this line contains a rule
                if ":-" in line or any(pred in line for pred in predicates):
                    # Collect full rule (might span multiple lines)
                    rule = line
                    while not rule.endswith(".") and i + 1 < len(lines):
                        i += 1
                        rule += " " + lines[i].strip()
                    
                    # Check if any predicates are in the rule
                    if any(pred in rule for pred in predicates):
                        relevant_rules.append(rule)
                
                i += 1
        
        return relevant_rules
    
    def _generate_explanation_text(self, facts: List[str], rules: List[str]) -> str:
        """Generate a natural language explanation for the inconsistency"""
        if not facts:
            return "Unable to determine the cause of inconsistency."
        
        # Simple explanation based on facts and rules
        explanation = "The following set of facts is inconsistent with the rules:\n"
        for fact in facts:
            explanation += f"- {fact}\n"
        
        if rules:
            explanation += "\nThe following rules are likely involved in the inconsistency:\n"
            for rule in rules:
                explanation += f"- {rule}\n"
        
        return explanation
    
    def suggest_repairs(self, facts: List[str], max_repairs: int = 3) -> List[Repair]:
        """Suggest repairs to make the facts consistent"""
        # Check if facts are already consistent
        result = self.check_consistency(facts)
        if result["status"] == "CONSISTENT":
            return []
        
        repairs = []
        
        # If we have a repair template, use it
        if self.repair_template:
            repairs = self._repair_with_template(facts, max_repairs)
        
        # If no repairs or no template, use a simple repair strategy
        if not repairs:
            repairs = self._simple_repair_strategy(facts, max_repairs)
        
        return repairs
    
    def _repair_with_template(self, facts: List[str], max_repairs: int) -> List[Repair]:
        """Use the repair template to generate repairs"""
        # Create a temporary file with facts
        facts_content = "\n".join(f"fact({fact.strip()})." if not fact.startswith("fact(") else fact.strip() 
                                for fact in facts)
        facts_file = self._create_temp_file(facts_content)
        
        repairs = []
        try:
            # Configure ASP solver with repair template
            ctl = Control(["--warn=none", "--opt-mode=optN", f"--opt-bound={max_repairs}"])
            ctl.load(self.rules_file)
            ctl.add("base", [], f'#include "{facts_file}".')
            ctl.add("base", [], self.repair_template)
            
            # Ground and solve
            ctl.ground([("base", [])])
            
            # Get all optimal repair models
            handle = ctl.solve(yield_=True)
            
            with handle as h:
                for model in h:
                    dropped_facts = []
                    added_facts = []
                    
                    # Extract dropped facts
                    for symbol in model.symbols(shown=True):
                        if symbol.name == "drop" and len(symbol.arguments) == 1:
                            arg = symbol.arguments[0]
                            if arg.type == SymbolType.Function:
                                dropped_fact = str(arg).replace("fact(", "").replace(")", "")
                                dropped_facts.append(dropped_fact)
                    
                    # Calculate repaired facts
                    repaired_facts = [f for f in facts if f not in dropped_facts] + added_facts
                    
                    # Create repair object
                    repair = Repair(
                        original_facts=facts,
                        repaired_facts=repaired_facts,
                        removed_facts=dropped_facts,
                        added_facts=added_facts,
                        confidence=1.0,
                        explanation=f"Removed {len(dropped_facts)} conflicting facts to restore consistency."
                    )
                    
                    repairs.append(repair)
                    
                    if len(repairs) >= max_repairs:
                        break
        finally:
            # Clean up
            os.remove(facts_file)
        
        return repairs
    
    def _simple_repair_strategy(self, facts: List[str], max_repairs: int) -> List[Repair]:
        """Simple repair strategy: remove facts from the minimal inconsistent subset"""
        # Find minimal inconsistent subset
        inconsistent_subset = self._find_minimal_inconsistent_subset(facts)
        
        if not inconsistent_subset:
            return []
        
        repairs = []
        
        # Generate repairs by removing each fact from the inconsistent subset
        for i, fact_to_remove in enumerate(inconsistent_subset):
            if i >= max_repairs:
                break
                
            repaired_facts = [f for f in facts if f != fact_to_remove]
            
            # Verify that the repair works
            if self.check_consistency(repaired_facts)["status"] == "CONSISTENT":
                repair = Repair(
                    original_facts=facts,
                    repaired_facts=repaired_facts,
                    removed_facts=[fact_to_remove],
                    added_facts=[],
                    confidence=1.0,
                    explanation=f"Removed contradictory fact: {fact_to_remove}"
                )
                repairs.append(repair)
        
        # If single removals don't work, try removing pairs
        if not repairs and len(inconsistent_subset) > 1:
            for i in range(len(inconsistent_subset)):
                for j in range(i+1, len(inconsistent_subset)):
                    if len(repairs) >= max_repairs:
                        break
                        
                    fact1 = inconsistent_subset[i]
                    fact2 = inconsistent_subset[j]
                    repaired_facts = [f for f in facts if f != fact1 and f != fact2]
                    
                    # Verify that the repair works
                    if self.check_consistency(repaired_facts)["status"] == "CONSISTENT":
                        repair = Repair(
                            original_facts=facts,
                            repaired_facts=repaired_facts,
                            removed_facts=[fact1, fact2],
                            added_facts=[],
                            confidence=0.8,  # Lower confidence for multi-fact repairs
                            explanation=f"Removed contradictory facts: {fact1} and {fact2}"
                        )
                        repairs.append(repair)
                
                if len(repairs) >= max_repairs:
                    break
        
        return repairs

if __name__ == "__main__":
    import numpy as np  # For visualization color map
    
    # Example usage
    solver = AdvancedSolver(
        rules_file="domain_rules.lp",
        repair_template_file="repair.lp",
        debug=True
    )
    
    # Example facts (some inconsistent)
    facts = [
        "bird(tweety).",
        "not_flies(tweety).",
        "married(john).",
        "single(john)."
    ]
    
    # Check consistency
    result = solver.check_consistency(facts)
    print(f"Status: {result['status']}")
    
    if result["status"] == "INCONSISTENT" and result["explanation"]:
        print("\nExplanation:")
        print(result["explanation"]["description"])
    
    # Suggest repairs
    repairs = solver.suggest_repairs(facts)
    print(f"\nFound {len(repairs)} possible repairs:")
    
    for i, repair in enumerate(repairs):
        print(f"\nRepair option {i+1}:")
        print(f"Explanation: {repair.explanation}")
        print("Removed facts:")
        for fact in repair.removed_facts:
            print(f"  - {fact}")
        if repair.added_facts:
            print("Added facts:")
            for fact in repair.added_facts:
                print(f"  + {fact}") 