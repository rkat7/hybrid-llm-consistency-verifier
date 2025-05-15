#!/usr/bin/env python3
import spacy
from spacy.tokens import Doc
import json
from typing import List, Dict, Optional, Tuple, Set
import re
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import neuralcoref but make it optional
try:
    import neuralcoref
    NEURALCOREF_AVAILABLE = True
except ImportError:
    logger.warning("neuralcoref is not available. Coreference resolution will be disabled.")
    NEURALCOREF_AVAILABLE = False

class EnhancedFactExtractor:
    def __init__(self, model_name="en_core_web_sm", use_gpu=True):
        # Load SpaCy model with transformers
        self.nlp = spacy.load(model_name)
        
        # Add neural coreference resolution if available
        if NEURALCOREF_AVAILABLE:
            if not Doc.has_extension("coref_clusters"):
                neuralcoref.add_to_pipe(self.nlp)
            logger.info("Coreference resolution enabled")
        else:
            logger.info("Coreference resolution disabled")
        
        # We won't try to load the relation extraction model as it's not essential
        # and causes issues with dependencies
        self.relation_extractor = None
        
        # Temporal extraction patterns
        self.temporal_patterns = [
            {"label": "DATE", "pattern": [{"LOWER": {"REGEX": r"(january|february|march|april|may|june|july|august|september|october|november|december)"}}]},
            {"label": "DURATION", "pattern": [{"LOWER": {"REGEX": r"\d+"}, "OP": "?"}, {"LOWER": {"IN": ["day", "days", "week", "weeks", "month", "months", "year", "years"]}}]},
            {"label": "TIME", "pattern": [{"SHAPE": {"REGEX": r"\d?\d:\d\d"}}]}
        ]
        
        # Check if entity_ruler already exists in pipeline
        if "entity_ruler" not in self.nlp.pipe_names:
            entity_ruler = self.nlp.add_pipe("entity_ruler")
            for pattern in self.temporal_patterns:
                entity_ruler.add_patterns([pattern])
        else:
            # Get existing entity ruler
            entity_ruler = self.nlp.get_pipe("entity_ruler")
            for pattern in self.temporal_patterns:
                entity_ruler.add_patterns([pattern])

        self.fact_types = {
            "entity": self._extract_entities,
            "relation": self._extract_relations,
            "temporal": self._extract_temporal_facts,
            "quantitative": self._extract_quantitative,
            "qualitative": self._extract_qualitative
        }

    def _resolve_coreferences(self, doc) -> Doc:
        """Resolve coreferences in text to improve entity tracking"""
        if NEURALCOREF_AVAILABLE and hasattr(doc._, 'has_coref') and doc._.has_coref:
            # Replace all mentions with their most representative mention
            for cluster in doc._.coref_clusters:
                for mention in cluster.mentions:
                    if mention != cluster.main:
                        if not mention._.has_extension("resolved"):
                            mention._.set_extension("resolved", default=None)
                        mention._.resolved = cluster.main.text
        return doc

    def _extract_entities(self, doc) -> List[str]:
        """Extract entities as ASP facts"""
        facts = []
        for ent in doc.ents:
            # Convert entity to ASP fact: person(john). location(paris).
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT"]:
                entity_type = ent.label_.lower()
                entity_name = re.sub(r'\W+', '_', ent.text.lower())
                facts.append(f"{entity_type}({entity_name}).")
        return facts

    def _extract_relations(self, doc) -> List[str]:
        """Extract relations between entities"""
        facts = []
        # Extract subject-verb-object patterns
        for sent in doc.sents:
            subjects = []
            relation = None
            objects = []
            
            for token in sent:
                # Find subjects
                if token.dep_ in ("nsubj", "nsubjpass"):
                    # Get the full noun phrase
                    subjects.append(token.text.lower())
                
                # Find relation (verb)
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    relation = token.lemma_.lower()
                
                # Find objects
                if token.dep_ in ("dobj", "pobj", "attr"):
                    objects.append(token.text.lower())
            
            # Create facts from SVO triples
            if subjects and relation and objects:
                for subj in subjects:
                    for obj in objects:
                        subj_clean = re.sub(r'\W+', '_', subj)
                        obj_clean = re.sub(r'\W+', '_', obj)
                        facts.append(f"{relation}({subj_clean},{obj_clean}).")
        
        return facts

    def _extract_temporal_facts(self, doc) -> List[str]:
        """Extract temporal facts (dates, durations, sequences)"""
        facts = []
        temporal_ents = [ent for ent in doc.ents if ent.label_ in ["DATE", "TIME", "DURATION"]]
        
        # Extract temporal facts
        for ent in temporal_ents:
            # Skip if part of larger entity
            if any(ent.start >= e.start and ent.end <= e.end and ent != e for e in doc.ents):
                continue
                
            # Find associated actions/events
            related_verbs = []
            for token in doc:
                if token.pos_ == "VERB" and any(t in token.children for t in ent):
                    related_verbs.append(token)
            
            # Create temporal facts
            for verb in related_verbs:
                verb_text = verb.lemma_.lower()
                time_text = re.sub(r'\W+', '_', ent.text.lower())
                facts.append(f"at_time({verb_text},{time_text}).")
                
        return facts

    def _extract_quantitative(self, doc) -> List[str]:
        """Extract quantitative facts (numbers, measurements, comparisons)"""
        facts = []
        
        # Find numbers and their associated nouns
        for token in doc:
            if token.like_num:
                # Look for associated nouns
                for child in token.children:
                    if child.pos_ in ["NOUN", "PROPN"]:
                        quantity = token.text
                        entity = re.sub(r'\W+', '_', child.text.lower())
                        facts.append(f"quantity({entity},{quantity}).")
                
                # Look for parent nouns
                if token.head.pos_ in ["NOUN", "PROPN"]:
                    quantity = token.text
                    entity = re.sub(r'\W+', '_', token.head.text.lower())
                    facts.append(f"quantity({entity},{quantity}).")
        
        # Extract comparisons
        for token in doc:
            if token.pos_ == "ADJ" and token.tag_ in ["JJR", "JJS"]:  # Comparative or superlative
                # Find what's being compared
                entities = []
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        entities.append(child.text.lower())
                
                if entities:
                    comparison = token.lemma_.lower()
                    for entity in entities:
                        entity_clean = re.sub(r'\W+', '_', entity)
                        facts.append(f"{comparison}({entity_clean}).")
        
        return facts

    def _extract_qualitative(self, doc) -> List[str]:
        """Extract qualitative facts (properties, states, attributes)"""
        facts = []
        
        for token in doc:
            # Extract attribute patterns (X is Y)
            if token.lemma_ == "be" and token.dep_ == "ROOT":
                subjects = []
                attributes = []
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subjects.append(child.text.lower())
                    if child.dep_ in ["attr", "acomp"]:
                        attributes.append(child.text.lower())
                
                # Create facts connecting subjects and attributes
                for subj in subjects:
                    for attr in attributes:
                        subj_clean = re.sub(r'\W+', '_', subj)
                        attr_clean = re.sub(r'\W+', '_', attr)
                        facts.append(f"{attr_clean}({subj_clean}).")
            
            # Extract property patterns (adjectives modifying nouns)
            if token.pos_ in ["NOUN", "PROPN"]:
                for child in token.children:
                    if child.pos_ == "ADJ":
                        entity = re.sub(r'\W+', '_', token.text.lower())
                        property_text = re.sub(r'\W+', '_', child.text.lower())
                        facts.append(f"{property_text}({entity}).")
                        
        return facts

    def extract_all_facts(self, text: str) -> List[str]:
        """Extract all types of facts from text"""
        doc = self.nlp(text)
        doc = self._resolve_coreferences(doc)
        
        all_facts = []
        for fact_type, extractor in self.fact_types.items():
            facts = extractor(doc)
            all_facts.extend(facts)
            
        # Remove duplicates while preserving order
        unique_facts = []
        for fact in all_facts:
            if fact not in unique_facts:
                unique_facts.append(fact)
                
        return unique_facts

    def process_document(self, doc_path: str) -> List[str]:
        """Process a document and extract facts"""
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Process per paragraph to handle larger documents
        paragraphs = text.split('\n\n')
        all_facts = []
        
        for para in paragraphs:
            if para.strip():
                facts = self.extract_all_facts(para)
                all_facts.extend(facts)
                
        return all_facts

if __name__ == "__main__":
    # Example usage
    extractor = EnhancedFactExtractor()
    facts = extractor.process_document("input.txt")
    
    print("Extracted facts:")
    for fact in facts:
        print(fact)
        
    # Write to facts.lp
    with open("facts.lp", "w") as f:
        for fact in facts:
            f.write(fact + "\n") 