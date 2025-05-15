#!/usr/bin/env python3
import os
import json
import sqlite3
import datetime
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Set, Any, Union
import hashlib

class KnowledgeGraph:
    def __init__(self, db_path: str = "knowledge.db"):
        self.db_path = db_path
        self.graph = nx.DiGraph()
        self.entity_types = set()
        self.relation_types = set()
        
        # Initialize database
        self._init_db()
        
        # Load existing data
        self._load_data()
    
    def _init_db(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create tables if they don't exist
        c.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT,
            source TEXT,
            confidence REAL,
            created_at TEXT,
            metadata TEXT
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            source_id TEXT,
            target_id TEXT,
            type TEXT,
            attributes TEXT,
            confidence REAL,
            created_at TEXT,
            metadata TEXT,
            FOREIGN KEY (source_id) REFERENCES entities (id),
            FOREIGN KEY (target_id) REFERENCES entities (id)
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS rules (
            id TEXT PRIMARY KEY,
            content TEXT,
            description TEXT,
            category TEXT,
            source TEXT,
            confidence REAL,
            version TEXT,
            created_at TEXT,
            metadata TEXT
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS facts (
            id TEXT PRIMARY KEY,
            content TEXT,
            source_doc TEXT,
            source_text TEXT,
            entity_ids TEXT,
            relation_ids TEXT,
            confidence REAL,
            created_at TEXT,
            metadata TEXT
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS document_facts (
            document_id TEXT,
            fact_id TEXT,
            PRIMARY KEY (document_id, fact_id)
        )
        ''')
        
        # Create indices for faster queries
        c.execute('CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (type)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships (type)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships (source_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships (target_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_rules_category ON rules (category)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_doc_facts_doc ON document_facts (document_id)')
        
        conn.commit()
        conn.close()
    
    def _load_data(self):
        """Load existing data from database into memory graph"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Load entities
        c.execute('SELECT * FROM entities')
        for row in c.fetchall():
            entity_data = dict(row)
            entity_id = entity_data['id']
            
            # Parse metadata
            if entity_data['metadata']:
                try:
                    metadata = json.loads(entity_data['metadata'])
                except json.JSONDecodeError:
                    metadata = {}
            else:
                metadata = {}
            
            # Add node to graph
            self.graph.add_node(
                entity_id,
                name=entity_data['name'],
                type=entity_data['type'],
                source=entity_data['source'],
                confidence=entity_data['confidence'],
                created_at=entity_data['created_at'],
                metadata=metadata,
                node_type='entity'
            )
            
            # Track entity type
            self.entity_types.add(entity_data['type'])
        
        # Load relationships
        c.execute('SELECT * FROM relationships')
        for row in c.fetchall():
            rel_data = dict(row)
            rel_id = rel_data['id']
            source_id = rel_data['source_id']
            target_id = rel_data['target_id']
            
            # Skip if source or target doesn't exist
            if source_id not in self.graph or target_id not in self.graph:
                continue
            
            # Parse attributes and metadata
            if rel_data['attributes']:
                try:
                    attributes = json.loads(rel_data['attributes'])
                except json.JSONDecodeError:
                    attributes = {}
            else:
                attributes = {}
            
            if rel_data['metadata']:
                try:
                    metadata = json.loads(rel_data['metadata'])
                except json.JSONDecodeError:
                    metadata = {}
            else:
                metadata = {}
            
            # Add edge to graph
            self.graph.add_edge(
                source_id,
                target_id,
                id=rel_id,
                type=rel_data['type'],
                attributes=attributes,
                confidence=rel_data['confidence'],
                created_at=rel_data['created_at'],
                metadata=metadata
            )
            
            # Track relation type
            self.relation_types.add(rel_data['type'])
        
        conn.close()
    
    def _generate_id(self, data: Any) -> str:
        """Generate a unique ID based on the content"""
        content_str = str(data) + str(datetime.datetime.now())
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def add_entity(self, 
                 name: str, 
                 entity_type: str, 
                 source: str = "manual",
                 confidence: float = 1.0,
                 metadata: Dict = None) -> str:
        """Add an entity to the knowledge graph"""
        # Generate ID
        entity_id = self._generate_id(f"{name}_{entity_type}")
        
        # Check if entity already exists
        if self._entity_exists(name, entity_type):
            return self._get_entity_id(name, entity_type)
        
        # Current timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            'INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?)',
            (
                entity_id,
                name,
                entity_type,
                source,
                confidence,
                timestamp,
                json.dumps(metadata or {})
            )
        )
        conn.commit()
        conn.close()
        
        # Add to in-memory graph
        self.graph.add_node(
            entity_id,
            name=name,
            type=entity_type,
            source=source,
            confidence=confidence,
            created_at=timestamp,
            metadata=metadata or {},
            node_type='entity'
        )
        
        # Track entity type
        self.entity_types.add(entity_type)
        
        return entity_id
    
    def _entity_exists(self, name: str, entity_type: str) -> bool:
        """Check if an entity already exists"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            'SELECT id FROM entities WHERE name = ? AND type = ?',
            (name, entity_type)
        )
        result = c.fetchone()
        conn.close()
        
        return result is not None
    
    def _get_entity_id(self, name: str, entity_type: str) -> Optional[str]:
        """Get the ID of an existing entity"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            'SELECT id FROM entities WHERE name = ? AND type = ?',
            (name, entity_type)
        )
        result = c.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def add_relationship(self, 
                       source_id: str,
                       target_id: str,
                       relation_type: str,
                       attributes: Dict = None,
                       confidence: float = 1.0,
                       metadata: Dict = None) -> Optional[str]:
        """Add a relationship between entities"""
        # Check if source and target exist
        if source_id not in self.graph or target_id not in self.graph:
            return None
        
        # Generate ID
        rel_id = self._generate_id(f"{source_id}_{relation_type}_{target_id}")
        
        # Current timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            'INSERT INTO relationships VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (
                rel_id,
                source_id,
                target_id,
                relation_type,
                json.dumps(attributes or {}),
                confidence,
                timestamp,
                json.dumps(metadata or {})
            )
        )
        conn.commit()
        conn.close()
        
        # Add to in-memory graph
        self.graph.add_edge(
            source_id,
            target_id,
            id=rel_id,
            type=relation_type,
            attributes=attributes or {},
            confidence=confidence,
            created_at=timestamp,
            metadata=metadata or {}
        )
        
        # Track relation type
        self.relation_types.add(relation_type)
        
        return rel_id
    
    def add_fact(self, 
               content: str,
               source_doc: str = "",
               source_text: str = "",
               entity_ids: List[str] = None,
               relation_ids: List[str] = None,
               confidence: float = 1.0,
               metadata: Dict = None) -> str:
        """Add a fact to the knowledge base"""
        # Generate ID
        fact_id = self._generate_id(content)
        
        # Current timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            'INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                fact_id,
                content,
                source_doc,
                source_text,
                json.dumps(entity_ids or []),
                json.dumps(relation_ids or []),
                confidence,
                timestamp,
                json.dumps(metadata or {})
            )
        )
        
        # Add document-fact association if source_doc is provided
        if source_doc:
            c.execute(
                'INSERT OR IGNORE INTO document_facts VALUES (?, ?)',
                (source_doc, fact_id)
            )
        
        conn.commit()
        conn.close()
        
        return fact_id
    
    def add_rule(self, 
               content: str,
               description: str = "",
               category: str = "general",
               source: str = "manual",
               confidence: float = 1.0,
               version: str = "1.0",
               metadata: Dict = None) -> str:
        """Add a rule to the knowledge base"""
        # Generate ID
        rule_id = self._generate_id(content)
        
        # Current timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            'INSERT INTO rules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                rule_id,
                content,
                description,
                category,
                source,
                confidence,
                version,
                timestamp,
                json.dumps(metadata or {})
            )
        )
        conn.commit()
        conn.close()
        
        return rule_id
    
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Get entity by ID"""
        if entity_id in self.graph and self.graph.nodes[entity_id].get('node_type') == 'entity':
            return dict(self.graph.nodes[entity_id])
        return None
    
    def get_entity_by_name_type(self, name: str, entity_type: str) -> Optional[Dict]:
        """Get entity by name and type"""
        entity_id = self._get_entity_id(name, entity_type)
        if entity_id:
            return self.get_entity(entity_id)
        return None
    
    def get_relationship(self, relationship_id: str) -> Optional[Dict]:
        """Get relationship by ID"""
        for source, target, data in self.graph.edges(data=True):
            if data.get('id') == relationship_id:
                return {**data, 'source_id': source, 'target_id': target}
        return None
    
    def get_relationships_between(self, source_id: str, target_id: str) -> List[Dict]:
        """Get relationships between two entities"""
        relationships = []
        if self.graph.has_edge(source_id, target_id):
            for key, value in self.graph[source_id][target_id].items():
                relationships.append({
                    **value,
                    'source_id': source_id,
                    'target_id': target_id
                })
        return relationships
    
    def get_neighbors(self, entity_id: str, relationship_types: List[str] = None) -> Dict[str, List[Dict]]:
        """Get neighboring entities and their relationships"""
        if entity_id not in self.graph:
            return {'incoming': [], 'outgoing': []}
        
        neighbors = {
            'incoming': [],
            'outgoing': []
        }
        
        # Get outgoing relationships
        for target in self.graph.successors(entity_id):
            for rel_data in self.graph[entity_id][target].values():
                if relationship_types is None or rel_data.get('type') in relationship_types:
                    neighbors['outgoing'].append({
                        'entity': self.get_entity(target),
                        'relationship': {**rel_data, 'source_id': entity_id, 'target_id': target}
                    })
        
        # Get incoming relationships
        for source in self.graph.predecessors(entity_id):
            for rel_data in self.graph[source][entity_id].values():
                if relationship_types is None or rel_data.get('type') in relationship_types:
                    neighbors['incoming'].append({
                        'entity': self.get_entity(source),
                        'relationship': {**rel_data, 'source_id': source, 'target_id': entity_id}
                    })
        
        return neighbors
    
    def get_facts_by_document(self, document_id: str) -> List[Dict]:
        """Get all facts associated with a document"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('''
        SELECT f.* FROM facts f
        JOIN document_facts df ON f.id = df.fact_id
        WHERE df.document_id = ?
        ''', (document_id,))
        
        facts = []
        for row in c.fetchall():
            fact_data = dict(row)
            
            # Parse JSON fields
            if fact_data['entity_ids']:
                fact_data['entity_ids'] = json.loads(fact_data['entity_ids'])
            else:
                fact_data['entity_ids'] = []
                
            if fact_data['relation_ids']:
                fact_data['relation_ids'] = json.loads(fact_data['relation_ids'])
            else:
                fact_data['relation_ids'] = []
                
            if fact_data['metadata']:
                fact_data['metadata'] = json.loads(fact_data['metadata'])
            else:
                fact_data['metadata'] = {}
            
            facts.append(fact_data)
        
        conn.close()
        return facts
    
    def get_rules_by_category(self, category: str) -> List[Dict]:
        """Get rules by category"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('SELECT * FROM rules WHERE category = ?', (category,))
        
        rules = []
        for row in c.fetchall():
            rule_data = dict(row)
            
            # Parse metadata
            if rule_data['metadata']:
                rule_data['metadata'] = json.loads(rule_data['metadata'])
            else:
                rule_data['metadata'] = {}
            
            rules.append(rule_data)
        
        conn.close()
        return rules
    
    def search_entities(self, 
                      query: str = None, 
                      entity_type: str = None,
                      min_confidence: float = 0.0) -> List[Dict]:
        """Search for entities"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        sql = 'SELECT * FROM entities WHERE confidence >= ?'
        params = [min_confidence]
        
        if query:
            sql += ' AND name LIKE ?'
            params.append(f'%{query}%')
        
        if entity_type:
            sql += ' AND type = ?'
            params.append(entity_type)
        
        c.execute(sql, params)
        
        entities = []
        for row in c.fetchall():
            entity_data = dict(row)
            
            # Parse metadata
            if entity_data['metadata']:
                entity_data['metadata'] = json.loads(entity_data['metadata'])
            else:
                entity_data['metadata'] = {}
            
            entities.append(entity_data)
        
        conn.close()
        return entities
    
    def search_facts(self, 
                   content_query: str = None,
                   source_doc: str = None,
                   min_confidence: float = 0.0) -> List[Dict]:
        """Search for facts"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        sql = 'SELECT * FROM facts WHERE confidence >= ?'
        params = [min_confidence]
        
        if content_query:
            sql += ' AND content LIKE ?'
            params.append(f'%{content_query}%')
        
        if source_doc:
            sql += ' AND source_doc = ?'
            params.append(source_doc)
        
        c.execute(sql, params)
        
        facts = []
        for row in c.fetchall():
            fact_data = dict(row)
            
            # Parse JSON fields
            if fact_data['entity_ids']:
                fact_data['entity_ids'] = json.loads(fact_data['entity_ids'])
            else:
                fact_data['entity_ids'] = []
                
            if fact_data['relation_ids']:
                fact_data['relation_ids'] = json.loads(fact_data['relation_ids'])
            else:
                fact_data['relation_ids'] = []
                
            if fact_data['metadata']:
                fact_data['metadata'] = json.loads(fact_data['metadata'])
            else:
                fact_data['metadata'] = {}
            
            facts.append(fact_data)
        
        conn.close()
        return facts
    
    def export_to_asp(self, output_file: str, 
                    entity_types: List[str] = None,
                    relation_types: List[str] = None,
                    min_confidence: float = 0.5) -> int:
        """Export knowledge graph to ASP facts file"""
        # Filter entities and relationships based on types and confidence
        entities_to_export = []
        relationships_to_export = []
        
        # Filter entities
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') != 'entity':
                continue
                
            if data.get('confidence', 0) < min_confidence:
                continue
                
            if entity_types and data.get('type') not in entity_types:
                continue
                
            entities_to_export.append((node, data))
        
        # Filter relationships
        for source, target, data in self.graph.edges(data=True):
            if data.get('confidence', 0) < min_confidence:
                continue
                
            if relation_types and data.get('type') not in relation_types:
                continue
                
            relationships_to_export.append((source, target, data))
        
        # Write to ASP file
        with open(output_file, 'w') as f:
            f.write("% Auto-generated ASP facts from Knowledge Graph\n")
            f.write(f"% Generated on: {datetime.datetime.now().isoformat()}\n")
            f.write(f"% Confidence threshold: {min_confidence}\n\n")
            
            # Write entities
            f.write("% === ENTITIES ===\n")
            for node, data in entities_to_export:
                entity_type = data.get('type', 'entity')
                entity_name = data.get('name', node)
                # Sanitize name for ASP
                entity_name = entity_name.lower().replace(' ', '_').replace('-', '_')
                f.write(f"{entity_type}({entity_name}).\n")
            
            # Write relationships
            f.write("\n% === RELATIONSHIPS ===\n")
            for source, target, data in relationships_to_export:
                relation_type = data.get('type', 'related_to')
                
                # Get source and target names
                source_data = self.graph.nodes[source]
                target_data = self.graph.nodes[target]
                
                source_name = source_data.get('name', source)
                target_name = target_data.get('name', target)
                
                # Sanitize names for ASP
                source_name = source_name.lower().replace(' ', '_').replace('-', '_')
                target_name = target_name.lower().replace(' ', '_').replace('-', '_')
                
                f.write(f"{relation_type}({source_name},{target_name}).\n")
        
        return len(entities_to_export) + len(relationships_to_export)
    
    def visualize(self, output_file: str = "knowledge_graph.png", 
                max_nodes: int = 100,
                entity_types: List[str] = None,
                relation_types: List[str] = None,
                min_confidence: float = 0.5):
        """Visualize knowledge graph"""
        # Create a subgraph for visualization
        if len(self.graph) > max_nodes:
            # Filter nodes based on types and confidence
            nodes_to_include = []
            
            for node, data in self.graph.nodes(data=True):
                if data.get('node_type') != 'entity':
                    continue
                    
                if data.get('confidence', 0) < min_confidence:
                    continue
                    
                if entity_types and data.get('type') not in entity_types:
                    continue
                    
                nodes_to_include.append(node)
            
            # Limit to max_nodes
            if len(nodes_to_include) > max_nodes:
                nodes_to_include = nodes_to_include[:max_nodes]
                
            # Create subgraph
            subgraph = self.graph.subgraph(nodes_to_include)
        else:
            subgraph = self.graph
        
        # Set up visualization
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
        
        # Color nodes by entity type
        entity_types = set(nx.get_node_attributes(subgraph, 'type').values())
        colors = plt.cm.tab20(np.linspace(0, 1, len(entity_types)))
        color_map = {et: colors[i] for i, et in enumerate(entity_types)}
        
        # Node labels
        node_labels = {node: data.get('name', node) for node, data in subgraph.nodes(data=True)}
        
        # Draw nodes by entity type
        for et in entity_types:
            nodes = [n for n, d in subgraph.nodes(data=True) if d.get('type') == et]
            nx.draw_networkx_nodes(subgraph, pos, 
                                  nodelist=nodes,
                                  node_color=[color_map[et]],
                                  node_size=1000,
                                  alpha=0.7,
                                  label=et)
        
        # Edge labels
        edge_labels = {(u, v): d.get('type', 'related_to') 
                      for u, v, d in subgraph.edges(data=True)}
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, width=1.5, alpha=0.7, arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=10, font_weight='bold')
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)
        
        # Add legend
        plt.legend(title="Entity Types")
        
        # Save figure
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return len(subgraph.nodes), len(subgraph.edges)

if __name__ == "__main__":
    import numpy as np  # For visualization
    
    # Example usage
    kg = KnowledgeGraph()
    
    # Add some entities
    john_id = kg.add_entity("John", "person", confidence=0.9)
    mary_id = kg.add_entity("Mary", "person", confidence=0.9)
    bob_id = kg.add_entity("Bob", "person", confidence=0.8)
    
    company_id = kg.add_entity("Acme Inc", "organization", confidence=0.9)
    product_id = kg.add_entity("Widget", "product", confidence=0.8)
    
    # Add relationships
    kg.add_relationship(john_id, mary_id, "married_to", confidence=0.9)
    kg.add_relationship(john_id, company_id, "works_for", confidence=0.8)
    kg.add_relationship(mary_id, company_id, "works_for", confidence=0.8)
    kg.add_relationship(bob_id, john_id, "reports_to", confidence=0.7)
    kg.add_relationship(company_id, product_id, "produces", confidence=0.9)
    
    # Add some facts
    kg.add_fact("John is married to Mary.", entity_ids=[john_id, mary_id], confidence=0.9)
    kg.add_fact("John works for Acme Inc.", entity_ids=[john_id, company_id], confidence=0.8)
    kg.add_fact("Acme Inc produces widgets.", entity_ids=[company_id, product_id], confidence=0.9)
    
    # Add a rule
    kg.add_rule(":- married_to(X,Y), single(X).",
               description="A person cannot be both married and single",
               category="relationships",
               confidence=1.0)
    
    # Export to ASP
    num_exported = kg.export_to_asp("knowledge_facts.lp")
    print(f"Exported {num_exported} facts and relationships to knowledge_facts.lp")
    
    # Visualize
    nodes, edges = kg.visualize()
    print(f"Created visualization with {nodes} nodes and {edges} edges") 