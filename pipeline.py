#!/usr/bin/env python3
import os
import time
import json
import logging
import threading
import queue
from typing import List, Dict, Tuple, Optional, Set, Any, Callable, Union
from dataclasses import dataclass, field, asdict
import traceback
import hashlib
import datetime
import sys

# Import our custom components
try:
    from enhanced_extractor import EnhancedFactExtractor
    from rule_manager import RuleManager, Rule
    from advanced_solver import AdvancedSolver, Explanation, Repair
    from knowledge_graph import KnowledgeGraph
except ImportError as e:
    # Print clear error message instead of silently failing
    print(f"ERROR: Missing required module: {e}")
    print("Make sure all required files are in the current directory:")
    print("  - enhanced_extractor.py")
    print("  - rule_manager.py")
    print("  - advanced_solver.py")
    print("  - knowledge_graph.py")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class PipelineTask:
    """Represents a task in the pipeline"""
    task_id: str
    task_type: str  # document, rule_generation, consistency_check, etc.
    status: str = "pending"  # pending, running, completed, failed
    priority: int = 1
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    input_data: Dict = field(default_factory=dict)
    output_data: Dict = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PipelineTask':
        return cls(**data)

class PipelineStage:
    """Base class for pipeline stages"""
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"Pipeline.{name}")
        
    def process(self, task: PipelineTask) -> PipelineTask:
        """Process a task and return the updated task"""
        raise NotImplementedError("Subclasses must implement process()")
    
    def can_process(self, task: PipelineTask) -> bool:
        """Check if this stage can process the given task"""
        raise NotImplementedError("Subclasses must implement can_process()")

class DocumentProcessingStage(PipelineStage):
    """Stage for processing documents and extracting facts"""
    def __init__(self, config: Dict = None):
        super().__init__("DocumentProcessing", config)
        # Initialize the fact extractor
        self.extractor = EnhancedFactExtractor(
            model_name=self.config.get("spacy_model", "en_core_web_trf"),
            use_gpu=self.config.get("use_gpu", True)
        )
        
    def can_process(self, task: PipelineTask) -> bool:
        return task.task_type == "document" and task.status == "pending"
    
    def process(self, task: PipelineTask) -> PipelineTask:
        try:
            # Mark task as running
            task.status = "running"
            task.started_at = datetime.datetime.now().isoformat()
            
            # Extract document path from task
            doc_path = task.input_data.get("document_path")
            if not doc_path or not os.path.exists(doc_path):
                raise ValueError(f"Invalid document path: {doc_path}")
            
            # Process document
            self.logger.info(f"Processing document: {doc_path}")
            facts = self.extractor.process_document(doc_path)
            
            # Store extracted facts in task output
            task.output_data["facts"] = facts
            task.output_data["document_path"] = doc_path
            task.output_data["document_id"] = hashlib.md5(doc_path.encode()).hexdigest()
            
            # Mark task as completed
            task.status = "completed"
            task.completed_at = datetime.datetime.now().isoformat()
            self.logger.info(f"Document processed successfully: {doc_path}")
            
        except Exception as e:
            # Handle errors
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.datetime.now().isoformat()
            self.logger.error(f"Error processing document: {e}")
            self.logger.error(traceback.format_exc())
        
        return task

class KnowledgeUpdateStage(PipelineStage):
    """Stage for updating the knowledge graph with extracted facts"""
    def __init__(self, config: Dict = None):
        super().__init__("KnowledgeUpdate", config)
        # Initialize the knowledge graph
        db_path = self.config.get("db_path", "knowledge.db")
        self.kg = KnowledgeGraph(db_path=db_path)
        
    def can_process(self, task: PipelineTask) -> bool:
        # Can process completed document tasks with extracted facts
        return (task.task_type == "document" and 
                task.status == "completed" and 
                "facts" in task.output_data)
    
    def process(self, task: PipelineTask) -> PipelineTask:
        try:
            # Mark task as running
            task.status = "running"
            task.started_at = datetime.datetime.now().isoformat()
            
            # Get facts from previous stage
            facts = task.output_data.get("facts", [])
            document_path = task.output_data.get("document_path")
            document_id = task.output_data.get("document_id")
            
            if not facts or not document_id:
                raise ValueError("No facts or document ID available")
            
            # Process each fact and update knowledge graph
            entity_ids = []
            relation_ids = []
            
            # First, extract and create entities
            self.logger.info(f"Updating knowledge graph with facts from document: {document_path}")
            for fact in facts:
                # Simple parsing of facts for entities
                # In a real system, this would use more sophisticated extraction
                entities, relations = self._parse_fact(fact)
                
                # Add entities to knowledge graph
                for entity_name, entity_type in entities:
                    entity_id = self.kg.add_entity(
                        name=entity_name,
                        entity_type=entity_type,
                        source="document",
                        confidence=0.9,
                        metadata={"document_id": document_id}
                    )
                    if entity_id and entity_id not in entity_ids:
                        entity_ids.append(entity_id)
                
                # Add relationships
                for rel_type, source_name, source_type, target_name, target_type in relations:
                    # Get or create entities
                    source_id = self.kg.add_entity(
                        name=source_name,
                        entity_type=source_type,
                        source="document",
                        confidence=0.9,
                        metadata={"document_id": document_id}
                    )
                    
                    target_id = self.kg.add_entity(
                        name=target_name,
                        entity_type=target_type,
                        source="document",
                        confidence=0.9,
                        metadata={"document_id": document_id}
                    )
                    
                    # Add relationship
                    rel_id = self.kg.add_relationship(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=rel_type,
                        confidence=0.9,
                        metadata={"document_id": document_id}
                    )
                    
                    if rel_id and rel_id not in relation_ids:
                        relation_ids.append(rel_id)
                
                # Store fact in knowledge graph
                self.kg.add_fact(
                    content=fact,
                    source_doc=document_id,
                    entity_ids=entity_ids,
                    relation_ids=relation_ids,
                    confidence=0.9
                )
            
            # Export knowledge graph facts to ASP file for consistency checking
            facts_file = f"facts_{document_id}.lp"
            num_exported = self.kg.export_to_asp(facts_file)
            
            # Update task output
            task.output_data["facts_file"] = facts_file
            task.output_data["num_entities"] = len(entity_ids)
            task.output_data["num_relations"] = len(relation_ids)
            task.output_data["num_exported"] = num_exported
            
            # Mark task as completed
            task.status = "completed"
            task.completed_at = datetime.datetime.now().isoformat()
            self.logger.info(f"Knowledge graph updated: {num_exported} facts exported")
            
        except Exception as e:
            # Handle errors
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.datetime.now().isoformat()
            self.logger.error(f"Error updating knowledge graph: {e}")
            self.logger.error(traceback.format_exc())
        
        return task
    
    def _parse_fact(self, fact: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str, str, str]]]:
        """Parse a fact to extract entities and relationships
        
        Returns:
            Tuple of (entities, relationships)
            - entities: List of (name, type) tuples
            - relationships: List of (rel_type, source_name, source_type, target_name, target_type) tuples
        """
        entities = []
        relationships = []
        
        # Simple parsing of ASP facts
        fact = fact.strip()
        if fact.endswith("."):
            fact = fact[:-1]  # Remove trailing dot
            
        if "(" not in fact:
            return entities, relationships
            
        predicate = fact[:fact.find("(")]
        args_str = fact[fact.find("(")+1:fact.rfind(")")]
        args = [arg.strip() for arg in args_str.split(",")]
        
        if len(args) == 1:
            # This is an entity fact: predicate(entity)
            entity_name = args[0]
            entity_type = predicate
            entities.append((entity_name, entity_type))
        elif len(args) == 2:
            # This is a relationship fact: predicate(entity1, entity2)
            source_name = args[0]
            target_name = args[1]
            
            # Infer entity types based on position and predicate
            # In a real system, this would be more sophisticated
            if predicate in ["married_to", "reports_to", "friends_with"]:
                source_type = "person"
                target_type = "person"
            elif predicate in ["works_for", "employed_by"]:
                source_type = "person"
                target_type = "organization"
            elif predicate in ["produces", "manufactures"]:
                source_type = "organization"
                target_type = "product"
            elif predicate in ["located_in", "based_in"]:
                source_type = "entity"
                target_type = "location"
            else:
                source_type = "entity"
                target_type = "entity"
            
            # Add entities
            entities.append((source_name, source_type))
            entities.append((target_name, target_type))
            
            # Add relationship
            relationships.append((predicate, source_name, source_type, target_name, target_type))
        
        return entities, relationships

class ConsistencyCheckStage(PipelineStage):
    """Stage for checking consistency of facts"""
    def __init__(self, config: Dict = None):
        super().__init__("ConsistencyCheck", config)
        # Initialize the solver
        rules_file = self.config.get("rules_file", "domain_rules.lp")
        repair_template = self.config.get("repair_template", "repair.lp")
        self.solver = AdvancedSolver(
            rules_file=rules_file,
            repair_template_file=repair_template,
            debug=self.config.get("debug", False)
        )
        
    def can_process(self, task: PipelineTask) -> bool:
        # Can process knowledge update tasks with facts file
        return (task.task_type in ["document", "fact_update"] and 
                task.status == "completed" and 
                "facts_file" in task.output_data)
    
    def process(self, task: PipelineTask) -> PipelineTask:
        try:
            # Mark task as running
            task.status = "running"
            task.started_at = datetime.datetime.now().isoformat()
            
            # Get facts file from previous stage
            facts_file = task.output_data.get("facts_file")
            if not facts_file or not os.path.exists(facts_file):
                raise ValueError(f"Invalid facts file: {facts_file}")
            
            # Read facts from file
            with open(facts_file, 'r') as f:
                facts = [line.strip() for line in f if line.strip() and not line.strip().startswith("%")]
            
            # Check consistency
            self.logger.info(f"Checking consistency of facts from: {facts_file}")
            result = self.solver.check_consistency(facts)
            
            # Store result in task output
            task.output_data["consistency_status"] = result["status"]
            if result["explanation"]:
                task.output_data["explanation"] = result["explanation"]
            
            # If inconsistent, generate repair suggestions
            if result["status"] == "INCONSISTENT":
                self.logger.info(f"Facts are inconsistent, generating repair suggestions")
                repairs = self.solver.suggest_repairs(facts)
                task.output_data["repairs"] = [repair.to_dict() for repair in repairs]
            
            # Mark task as completed
            task.status = "completed"
            task.completed_at = datetime.datetime.now().isoformat()
            self.logger.info(f"Consistency check completed: {result['status']}")
            
        except Exception as e:
            # Handle errors
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.datetime.now().isoformat()
            self.logger.error(f"Error checking consistency: {e}")
            self.logger.error(traceback.format_exc())
        
        return task

class RuleGenerationStage(PipelineStage):
    """Stage for generating rules from document content"""
    def __init__(self, config: Dict = None):
        super().__init__("RuleGeneration", config)
        # Initialize the rule manager
        rules_dir = self.config.get("rules_dir", "rules")
        openai_api_key = self.config.get("openai_api_key")
        llm_model = self.config.get("llm_model", "gpt-3.5-turbo")
        self.rule_manager = RuleManager(
            rules_dir=rules_dir,
            openai_api_key=openai_api_key,
            llm_model=llm_model
        )
        
    def can_process(self, task: PipelineTask) -> bool:
        # Can process rule generation tasks or completed document tasks
        return (task.task_type == "rule_generation" and task.status == "pending") or \
               (task.task_type == "document" and task.status == "completed" and 
                task.input_data.get("generate_rules", False))
    
    def process(self, task: PipelineTask) -> PipelineTask:
        try:
            # Mark task as running
            task.status = "running"
            task.started_at = datetime.datetime.now().isoformat()
            
            # Get document content for rule generation
            doc_path = task.input_data.get("document_path") or task.output_data.get("document_path")
            if not doc_path or not os.path.exists(doc_path):
                raise ValueError(f"Invalid document path: {doc_path}")
            
            # Read document content
            with open(doc_path, 'r') as f:
                doc_content = f.read()
            
            # Number of rules to generate
            num_rules = task.input_data.get("num_rules", 5)
            category = task.input_data.get("category", "auto_generated")
            
            # Generate rules using LLM
            self.logger.info(f"Generating rules from document: {doc_path}")
            new_rules = self.rule_manager.generate_rules_from_text(
                text=doc_content,
                num_rules=num_rules,
                category=category
            )
            
            # Export rules to file
            rules_file = f"rules_{task.task_id}.lp"
            num_exported = self.rule_manager.export_asp_rules(
                output_path=rules_file,
                categories=[category],
                min_confidence=0.7
            )
            
            # Update task output
            task.output_data["rules_file"] = rules_file
            task.output_data["num_rules"] = len(new_rules)
            task.output_data["rules"] = [rule.to_dict() for rule in new_rules]
            
            # Mark task as completed
            task.status = "completed"
            task.completed_at = datetime.datetime.now().isoformat()
            self.logger.info(f"Rule generation completed: {len(new_rules)} rules generated")
            
        except Exception as e:
            # Handle errors
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.datetime.now().isoformat()
            self.logger.error(f"Error generating rules: {e}")
            self.logger.error(traceback.format_exc())
        
        return task

class Pipeline:
    """Main pipeline for processing documents and checking consistency"""
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("Pipeline")
        
        # Task queues
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks: Dict[str, PipelineTask] = {}
        self.failed_tasks: Dict[str, PipelineTask] = {}
        
        # Initialize stages
        self.stages = [
            DocumentProcessingStage(self.config.get("document_processing", {})),
            KnowledgeUpdateStage(self.config.get("knowledge_update", {})),
            ConsistencyCheckStage(self.config.get("consistency_check", {})),
            RuleGenerationStage(self.config.get("rule_generation", {}))
        ]
        
        # Worker threads
        self.workers: List[threading.Thread] = []
        self.running = False
        self.task_lock = threading.Lock()
    
    def start(self, num_workers: int = 4):
        """Start the pipeline with specified number of workers"""
        self.running = True
        
        # Create and start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"Pipeline-Worker-{i}",
                daemon=True
            )
            self.workers.append(worker)
            worker.start()
            
        self.logger.info(f"Pipeline started with {num_workers} workers")
    
    def stop(self):
        """Stop the pipeline"""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            
        self.logger.info("Pipeline stopped")
    
    def _worker_loop(self):
        """Worker loop for processing tasks"""
        while self.running:
            try:
                # Get task from queue with timeout
                try:
                    priority, _, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Find a stage that can process this task
                processed = False
                for stage in self.stages:
                    if stage.can_process(task):
                        self.logger.info(f"Processing task {task.task_id} with stage {stage.name}")
                        # Process the task
                        updated_task = stage.process(task)
                        
                        # Update task status
                        with self.task_lock:
                            if updated_task.status == "completed":
                                self.completed_tasks[updated_task.task_id] = updated_task
                                
                                # Check if any further processing is needed
                                if self._needs_further_processing(updated_task):
                                    # Re-queue the task for further processing
                                    self.task_queue.put((updated_task.priority, time.time(), updated_task))
                                
                            elif updated_task.status == "failed":
                                self.failed_tasks[updated_task.task_id] = updated_task
                            else:
                                # Task still needs processing, re-queue it
                                self.task_queue.put((updated_task.priority, time.time(), updated_task))
                                
                        processed = True
                        break
                
                if not processed:
                    # No stage could process this task, mark as failed
                    task.status = "failed"
                    task.error = "No stage could process this task"
                    task.completed_at = datetime.datetime.now().isoformat()
                    with self.task_lock:
                        self.failed_tasks[task.task_id] = task
                    self.logger.warning(f"No stage could process task {task.task_id}")
                
                # Mark task as done in queue
                self.task_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
                self.logger.error(traceback.format_exc())
    
    def _needs_further_processing(self, task: PipelineTask) -> bool:
        """Check if a completed task needs further processing by another stage"""
        # Prevent infinite processing loops - track stages that already processed this task
        if 'processed_by' not in task.metadata:
            task.metadata['processed_by'] = []
        
        # Check for any stage that can process this task and hasn't processed it yet
        for stage in self.stages:
            if stage.name not in task.metadata['processed_by'] and stage.can_process(task):
                # Mark that this stage will process the task
                task.metadata['processed_by'].append(stage.name)
                return True
            
        return False
    
    def add_task(self, task: PipelineTask):
        """Add a task to the pipeline"""
        self.task_queue.put((task.priority, time.time(), task))
        self.logger.info(f"Added task {task.task_id} of type {task.task_type} to queue")
    
    def create_document_task(self, document_path: str, generate_rules: bool = False) -> str:
        """Create and add a document processing task"""
        task_id = hashlib.md5(f"{document_path}_{time.time()}".encode()).hexdigest()
        task = PipelineTask(
            task_id=task_id,
            task_type="document",
            priority=1,
            input_data={
                "document_path": document_path,
                "generate_rules": generate_rules
            }
        )
        self.add_task(task)
        return task_id
    
    def create_rule_generation_task(self, document_path: str, num_rules: int = 5, category: str = "auto_generated") -> str:
        """Create and add a rule generation task"""
        task_id = hashlib.md5(f"rule_gen_{document_path}_{time.time()}".encode()).hexdigest()
        task = PipelineTask(
            task_id=task_id,
            task_type="rule_generation",
            priority=2,
            input_data={
                "document_path": document_path,
                "num_rules": num_rules,
                "category": category
            }
        )
        self.add_task(task)
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get the status of a task"""
        with self.task_lock:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].to_dict()
            elif task_id in self.failed_tasks:
                return self.failed_tasks[task_id].to_dict()
        
        # Task might still be in queue or processing
        # Iterate through queue to find it (inefficient but okay for demo)
        with self.task_queue.mutex:
            for _, _, task in self.task_queue.queue:
                if task.task_id == task_id:
                    return task.to_dict()
        
        return None
    
    def wait_for_task(self, task_id: str, timeout: float = None) -> Optional[Dict]:
        """Wait for a task to complete and return its status"""
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            with self.task_lock:
                if task_id in self.completed_tasks:
                    return self.completed_tasks[task_id].to_dict()
                elif task_id in self.failed_tasks:
                    return self.failed_tasks[task_id].to_dict()
            
            # Sleep a bit before checking again
            time.sleep(0.5)
        
        return None

if __name__ == "__main__":
    # Example usage
    config = {
        "document_processing": {
            "spacy_model": "en_core_web_trf",
            "use_gpu": True
        },
        "knowledge_update": {
            "db_path": "knowledge.db"
        },
        "consistency_check": {
            "rules_file": "domain_rules.lp",
            "repair_template": "repair.lp",
            "debug": True
        },
        "rule_generation": {
            "rules_dir": "rules",
            "llm_model": "gpt-3.5-turbo"
        }
    }
    
    # Create and start pipeline
    pipeline = Pipeline(config)
    pipeline.start(num_workers=2)
    
    try:
        # Process a document
        doc_path = "input.txt"
        task_id = pipeline.create_document_task(doc_path, generate_rules=True)
        
        # Wait for processing to complete
        result = pipeline.wait_for_task(task_id, timeout=120)
        
        if result:
            print(f"Task completed with status: {result['status']}")
            if result['status'] == 'completed':
                # Check consistency results
                if 'consistency_status' in result['output_data']:
                    consistency = result['output_data']['consistency_status']
                    print(f"Consistency status: {consistency}")
                    
                    if consistency == "INCONSISTENT" and 'explanation' in result['output_data']:
                        print("Explanation:")
                        print(result['output_data']['explanation'])
                        
                        if 'repairs' in result['output_data']:
                            print("\nRepair suggestions:")
                            for i, repair in enumerate(result['output_data']['repairs']):
                                print(f"Repair {i+1}: {repair['explanation']}")
        else:
            print("Task did not complete within timeout")
            
    finally:
        # Stop pipeline
        pipeline.stop() 