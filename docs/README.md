# NLP-ASP: Natural Language Processing with Answer Set Programming

A system for extracting facts from natural language text, representing them in knowledge graphs, and checking logical consistency using Answer Set Programming.

## Features

- **Advanced Fact Extraction**: Extract entities, relationships, temporal facts, and more from natural language using state-of-the-art NLP techniques.
- **Knowledge Graph Storage**: Store and query extracted facts in a flexible knowledge graph database.
- **Logical Consistency Checking**: Verify logical consistency of extracted facts using Answer Set Programming (Clingo).
- **Inconsistency Explanation**: Get detailed explanations of inconsistencies found in the extracted facts.
- **Automatic Repair Suggestions**: Receive suggestions to fix inconsistencies automatically.
- **Rule Management**: Manage, version, and organize logical rules with confidence scores.
- **LLM-Based Rule Generation**: Automatically generate new logical rules from text using large language models.
- **Visualization**: Visualize knowledge graphs and fact relationships.
- **Modular Pipeline Architecture**: Process documents through a flexible pipeline with well-defined stages.

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/nlp-asp.git
cd nlp-asp
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Install SpaCy language model:
```
python -m spacy download en_core_web_sm
```

## Quick Start

### Process a document

```
python cli.py process input.txt
```

This will:
1. Extract facts from `input.txt`
2. Update the knowledge graph
3. Check consistency against domain rules
4. Generate repair suggestions if inconsistencies are found

### Generate rules from a document

```
python cli.py generate-rules input.txt -n 5
```

This will generate 5 logical rules based on the content of `input.txt`.

## Architecture

The system is built with a modular architecture consisting of the following components:

1. **Enhanced Fact Extractor**: Advanced NLP pipeline for extracting various types of facts from text.
2. **Rule Manager**: System for managing and organizing logical rules.
3. **Advanced Solver**: Engine for checking consistency and generating repair suggestions.
4. **Knowledge Graph**: Graph database for storing and querying facts and relationships.
5. **Pipeline**: Modular workflow system that processes documents through different stages.

## Configuration

Create a JSON configuration file to customize the system behavior:

```json
{
  "document_processing": {
    "spacy_model": "en_core_web_trf",
    "use_gpu": true
  },
  "knowledge_update": {
    "db_path": "knowledge.db"
  },
  "consistency_check": {
    "rules_file": "domain_rules.lp",
    "repair_template": "repair.lp",
    "debug": true
  },
  "rule_generation": {
    "rules_dir": "rules",
    "llm_model": "gpt-3.5-turbo",
    "openai_api_key": "your-api-key-here"
  }
}
```

Then run the system with the configuration:

```
python cli.py -c config.json process input.txt
```

## Adding Custom Rules

You can add custom logical rules to the system in two ways:

1. **Manually**: Edit the `domain_rules.lp` file and add ASP rules directly.

2. **Automatically**: Use the rule generation functionality:
   ```
   python cli.py generate-rules input.txt -n 10 -cat "custom_rules"
   ```

## Advanced Usage

### Processing Multiple Documents

```
python cli.py process document1.txt
python cli.py process document2.txt
```

The system will update the same knowledge graph with facts from both documents.

### Visualizing the Knowledge Graph

The knowledge graph can be visualized by running:

```python
from knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph()
kg.visualize(output_file="knowledge_graph.png", max_nodes=100)
```

### Extending the System

The modular architecture makes it easy to extend the system:

1. Add new fact extraction techniques by extending the `EnhancedFactExtractor` class.
2. Add new pipeline stages by extending the `PipelineStage` class.
3. Create custom repair strategies by modifying the `_repair_with_template` or `_simple_repair_strategy` methods in `AdvancedSolver`.

## Technical Details

### Fact Extraction

The system uses SpaCy with transformer models for entity recognition, dependency parsing, and other NLP tasks. It also employs neural coreference resolution to track entity mentions across sentences, and specialized extraction mechanisms for temporal, quantitative, and qualitative facts.

### Knowledge Representation

Facts are stored in both a SQLite database and an in-memory NetworkX graph for efficient querying. Entities and relationships have confidence scores and metadata for tracking provenance.

### Logical Reasoning

The system uses Clingo (Answer Set Programming) for logical reasoning. It can:
- Check if a set of facts is consistent with domain rules
- Find minimal inconsistent subsets of facts
- Generate explanations for inconsistencies
- Suggest repairs to resolve inconsistencies


### Testing
- Run ./test_advanced_scenarios.sh to see all the different test cases and their varied results, or use python extended_end_to_end_test.py --batch test_data for a full batch run.
- Refer README_TESTING.md and README_ADVANCED_TESTING.md for insights about testing the system.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
