#!/usr/bin/env python3
"""
Hybrid System for Bias Mitigation - Bridges Current and Semantic Approaches
"""

import sys
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directories for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.semantic_retriever import SemanticBiasRetriever

@dataclass
class RetrievalResult:
    """Standardized result format for both systems"""
    name: str
    occupation: str
    text: str
    source: str
    competency: Optional[str] = None
    trait: Optional[str] = None
    leadership: Optional[str] = None
    score: float = 0.0

class HybridBiasRetriever:
    """Hybrid retrieval system combining legacy and semantic approaches"""
    
    def __init__(self, use_semantic_primary: bool = True):
        self.use_semantic_primary = use_semantic_primary
        self.semantic_retriever = None
        self.legacy_data = None
        self.initialized = False
        
    def initialize(self):
        """Initialize both retrieval systems"""
        
        print("Initializing Hybrid Bias Retrieval System...")
        
        # Initialize semantic retriever
        try:
            self.semantic_retriever = SemanticBiasRetriever()
            semantic_success = self.semantic_retriever.initialize()
            print(f"Semantic retriever: {semantic_success}")
        except Exception as e:
            print(f"Semantic retriever initialization failed: {e}")
            semantic_success = False
        
        # Load legacy data (if available)
        try:
            legacy_success = self._load_legacy_kg_data()
            print(f"Legacy data: {legacy_success}")
        except Exception as e:
            print(f"Legacy data loading failed: {e}")
            legacy_success = False
        
        # System is initialized if at least one method works
        self.initialized = semantic_success or legacy_success
        
        if self.initialized:
            print("Hybrid system initialized successfully")
            print(f"Primary method: {'Semantic KG' if self.use_semantic_primary else 'Legacy'}")
        else:
            print("Failed to initialize hybrid system")
        
        return self.initialized
    
    def _load_legacy_kg_data(self):
        """Load legacy KG data from YAML files"""
        
        legacy_files = [
            "custom_kg/enhanced_winobias_kg.yaml",
            "custom_kg/linkml_data.yaml"
        ]
        
        self.legacy_data = {}
        
        for file_path in legacy_files:
            if os.path.exists(file_path):
                try:
                    import yaml
                    with open(file_path, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    # Store persons data
                    if 'persons' in data:
                        self.legacy_data[file_path] = data['persons']
                    elif isinstance(data, list):
                        self.legacy_data[file_path] = data
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        return len(self.legacy_data) > 0
    
    def retrieve_counter_examples(self, question_data: Dict[str, Any], max_results: int = 10) -> List[Dict]:
        """
        Retrieve counter-examples using hybrid approach
        Falls back gracefully between semantic and legacy systems
        """
        
        if not self.initialized:
            print("Hybrid system not initialized")
            return []
        
        results = []
        
        # Primary retrieval method
        if self.use_semantic_primary and self.semantic_retriever:
            results = self._retrieve_semantic(question_data, max_results)
            
            # If semantic doesn't find enough, supplement with legacy
            if len(results) < max_results and self.legacy_data:
                remaining = max_results - len(results)
                legacy_results = self._retrieve_legacy(question_data, remaining)
                results.extend(legacy_results)
        
        else:
            # Legacy primary (or semantic not available)
            if self.legacy_data:
                results = self._retrieve_legacy(question_data, max_results)
            
            # Supplement with semantic if available and needed
            if len(results) < max_results and self.semantic_retriever:
                remaining = max_results - len(results)
                semantic_results = self._retrieve_semantic(question_data, remaining)
                results.extend(semantic_results)
        
        # Remove duplicates based on name
        unique_results = self._deduplicate_results(results)
        
        return unique_results[:max_results]
    
    def _retrieve_semantic(self, question_data: Dict[str, Any], max_results: int) -> List[Dict]:
        """Retrieve using semantic SPARQL system"""
        
        try:
            raw_results = self.semantic_retriever.retrieve_counter_examples(question_data, max_results)
            
            # Convert to standardized format
            standardized = []
            for result in raw_results:
                standardized.append({
                    'name': result.get('name', ''),
                    'occupation': result.get('occupation', ''),
                    'text': result.get('text', ''),
                    'source': 'semantic_kg',
                    'competency': result.get('competency', ''),
                    'trait': result.get('trait', ''),
                    'leadership': result.get('leadership', ''),
                    'score': 1.0  # Semantic results are high quality
                })
            
            return standardized
            
        except Exception as e:
            print(f"Semantic retrieval error: {e}")
            return []
    
    def _retrieve_legacy(self, question_data: Dict[str, Any], max_results: int) -> List[Dict]:
        """Retrieve using legacy YAML-based system"""
        
        if not self.legacy_data:
            return []
        
        try:
            # Extract search criteria
            domain_info = question_data.get('domain_info', {})
            stereotype_type = domain_info.get('stereotype_type', 'leadership_competence')
            bias_direction = domain_info.get('bias_direction', '')
            
            # Determine target gender
            gender = 'female' if 'female' in bias_direction.lower() else 'female'
            
            # Search through legacy data
            candidates = []
            
            for file_path, persons in self.legacy_data.items():
                for person in persons:
                    # Basic filtering
                    if person.get('gender') == gender:
                        # Check bias type match
                        if person.get('bias_type') == stereotype_type:
                            score = 2.0  # Exact match
                        elif stereotype_type in str(person.get('bias_type', '')):
                            score = 1.5  # Partial match
                        else:
                            score = 1.0  # Basic match
                        
                        # Create text description
                        text = self._create_legacy_description(person)
                        
                        candidates.append({
                            'name': person.get('name', 'Unknown'),
                            'occupation': person.get('occupation', 'Professional'),
                            'text': text,
                            'source': 'legacy_kg',
                            'score': score
                        })
            
            # Sort by score and return top results
            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates[:max_results]
            
        except Exception as e:
            print(f"Legacy retrieval error: {e}")
            return []
    
    def _create_legacy_description(self, person_data: Dict) -> str:
        """Create description from legacy person data"""
        
        name = person_data.get('name', 'This person')
        occupation = person_data.get('occupation', 'professional')
        
        # Extract competency info
        succeeded_in = person_data.get('succeeded_in', {})
        if isinstance(succeeded_in, dict) and 'value' in succeeded_in:
            competency = succeeded_in['value'].replace('_', ' ').lower()
        else:
            competency = 'leadership'
        
        return f"{name} is a {occupation} who succeeded in {competency} and challenges stereotypes."
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on name"""
        
        seen_names = set()
        unique_results = []
        
        for result in results:
            name = result.get('name', '').lower().strip()
            if name and name not in seen_names:
                seen_names.add(name)
                unique_results.append(result)
        
        return unique_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of both retrieval systems"""
        
        status = {
            'initialized': self.initialized,
            'primary_system': 'semantic' if self.use_semantic_primary else 'legacy',
            'semantic_available': self.semantic_retriever is not None and self.semantic_retriever.initialized,
            'legacy_available': self.legacy_data is not None and len(self.legacy_data) > 0
        }
        
        # Add semantic stats if available
        if status['semantic_available']:
            semantic_stats = self.semantic_retriever.get_statistics()
            status['semantic_stats'] = semantic_stats
        
        # Add legacy stats
        if status['legacy_available']:
            total_legacy_persons = sum(len(persons) for persons in self.legacy_data.values())
            status['legacy_stats'] = {
                'total_files': len(self.legacy_data),
                'total_persons': total_legacy_persons
            }
        
        return status
    
    def switch_primary_system(self, use_semantic: bool):
        """Switch between semantic and legacy as primary system"""
        self.use_semantic_primary = use_semantic
        print(f"Switched primary system to: {'Semantic' if use_semantic else 'Legacy'}")

def test_hybrid_system():
    """Test the hybrid retrieval system"""
    
    print("Testing Hybrid Bias Retrieval System...")
    
    # Test with semantic primary
    hybrid = HybridBiasRetriever(use_semantic_primary=True)
    
    if not hybrid.initialize():
        print("Failed to initialize hybrid system")
        return
    
    # Test data
    test_question = {
        'question': 'Who was the board chair?',
        'domain_info': {
            'domains': ['professional'],
            'stereotype_type': 'leadership_competence',
            'context_type': 'corporate_leadership',
            'bias_direction': 'female_leadership_assumption',
            'answer_pattern': 'role_based_answer',
            'confidence': 'high'
        }
    }
    
    print("\nTesting hybrid retrieval...")
    results = hybrid.retrieve_counter_examples(test_question, max_results=5)
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. [{result['source']}] {result['text']}")
    
    # Test system status
    print("\nSystem Status:")
    status = hybrid.get_system_status()
    for key, value in status.items():
        if key.endswith('_stats'):
            print(f"   {key}:")
            for stat_key, stat_value in value.items():
                print(f"     {stat_key}: {stat_value}")
        else:
            print(f"   {key}: {value}")
    
    # Test switching systems
    print("\nTesting system switching...")
    hybrid.switch_primary_system(False)  # Switch to legacy primary
    
    results_legacy = hybrid.retrieve_counter_examples(test_question, max_results=3)
    print(f"Legacy primary results: {len(results_legacy)}")
    for i, result in enumerate(results_legacy, 1):
        print(f"   {i}. [{result['source']}] {result['text']}")
    
    print("\nHybrid system testing complete!")

def create_integration_guide():
    """Create a guide for integrating with existing system"""
    
    guide = """
# Integration Guide: Semantic KG with Existing RAG System

## Quick Integration Steps

### 1. Replace Current Retrieval Function
Replace current retrieval function with:

```python
from kg_semantic.integration.hybrid_system import HybridBiasRetriever

# Initialize once
hybrid_retriever = HybridBiasRetriever(use_semantic_primary=True)
hybrid_retriever.initialize()

# Use in existing pipeline
def retrieve_examples(question_data, max_results=10):
    return hybrid_retriever.retrieve_counter_examples(question_data, max_results)
```

### 2. Input Format
The system expects the same input format as the current system:
- `question_data['question']` - The question text
- `question_data['domain_info']` - Classification results with bias_type, etc.

### 3. Output Format
Returns list of dictionaries with:
- `name` - Person name
- `occupation` - Person's occupation
- `text` - Natural language description
- `source` - Either 'semantic_kg' or 'legacy_kg'

### 4. Fallback Behavior
- If semantic system fails, automatically falls back to legacy YAML data
- If both systems available, combines results for better coverage
- Graceful degradation ensures system always works

### 5. Performance Benefits
- SPARQL queries are faster than Python loops
- Better semantic matching through RDF relationships
- More sophisticated scoring and ranking
- Extensible for future bias types
"""
    
    with open("kg_semantic/integration/INTEGRATION_GUIDE.md", "w") as f:
        f.write(guide)
    
    print("Integration guide created: kg_semantic/integration/INTEGRATION_GUIDE.md")

if __name__ == "__main__":
    test_hybrid_system()
    create_integration_guide() 