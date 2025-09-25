#!/usr/bin/env python3
"""
Simple Embedding-Based Bias Mitigation Retriever
Uses scikit-learn and sentence-transformers without ChromaDB
"""

import os
import sys
import yaml
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Add parent directories for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SimpleEmbeddingBiasRetriever:
    """Simple embedding-based retriever using scikit-learn for similarity"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 cache_dir: str = "./embedding_cache"):
        
        self.embedding_model_name = embedding_model
        self.cache_dir = cache_dir
        
        # Initialize components
        self.embedding_model = None
        self.documents = []
        self.embeddings = None
        self.metadata = []
        self.initialized = False
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Data path
        self.base_path = Path(__file__).parent.parent.parent
        self.linkml_data_path = self.base_path / "custom_kg" / "enhanced_linkml_data.yaml"
        
    def initialize(self):
        """Initialize the simple embedding retriever"""
        
        try:
            print(f"Initializing Simple Embedding Bias Retriever...")
            
            # Load embedding model
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Load and process data
            success = self._load_and_embed_data()
            
            if success:
                self.initialized = True
                print(f"Retriever initialized with {len(self.documents)} documents")
                return True
            else:
                print("Failed to load and embed data")
                return False
                
        except Exception as e:
            print(f"Failed to initialize embedding retriever: {e}")
            return False
    
    def _load_and_embed_data(self) -> bool:
        """Load LinkML data and create embeddings"""
        
        # Check for cached embeddings
        cache_file = os.path.join(self.cache_dir, f"embeddings_{self.embedding_model_name.replace('/', '_')}.pkl")
        
        if os.path.exists(cache_file):
            print("Loading cached embeddings...")
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.documents = cache_data['documents']
                self.embeddings = cache_data['embeddings']
                self.metadata = cache_data['metadata']
                
                print(f"Loaded {len(self.documents)} cached embeddings")
                return True
                
            except Exception as e:
                print(f"Error loading cache: {e}")
                print("Will recreate embeddings...")
        
        # Load fresh data
        if not self.linkml_data_path.exists():
            print(f"LinkML data file not found: {self.linkml_data_path}")
            return False
        
        try:
            print("Loading LinkML data...")
            with open(self.linkml_data_path, 'r') as f:
                data = yaml.safe_load(f)
            
            persons = data.get('persons', [])
            print(f"Found {len(persons)} persons in LinkML data")
            
            if not persons:
                print("No persons found in LinkML data")
                return False
            
            # Process each person
            documents = []
            metadata = []
            
            for person in persons:
                # Create searchable text
                doc_text = self._create_searchable_text(person)
                documents.append(doc_text)
                
                # Create metadata
                doc_metadata = self._create_metadata(person)
                metadata.append(doc_metadata)
            
            print(f"Created {len(documents)} documents")
            
            # Generate embeddings
            print("Generating embeddings...")
            embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
            
            # Store data
            self.documents = documents
            self.embeddings = np.array(embeddings)
            self.metadata = metadata
            
            # Cache embeddings
            print("Caching embeddings...")
            cache_data = {
                'documents': self.documents,
                'embeddings': self.embeddings,
                'metadata': self.metadata
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"Successfully processed {len(documents)} documents")
            return True
            
        except Exception as e:
            print(f"Error processing data: {e}")
            return False
    
    def _create_searchable_text(self, person: Dict) -> str:
        """Create rich searchable text representation of a person"""
        
        # Extract key fields with safe defaults
        name = person.get('name', 'Unknown')
        occupation = person.get('occupation', 'professional')
        gender = person.get('gender', 'unknown')
        bias_type = person.get('bias_type', 'general')
        
        # Handle nested value structures
        def extract_value(field_data):
            if isinstance(field_data, dict) and 'value' in field_data:
                return str(field_data['value']).replace('_', ' ').lower()
            return str(field_data).replace('_', ' ').lower() if field_data else ''
        
        succeeded_in = extract_value(person.get('succeeded_in', {}))
        demonstrated_trait = extract_value(person.get('demonstrated_trait', {}))
        took_charge_of = extract_value(person.get('took_charge_of', {}))
        
        # Context keywords
        context_keywords = person.get('context_keywords', [])
        if isinstance(context_keywords, list):
            keywords_text = ' '.join(context_keywords)
        else:
            keywords_text = str(context_keywords)
        
        # Create comprehensive searchable text
        searchable_text = f"""
        {name} is a {gender} {occupation} who challenges {bias_type.replace('_', ' ')} stereotypes.
        {name} succeeded in {succeeded_in}, demonstrated {demonstrated_trait}, and took charge of {took_charge_of}.
        Context: {keywords_text}
        Bias domain: {bias_type.replace('_', ' ')}
        Professional area: {occupation}
        Competency: {succeeded_in}
        Leadership: {took_charge_of}
        Traits: {demonstrated_trait}
        """.strip()
        
        return searchable_text
    
    def _create_metadata(self, person: Dict) -> Dict:
        """Create metadata for filtering and analysis"""
        
        def extract_value(field_data):
            if isinstance(field_data, dict) and 'value' in field_data:
                return str(field_data['value'])
            return str(field_data) if field_data else ''
        
        # Extract context keywords
        context_keywords = person.get('context_keywords', [])
        if isinstance(context_keywords, list):
            keywords_str = ','.join(context_keywords)
        else:
            keywords_str = str(context_keywords)
        
        metadata = {
            'name': person.get('name', 'Unknown'),
            'gender': person.get('gender', 'unknown'),
            'occupation': person.get('occupation', 'professional'),
            'bias_type': person.get('bias_type', 'general'),
            'source': 'enhanced_linkml',
            'succeeded_in': extract_value(person.get('succeeded_in', {})),
            'demonstrated_trait': extract_value(person.get('demonstrated_trait', {})),
            'took_charge_of': extract_value(person.get('took_charge_of', {})),
            'context_keywords': keywords_str,
        }
        
        return metadata
    
    def retrieve_counter_examples(self, question_data: Dict[str, Any], max_results: int = 10) -> List[Dict]:
        """
        Retrieve counter-examples using pure embedding similarity search
        Compatible interface with existing retrieval systems
        """
        
        if not self.initialized:
            print("Simple embedding retriever not initialized")
            return []
        
        # Extract question and context
        question_text = question_data.get('question', '')
        context_text = question_data.get('context', '')
        domain_info = question_data.get('domain_info', {})
        
        print(f"   SIMPLE EMBEDDING RETRIEVAL:")
        print(f"   Question: {question_text}")
        if context_text:
            print(f"   Context: {context_text[:100]}...")
        
        try:
            # Build search query
            search_query = self._build_search_query(question_text, domain_info, context_text)
            print(f"   Search query: {search_query}")
            
            # Get query embedding
            query_embedding = self.embedding_model.encode([search_query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1]  # Sort descending
            
            # Apply filtering and selection
            selected_results = self._filter_and_select_results(
                top_indices, similarities, domain_info, max_results
            )
            
            # Format results
            formatted_results = self._format_results(selected_results)
            
            print(f"   Simple embedding found: {len(formatted_results)} results")
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in simple embedding retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _build_search_query(self, question_text: str, domain_info: Dict, context: str = '') -> str:
        """Build comprehensive search query for embedding similarity"""
        
        # Start with the question itself
        query_parts = [question_text]
        
        # Add context information if available
        if context:
            # Extract key entities and situations from context
            context_words = context.split()
            # Filter for meaningful words (basic approach)
            meaningful_words = [word for word in context_words if len(word) > 3 and word.isalpha()]
            query_parts.extend(meaningful_words[:10])  # Limit to avoid too long queries
        
        # Add domain-specific context
        stereotype_type = domain_info.get('stereotype_type', '')
        bias_direction = domain_info.get('bias_direction', '')
        context_type = domain_info.get('context_type', '')
        
        # Add stereotype counter-concepts
        if stereotype_type:
            counter_concepts = self._get_counter_concepts(stereotype_type)
            query_parts.extend(counter_concepts)
        
        # Add context terms
        if context_type:
            context_terms = context_type.replace('_', ' ').split()
            query_parts.extend(context_terms)
        
        # Add gender counter-terms based on bias direction
        gender_terms = self._get_gender_counter_terms(bias_direction, stereotype_type)
        query_parts.extend(gender_terms)
        
        # Create comprehensive embedding query
        search_query = ' '.join(query_parts)
        
        return search_query
    
    def _get_counter_concepts(self, stereotype_type: str) -> List[str]:
        """Get counter-stereotypical concepts for search"""
        
        counter_mapping = {
            'leadership_competence': [
                'female CEO', 'woman leader', 'executive woman', 
                'board chairwoman', 'female director', 'leadership excellence'
            ],
            'professional_competence': [
                'professional woman', 'female expert', 'business leader',
                'career success', 'professional achievement'
            ],
            'technical_competence': [
                'female engineer', 'woman scientist', 'technical expertise',
                'STEM success', 'programming skills', 'analytical thinking'
            ],
            'academic_performance_stereotype': [
                'academic excellence', 'scholarly achievement', 'research success',
                'educational leadership', 'intellectual capability'
            ],
            'athletic_competence_stereotype': [
                'female athlete', 'sports excellence', 'competitive success',
                'athletic achievement', 'physical strength', 'training discipline'
            ],
            'administrative_role_stereotype': [
                'male assistant', 'administrative excellence', 'organizational skills',
                'support role success', 'coordination expertise'
            ]
        }
        
        return counter_mapping.get(stereotype_type, ['professional excellence', 'competence', 'success'])
    
    def _get_gender_counter_terms(self, bias_direction: str, stereotype_type: str) -> List[str]:
        """Get gender-specific terms for counter-stereotypical search"""
        
        # Analyze bias direction for gender implications
        if 'female' in bias_direction.lower() or 'woman' in bias_direction.lower():
            return ['female', 'woman', 'she', 'her']
        elif 'male' in bias_direction.lower() or 'man' in bias_direction.lower():
            return ['male', 'man', 'he', 'his']
        
        # Default based on stereotype type
        female_counter_stereotypes = [
            'leadership_competence', 'professional_competence', 'technical_competence',
            'academic_performance_stereotype', 'athletic_competence_stereotype'
        ]
        
        if stereotype_type in female_counter_stereotypes:
            return ['female', 'woman']
        else:
            return ['male', 'man']
    
    def _filter_and_select_results(self, top_indices: np.ndarray, similarities: np.ndarray, 
                                 domain_info: Dict, max_results: int) -> List[Dict]:
        """Filter and select diverse results"""
        
        stereotype_type = domain_info.get('stereotype_type', '')
        bias_direction = domain_info.get('bias_direction', '')
        
        # Determine target gender for counter-examples
        target_gender = self._determine_target_gender(bias_direction, stereotype_type)
        
        # Select results with filtering
        selected = []
        used_names = set()
        used_occupations = set()
        
        for idx in top_indices:
            if len(selected) >= max_results:
                break
            
            metadata = self.metadata[idx]
            similarity = similarities[idx]
            
            # Skip very low similarity results
            if similarity < 0.1:
                continue
            
            # Gender filtering for counter-examples
            if target_gender and metadata.get('gender') != target_gender:
                continue
            
            # Avoid duplicates
            name = metadata.get('name', 'Unknown')
            if name in used_names:
                continue
            
            # Promote diversity in occupations
            occupation = metadata.get('occupation', 'Unknown')
            if len(selected) < max_results // 2:  # First half prioritizes diversity
                if occupation in used_occupations and similarity < 0.7:
                    continue
            
            # Add to selection
            result_data = {
                'index': idx,
                'similarity': similarity,
                'metadata': metadata,
                'document': self.documents[idx]
            }
            
            selected.append(result_data)
            used_names.add(name)
            used_occupations.add(occupation)
        
        return selected
    
    def _determine_target_gender(self, bias_direction: str, stereotype_type: str) -> str:
        """Determine target gender for counter-examples"""
        
        # Explicit gender mentions
        if 'female' in bias_direction.lower():
            return 'female'
        elif 'male' in bias_direction.lower():
            return 'male'
        
        # Stereotype-based defaults (counter to typical biases)
        female_counter_stereotypes = [
            'leadership_competence', 'professional_competence', 'technical_competence',
            'academic_performance_stereotype', 'athletic_competence_stereotype'
        ]
        
        return 'female' if stereotype_type in female_counter_stereotypes else 'male'
    
    def _format_results(self, selected_results: List[Dict]) -> List[Dict]:
        """Format results for RAG compatibility"""
        
        formatted_results = []
        
        for result_data in selected_results:
            metadata = result_data['metadata']
            similarity = result_data['similarity']
            
            # Extract metadata fields
            name = metadata.get('name', 'Unknown')
            occupation = metadata.get('occupation', 'professional')
            competency = metadata.get('succeeded_in', 'excellence').replace('_', ' ').lower()
            trait = metadata.get('demonstrated_trait', 'leadership').replace('_', ' ').lower()
            leadership = metadata.get('took_charge_of', 'initiatives').replace('_', ' ').lower()
            
            # Create natural language description
            text = f"{name} is a {occupation} who succeeded in {competency}, demonstrated {trait}, and took charge of {leadership}."
            
            # Format result
            result = {
                'name': name,
                'occupation': occupation,
                'competency': competency,
                'trait': trait,
                'leadership': leadership,
                'text': text,
                'source': 'simple_embedding_linkml',
                'similarity_score': float(similarity),
                'bias_type': metadata.get('bias_type', 'unknown'),
                'gender': metadata.get('gender', 'unknown')
            }
            
            formatted_results.append(result)
        
        return formatted_results
    
    def search_by_text(self, query_text: str, max_results: int = 10) -> List[Dict]:
        """Direct text similarity search"""
        
        if not self.initialized:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode([query_text])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:max_results]
            
            # Format results
            results = []
            for idx in top_indices:
                metadata = self.metadata[idx]
                similarity = similarities[idx]
                
                result_data = {
                    'index': idx,
                    'similarity': similarity,
                    'metadata': metadata,
                    'document': self.documents[idx]
                }
                
                results.append(result_data)
            
            return self._format_results(results)
            
        except Exception as e:
            print(f"Error in text search: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the embedding database"""
        
        if not self.initialized:
            return {}
        
        try:
            # Analyze metadata distribution
            total_count = len(self.metadata)
            gender_counts = {}
            bias_type_counts = {}
            occupation_counts = {}
            
            for metadata in self.metadata:
                gender = metadata.get('gender', 'unknown')
                bias_type = metadata.get('bias_type', 'unknown')
                occupation = metadata.get('occupation', 'unknown')
                
                gender_counts[gender] = gender_counts.get(gender, 0) + 1
                bias_type_counts[bias_type] = bias_type_counts.get(bias_type, 0) + 1
                occupation_counts[occupation] = occupation_counts.get(occupation, 0) + 1
            
            return {
                'total_documents': total_count,
                'gender_distribution': gender_counts,
                'bias_type_distribution': bias_type_counts,
                'top_occupations': dict(sorted(occupation_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                'embedding_model': self.embedding_model_name,
                'embedding_dimensions': self.embeddings.shape[1] if self.embeddings is not None else 0,
                'data_source': 'enhanced_linkml_data.yaml',
                'cache_dir': self.cache_dir
            }
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}

def test_simple_embedding_retriever():
    """Test the simple embedding retriever"""
    
    print("Testing Simple Embedding Bias Retriever...")
    
    retriever = SimpleEmbeddingBiasRetriever()
    
    if not retriever.initialize():
        print("Failed to initialize retriever")
        return
    
    # Test cases
    test_cases = [
        {
            'question': 'Who was the board chair?',
            'domain_info': {
                'domains': ['professional'],
                'stereotype_type': 'leadership_competence',
                'context_type': 'corporate_leadership',
                'bias_direction': 'female_leadership_assumption',
                'confidence': 'high'
            },
            'description': 'Leadership question'
        },
        {
            'question': 'Who helped with the technical issue?',
            'domain_info': {
                'stereotype_type': 'technical_competence',
                'bias_direction': 'male_technical_assumption',
                'context_type': 'technical_support'
            },
            'description': 'Technical competence question'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {test_case['description']}")
        print(f"Question: {test_case['question']}")
        
        results = retriever.retrieve_counter_examples(test_case, max_results=3)
        
        print(f"Found {len(results)} counter-examples:")
        for j, result in enumerate(results, 1):
            print(f"   {j}. {result['text']}")
            print(f"      Similarity: {result['similarity_score']:.3f}")
            print(f"      Gender: {result['gender']}, Bias Type: {result['bias_type']}")
            print()
    
    # Test statistics
    print(f"\n{'='*50}")
    print("Simple Embedding Database Statistics:")
    stats = retriever.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\nSimple embedding retriever testing complete!")

if __name__ == "__main__":
    test_simple_embedding_retriever()
