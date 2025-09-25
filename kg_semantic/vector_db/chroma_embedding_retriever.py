#!/usr/bin/env python3
"""
ChromaDB-based Bias Mitigation Retriever
Simple integration of ChromaDB with the embedding approach
"""

import os
import sys
import yaml
import chromadb
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Add parent directories for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ChromaEmbeddingBiasRetriever:
    """ChromaDB-based retriever for bias mitigation using LinkML data"""
    
    def __init__(self, 
                 collection_name: str = "linkml_bias_mitigation_chroma",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_data"):
        
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        
        # Initialize components
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.initialized = False
        
        # Data path
        self.base_path = Path(__file__).parent.parent.parent
        self.linkml_data_path = self.base_path / "custom_kg" / "enhanced_linkml_data.yaml"
        
    def initialize(self):
        """Initialize the ChromaDB embedding retriever"""
        
        try:
            print(f"Initializing ChromaDB Embedding Bias Retriever...")
            
            # Load embedding model
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Check if collection exists and has data
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                count = self.collection.count()
                if count > 0:
                    print(f"Found existing collection with {count} documents")
                    self.initialized = True
                    return True
                else:
                    print("Collection exists but is empty, will rebuild...")
            except Exception:
                print("Collection doesn't exist, will create new one...")
            
            # Create collection and index data
            success = self._create_and_index_collection()
            
            if success:
                self.initialized = True
                print(f"ChromaDB retriever initialized successfully")
                return True
            else:
                print("Failed to initialize ChromaDB retriever")
                return False
                
        except Exception as e:
            print(f"Failed to initialize ChromaDB retriever: {e}")
            return False
    
    def _create_and_index_collection(self) -> bool:
        """Create ChromaDB collection and index LinkML data"""
        
        try:
            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                print(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                pass  # Collection doesn't exist
            
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "LinkML bias mitigation examples"}
            )
            
            # Load LinkML data
            if not self.linkml_data_path.exists():
                print(f"LinkML data file not found: {self.linkml_data_path}")
                return False
            
            with open(self.linkml_data_path, 'r') as f:
                data = yaml.safe_load(f)
            
            persons = data.get('persons', [])
            if not persons:
                print("No persons found in LinkML data")
                return False
            
            print(f"Processing {len(persons)} persons for ChromaDB...")
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            used_ids = set()
            
            for i, person in enumerate(persons):
                # Create searchable text
                doc_text = self._create_searchable_text(person)
                documents.append(doc_text)
                
                # Create metadata
                metadata = self._create_metadata(person)
                metadatas.append(metadata)
                
                # Create unique ID
                person_id = person.get('id', f"person_{i:04d}")
                if isinstance(person_id, str) and ':' in person_id:
                    person_id = person_id.replace(':', '_')
                
                # Ensure ID uniqueness
                original_id = str(person_id)
                counter = 0
                while original_id in used_ids:
                    counter += 1
                    original_id = f"{person_id}_{counter}"
                
                used_ids.add(original_id)
                ids.append(original_id)
            
            # Add to ChromaDB collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Successfully indexed {len(documents)} documents in ChromaDB")
            return True
            
        except Exception as e:
            print(f"Error creating ChromaDB collection: {e}")
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
        Retrieve counter-examples using ChromaDB similarity search
        Compatible interface with existing retrieval systems
        """
        
        if not self.initialized:
            print("ChromaDB retriever not initialized")
            return []
        
        # Extract question and context
        question_text = question_data.get('question', '')
        context_text = question_data.get('context', '')
        domain_info = question_data.get('domain_info', {})
        
        print(f"   CHROMADB EMBEDDING RETRIEVAL:")
        print(f"   Question: {question_text}")
        if context_text:
            print(f"   Context: {context_text[:100]}...")
        
        try:
            # Build search query
            search_query = self._build_search_query(question_text, domain_info, context_text)
            print(f"   Search query: {search_query}")
            
            # Determine metadata filters
            where_filter = self._build_metadata_filter(domain_info)
            
            # Perform ChromaDB search
            results = self.collection.query(
                query_texts=[search_query],
                n_results=max_results * 2,  # Get extra for filtering
                where=where_filter if where_filter else None
            )
            
            # Format results
            formatted_results = self._format_chroma_results(results, max_results)
            
            print(f"   ChromaDB found: {len(formatted_results)} results")
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in ChromaDB retrieval: {e}")
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
            # Filter for meaningful words
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
            'leadership_competence': ['female CEO', 'woman leader', 'executive woman', 'board chairwoman'],
            'professional_competence': ['professional woman', 'female expert', 'business leader'],
            'technical_competence': ['female engineer', 'woman scientist', 'technical expertise'],
            'academic_performance_stereotype': ['academic excellence', 'scholarly achievement'],
            'athletic_competence_stereotype': ['female athlete', 'sports excellence'],
            'administrative_role_stereotype': ['male assistant', 'administrative excellence'],
            'mental_health_stereotype': ['emotional intelligence', 'counseling skills']
        }
        
        return counter_mapping.get(stereotype_type, ['professional excellence', 'competence'])
    
    def _get_gender_counter_terms(self, bias_direction: str, stereotype_type: str) -> List[str]:
        """Get gender-specific terms for counter-stereotypical search"""
        
        # Analyze bias direction for gender implications
        if 'female' in bias_direction.lower() or 'woman' in bias_direction.lower():
            return ['female', 'woman']
        elif 'male' in bias_direction.lower() or 'man' in bias_direction.lower():
            return ['male', 'man']
        
        # Default based on stereotype type
        female_counter_stereotypes = [
            'leadership_competence', 'professional_competence', 'technical_competence',
            'academic_performance_stereotype', 'athletic_competence_stereotype'
        ]
        
        if stereotype_type in female_counter_stereotypes:
            return ['female', 'woman']
        else:
            return ['male', 'man']
    
    def _build_metadata_filter(self, domain_info: Dict) -> Optional[Dict]:
        """Build metadata filter for targeted search using ChromaDB filter format"""
        
        stereotype_type = domain_info.get('stereotype_type', '')
        bias_direction = domain_info.get('bias_direction', '')
        
        # Gender filter for counter-examples
        target_gender = self._determine_target_gender(bias_direction, stereotype_type)
        
        # For ChromaDB, we need to use proper operator format
        # Let's keep it simple and just filter by gender for now
        if target_gender:
            return {"gender": {"$eq": target_gender}}
        
        return None
    
    def _determine_target_gender(self, bias_direction: str, stereotype_type: str) -> str:
        """Determine target gender for counter-examples"""
        
        # Explicit gender mentions
        if 'female' in bias_direction.lower():
            return 'female'
        elif 'male' in bias_direction.lower():
            return 'male'
        
        # Stereotype-based defaults
        female_counter_stereotypes = [
            'leadership_competence', 'professional_competence', 'technical_competence',
            'academic_performance_stereotype', 'athletic_competence_stereotype'
        ]
        
        return 'female' if stereotype_type in female_counter_stereotypes else 'male'
    
    def _get_relevant_bias_types(self, stereotype_type: str) -> List[str]:
        """Get relevant bias types from LinkML data"""
        
        bias_type_mapping = {
            'leadership_competence': ['leadership_competence'],
            'professional_competence': ['professional_competence'],
            'technical_competence': ['technical_competence'],
            'academic_performance_stereotype': ['academic_performance_stereotype'],
            'athletic_competence_stereotype': ['athletic_competence_stereotype'],
            'administrative_role_stereotype': ['administrative_role_stereotype'],
            'mental_health_stereotype': ['mental_health_stereotype']
        }
        
        return bias_type_mapping.get(stereotype_type, [])
    
    def _format_chroma_results(self, results: Dict, max_results: int) -> List[Dict]:
        """Format ChromaDB search results for RAG compatibility"""
        
        formatted_results = []
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0] 
        distances = results.get('distances', [[]])[0]
        
        # Apply diversity selection
        selected_results = self._select_diverse_results(documents, metadatas, distances, max_results)
        
        for doc, metadata, distance in selected_results:
            # Calculate similarity score (1 - distance)
            similarity_score = 1 - distance
            
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
                'source': 'chromadb_embedding_search',
                'similarity_score': float(similarity_score),
                'embedding_distance': float(distance),
                'bias_type': metadata.get('bias_type', 'unknown'),
                'gender': metadata.get('gender', 'unknown')
            }
            
            formatted_results.append(result)
        
        return formatted_results
    
    def _select_diverse_results(self, documents, metadatas, distances, max_results):
        """Select diverse results to avoid redundancy"""
        
        selected = []
        used_names = set()
        used_occupations = set()
        
        # Combine data for sorting
        combined = list(zip(documents, metadatas, distances))
        # Sort by distance (lower = more similar)
        combined.sort(key=lambda x: x[2])
        
        for doc, metadata, distance in combined:
            if len(selected) >= max_results:
                break
            
            name = metadata.get('name', 'Unknown')
            occupation = metadata.get('occupation', 'Unknown')
            
            # Skip exact duplicates
            if name in used_names:
                continue
            
            # Prefer diversity in occupations for first few results
            if len(selected) < max_results // 2:
                if occupation in used_occupations and distance > 0.3:  # Similarity threshold
                    continue
            
            selected.append((doc, metadata, distance))
            used_names.add(name)
            used_occupations.add(occupation)
        
        return selected
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB database"""
        
        if not self.initialized:
            return {}
        
        try:
            # Get collection info
            count = self.collection.count()
            
            # Get sample data for analysis
            sample_results = self.collection.get(limit=count)
            metadatas = sample_results.get('metadatas', [])
            
            # Analyze metadata distribution
            gender_counts = {}
            bias_type_counts = {}
            occupation_counts = {}
            
            for metadata in metadatas:
                gender = metadata.get('gender', 'unknown')
                bias_type = metadata.get('bias_type', 'unknown')
                occupation = metadata.get('occupation', 'unknown')
                
                gender_counts[gender] = gender_counts.get(gender, 0) + 1
                bias_type_counts[bias_type] = bias_type_counts.get(bias_type, 0) + 1
                occupation_counts[occupation] = occupation_counts.get(occupation, 0) + 1
            
            return {
                'total_documents': count,
                'gender_distribution': gender_counts,
                'bias_type_distribution': bias_type_counts,
                'top_occupations': dict(sorted(occupation_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                'embedding_model': self.embedding_model_name,
                'collection_name': self.collection_name,
                'data_source': 'enhanced_linkml_data.yaml',
                'storage': 'chromadb'
            }
            
        except Exception as e:
            print(f"Error getting ChromaDB statistics: {e}")
            return {}

def test_chroma_embedding_retriever():
    """Test the ChromaDB embedding retriever"""
    
    print("Testing ChromaDB Embedding Bias Retriever...")
    
    retriever = ChromaEmbeddingBiasRetriever()
    
    if not retriever.initialize():
        print("Failed to initialize ChromaDB retriever")
        return
    
    # Test cases
    test_cases = [
        {
            'question': 'Who was the board chair?',
            'context': 'The company board meeting discussed leadership changes.',
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
            'question': 'Who solved the technical problem?',
            'context': 'The engineering team faced a complex software issue.',
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
    print("ChromaDB Statistics:")
    stats = retriever.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\nChromaDB embedding retriever testing complete!")

if __name__ == "__main__":
    test_chroma_embedding_retriever()
