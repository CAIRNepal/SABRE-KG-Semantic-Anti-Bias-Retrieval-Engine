#!/usr/bin/env python3
"""
Semantic Retriever for Bias Mitigation using SPARQL
This replaces the current Python-based retrieval system
"""

import sys
import os
from typing import List, Dict, Any

# Add parent directories for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.triple_store import BiasKnowledgeGraph
from queries.query_engine import SPARQLQueryEngine

class SemanticBiasRetriever:
    """Semantic retriever using SPARQL queries for bias mitigation"""
    
    def __init__(self):
        self.kg = BiasKnowledgeGraph()
        self.query_engine = None
        self.initialized = False
    
    def initialize(self):
        """Initialize the semantic retriever"""
        if self.kg.initialize():
            self.query_engine = SPARQLQueryEngine(self.kg)
            self.initialized = True
            print("Semantic retriever initialized successfully")
            return True
        else:
            print("Failed to initialize semantic retriever")
            return False
    
    def retrieve_counter_examples(self, question_data: Dict[str, Any], max_results: int = 10) -> List[Dict]:
        """
        Simple, robust retrieval using direct SPARQL queries
        """
        
        if not self.initialized:
            print("Semantic retriever not initialized")
            return []
        
        # Extract classification data
        domain_info = question_data.get('domain_info', {})
        stereotype_type = domain_info.get('stereotype_type', 'general')
        bias_direction = domain_info.get('bias_direction', 'unknown')
        
        print(f"   ENHANCED SEMANTIC RETRIEVAL:")
        print(f"   Stereotype type: {stereotype_type}")
        print(f"   Bias direction: {bias_direction}")
        
        try:
            # Map stereotype to bias type
            mapped_bias_type = self._map_stereotype_to_bias_type(stereotype_type)
            print(f"   Mapped to: {mapped_bias_type}")
            
            # Simple SPARQL query - just get examples of this bias type
            query = f"""
            PREFIX targeted: <https://example.org/targeted-failing-domains/>
            SELECT ?person ?name ?occupation WHERE {{
                ?person rdf:type targeted:Person ;
                        targeted:name ?name ;
                        targeted:occupation ?occupation ;
                        targeted:bias_type <https://example.org/targeted-failing-domains/BiasTypeEnum#{mapped_bias_type}> .
            }} ORDER BY ?name LIMIT {max_results}
            """
            
            print(f"   Executing query for: {mapped_bias_type}")
            results = self.kg.query(query)
            print(f"   Raw SPARQL results: {len(results)}")
            
            if results:
                # Convert to RAG format
                formatted_results = []
                for result in results:
                    person_uri, name, occupation = result
                    formatted_results.append({
                        'name': str(name),
                        'occupation': str(occupation),
                        'text': f"{name} is a {occupation} who challenges stereotypes about {mapped_bias_type.replace('_', ' ')}.",
                        'source': 'enhanced_semantic_kg',
                        'competency': 'excellence',
                        'trait': 'success',
                        'leadership': 'initiative'
                    })
                
                print(f"   Enhanced semantic found: {len(formatted_results)} results")
                return formatted_results
            
            print(f"   No examples found for bias type: {mapped_bias_type}")
            return []
            
        except Exception as e:
            print(f"Error in enhanced semantic retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _query_by_bias_type_and_gender(self, bias_type: str, gender: str) -> List:
        """Query for persons by bias type and gender"""
        
        query = f"""
        PREFIX targeted: <https://example.org/targeted-failing-domains/>
        SELECT ?person ?name ?occupation ?competency ?trait ?leadership WHERE {{
            ?person rdf:type targeted:Person ;
                    targeted:name ?name ;
                    targeted:occupation ?occupation ;
                    targeted:gender <https://example.org/targeted-failing-domains/GenderEnum#{gender}> ;
                    targeted:bias_type <https://example.org/targeted-failing-domains/BiasTypeEnum#{bias_type}> .
            
            OPTIONAL {{
                ?person targeted:succeeded_in ?comp_val .
                ?comp_val targeted:value ?competency .
            }}
            OPTIONAL {{
                ?person targeted:demonstrated_trait ?trait_val .
                ?trait_val targeted:value ?trait .
            }}
            OPTIONAL {{
                ?person targeted:took_charge_of ?lead_val .
                ?lead_val targeted:value ?leadership .
            }}
        }} ORDER BY ?name LIMIT 10
        """
        
        try:
            return self.kg.query(query)
        except Exception as e:
            print(f"   Query error (bias+gender): {e}")
            return []

    def _query_by_bias_type_only(self, bias_type: str) -> List:
        """Query for persons by bias type only (ignore gender)"""
        
        query = f"""
        PREFIX targeted: <https://example.org/targeted-failing-domains/>
        SELECT ?person ?name ?occupation ?competency ?trait ?leadership WHERE {{
            ?person rdf:type targeted:Person ;
                    targeted:name ?name ;
                    targeted:occupation ?occupation ;
                    targeted:bias_type <https://example.org/targeted-failing-domains/BiasTypeEnum#{bias_type}> .
            
            OPTIONAL {{
                ?person targeted:succeeded_in ?comp_val .
                ?comp_val targeted:value ?competency .
            }}
            OPTIONAL {{
                ?person targeted:demonstrated_trait ?trait_val .
                ?trait_val targeted:value ?trait .
            }}
            OPTIONAL {{
                ?person targeted:took_charge_of ?lead_val .
                ?lead_val targeted:value ?leadership .
            }}
        }} ORDER BY ?name LIMIT 10
        """
        
        try:
            return self.kg.query(query)
        except Exception as e:
            print(f"   Query error (bias only): {e}")
            return []

    def _get_stereotype_specific_examples(self, stereotype_type, context_type, bias_direction, answer_pattern):
        """Get high-priority examples using proven stereotype matching logic"""
        
        candidates = []
        
        if stereotype_type == "academic_performance_stereotype":
            # Female STEM excellence (score: 20)
            female_stem = self.query_engine.find_academic_performance_counter_examples(gender="female")
            for result in female_stem:
                candidates.append({
                    'sparql_result': result,
                    'score': 20,
                    'match_type': 'academic_performance_female_stem',
                    'priority': 'high'
                })
                print(f"     ACADEMIC PERFORMANCE MATCH (Female STEM): {result[1]}")
            
            # Male help-seeking in STEM (score: 15)  
            male_help = self.query_engine.find_help_seeking_examples(gender="male", domain="technical")
            for result in male_help:
                candidates.append({
                    'sparql_result': result,
                    'score': 15,
                    'match_type': 'academic_performance_male_help',
                    'priority': 'high'
                })
                print(f"     ACADEMIC STRUGGLE MATCH (Male help-seeking): {result[1]}")
        
        elif stereotype_type == "leadership_stereotype":
            # Female leadership excellence (score: 25)
            female_leadership = self.query_engine.find_leadership_counter_examples(gender="female")
            for result in female_leadership:
                candidates.append({
                    'sparql_result': result,
                    'score': 25,
                    'match_type': 'female_leadership_excellence',
                    'priority': 'high'
                })
                print(f"     FEMALE LEADERSHIP MATCH: {result[1]}")
        
        elif stereotype_type == "administrative_role_stereotype":
            # Male admin role excellence (score: 25)
            male_admin = self.query_engine.find_administrative_role_examples(gender="male")
            for result in male_admin:
                candidates.append({
                    'sparql_result': result,
                    'score': 25,
                    'match_type': 'male_admin_role_excellence',
                    'priority': 'high'
                })
                print(f"     MALE ADMIN ROLE MATCH: {result[1]}")
        
        elif stereotype_type == "athletic_competence_stereotype":
            # Female athletic excellence (score: 25)
            female_athletic = self.query_engine.find_athletic_competence_examples(gender="female")
            for result in female_athletic:
                candidates.append({
                    'sparql_result': result,
                    'score': 25,
                    'match_type': 'female_athletic_excellence',
                    'priority': 'high'
                })
                print(f"     FEMALE ATHLETE MATCH: {result[1]}")
        
        # Add fallback for unmapped stereotypes
        if not candidates:
            # Try basic gender/bias type matching
            mapped_bias_type = self._map_stereotype_to_bias_type(stereotype_type)
            gender = self._determine_counter_gender(bias_direction)
            
            basic_results = self.query_engine.find_persons_by_gender_and_bias_type(gender, mapped_bias_type)
            for result in basic_results:
                candidates.append({
                    'sparql_result': result,
                    'score': 10,
                    'match_type': 'basic_fallback',
                    'priority': 'medium'
                })
                print(f"     BASIC FALLBACK MATCH: {result[1]}")
        
        return candidates

    def _get_context_specific_examples(self, context_type):
        """Get context-specific examples (score: 12)"""
        
        candidates = []
        context_results = self.query_engine.find_context_aware_examples_enhanced(context_type)
        
        for result in context_results:
            candidates.append({
                'sparql_result': result,
                'score': 12,
                'match_type': f'context_{context_type}',
                'priority': 'medium'
            })
            print(f"     Context match ({context_type}): {result[1]}")
        
        return candidates

    def _get_bias_direction_examples(self, bias_direction):
        """Get bias direction examples (score: 10)"""
        
        candidates = []
        
        if bias_direction == "academic_gender_reversal":
            # Female technical + Male help-seeking
            female_tech = self.query_engine.find_academic_performance_counter_examples(gender="female")
            male_help = self.query_engine.find_help_seeking_examples(gender="male")
            
            for result in female_tech + male_help:
                candidates.append({
                    'sparql_result': result,
                    'score': 10,
                    'match_type': 'gender_reversal',
                    'priority': 'low'
                })
                print(f"     Gender reversal match: {result[1]}")
        
        elif bias_direction == "female_leadership_assumption":
            # Female leadership examples
            female_leadership = self.query_engine.find_leadership_counter_examples(gender="female")
            for result in female_leadership:
                candidates.append({
                    'sparql_result': result,
                    'score': 10,
                    'match_type': 'female_leadership_bias',
                    'priority': 'low'
                })
                print(f"     Female leadership bias match: {result[1]}")
        
        return candidates

    def _combine_and_score_candidates(self, high_priority, context, bias, confidence, question, context_text):
        """Combine candidates and apply proven scoring logic"""
        
        # Collect all candidates by person name (avoid duplicates)
        all_candidates = {}
        
        for candidate_list in [high_priority, context, bias]:
            for candidate in candidate_list:
                result = candidate['sparql_result']
                name = str(result[1]) if len(result) > 1 else 'Unknown'
                
                if name not in all_candidates:
                    all_candidates[name] = {
                        'name': name,
                        'sparql_data': result,
                        'total_score': 0,
                        'match_types': [],
                        'priority': candidate['priority']
                    }
                
                # Add score (avoid double-counting same person)
                all_candidates[name]['total_score'] += candidate['score']
                all_candidates[name]['match_types'].append(candidate['match_type'])
        
        # Apply confidence multiplier (proven logic)
        confidence_multiplier = {
            'high': 1.2,
            'medium': 1.0,
            'low': 0.8
        }.get(confidence, 1.0)
        
        # Apply keyword matching bonus (proven logic)
        question_words = set(question.split())
        
        for candidate in all_candidates.values():
            # Confidence multiplier
            candidate['total_score'] *= confidence_multiplier
            
            # Keyword bonus (simplified but consistent with proven system)
            if self._has_keyword_overlap(candidate['sparql_data'], question_words):
                candidate['total_score'] += 3  # Keyword bonus like proven system
        
        # Convert to list and sort
        sorted_candidates = list(all_candidates.values())
        sorted_candidates.sort(key=lambda x: x['total_score'], reverse=True)
        
        print(f"\n     TOP SCORED EXAMPLES:")
        for i, candidate in enumerate(sorted_candidates[:10]):
            print(f"      {i+1:2d}. {candidate['total_score']:5.1f} - {candidate['name']}")
        
        return sorted_candidates

    def _apply_diversity_selection(self, candidates, max_results):
        """Apply diversity selection like proven system"""
        
        selected = []
        used_names = set()
        used_occupations = set()
        used_genders = []
        
        for candidate in candidates:
            name = candidate['name']
            sparql_data = candidate['sparql_data']
            
            # Skip duplicates
            if name in used_names:
                continue
            
            # Extract occupation from SPARQL result
            occupation = str(sparql_data[2]) if len(sparql_data) > 2 else 'Unknown'
            
            # Get gender from data or determine from context
            gender = self._get_person_gender_from_data(sparql_data)
            
            # High score gets priority (proven logic)
            if candidate['total_score'] > 15:
                selected.append(self._format_candidate_result(candidate, occupation, gender))
                used_names.add(name)
                used_occupations.add(occupation)
                used_genders.append(gender)
            
            # Diversity selection for lower scores (proven logic)
            elif len(used_genders) < 2 and gender not in used_genders:
                selected.append(self._format_candidate_result(candidate, occupation, gender))
                used_names.add(name)
                used_occupations.add(occupation)
                used_genders.append(gender)
            
            elif occupation not in used_occupations:
                selected.append(self._format_candidate_result(candidate, occupation, gender))
                used_names.add(name)
                used_occupations.add(occupation)
                used_genders.append(gender)
            
            if len(selected) >= max_results:
                break
        
        # Fill remaining slots if needed (proven system logic)
        if len(selected) < max_results:
            for candidate in candidates:
                if candidate['name'] not in used_names:
                    occupation = str(candidate['sparql_data'][2]) if len(candidate['sparql_data']) > 2 else 'Unknown'
                    gender = self._get_person_gender_from_data(candidate['sparql_data'])
                    selected.append(self._format_candidate_result(candidate, occupation, gender))
                    used_names.add(candidate['name'])
                    if len(selected) >= max_results:
                        break
        
        print(f"\n     FINAL SELECTED EXAMPLES:")
        for i, ex in enumerate(selected):
            print(f"      {i+1}. {ex.get('score', 0):5.1f} - {ex['name']} ({ex.get('occupation', 'Unknown')})")
        
        return selected

    def _format_candidate_result(self, candidate, occupation, gender):
        """Format result for RAG compatibility"""
        
        name = candidate['name']
        sparql_data = candidate['sparql_data']
        
        # Extract details from SPARQL result if available
        competency = 'leadership'
        trait = 'excellence'
        leadership = 'initiatives'
        
        if len(sparql_data) > 3:
            competency = str(sparql_data[3]).split('#')[-1].replace('_', ' ').lower()
        if len(sparql_data) > 4:
            trait = str(sparql_data[4]).split('#')[-1].replace('_', ' ').lower()
        if len(sparql_data) > 5:
            leadership = str(sparql_data[5]).split('#')[-1].replace('_', ' ').lower()
        
        # Create rich description (proven system style)
        text = f"{name} is a {occupation} who excelled in {competency}, demonstrated {trait}, and successfully led {leadership}."
        
        return {
            'name': name,
            'occupation': occupation,
            'text': text,
            'source': 'enhanced_semantic_kg',
            'competency': competency,
            'trait': trait,
            'leadership': leadership,
            'score': candidate['total_score'],
            'match_types': candidate['match_types']
        }

    def _has_keyword_overlap(self, sparql_data, question_words):
        """Check if SPARQL result has keyword overlap with question"""
        
        # Get all text content from SPARQL result
        text_content = []
        for item in sparql_data:
            text_content.extend(str(item).lower().split())
        
        result_words = set(text_content)
        overlap = len(question_words & result_words)
        return overlap > 0

    def _get_person_gender_from_data(self, sparql_data):
        """Extract gender from SPARQL data or default"""
        
        # For now, return default - could be enhanced with proper gender extraction
        return 'female'  # Default, matches proven system behavior
    
    def _determine_counter_gender(self, bias_direction: str) -> str:
        """Determine the gender for counter-examples based on bias direction"""
        
        if 'female' in bias_direction.lower() or 'woman' in bias_direction.lower():
            return 'female'
        elif 'male' in bias_direction.lower() or 'man' in bias_direction.lower():
            return 'male'
        else:
            return 'female'  # Default to female counter-examples
    
    def _extract_context_keywords(self, question_data: Dict, domain_info: Dict) -> List[str]:
        """Extract context keywords from question and domain data"""
        
        keywords = []
        
        # From question text
        question = question_data.get('question', '').lower()
        
        # Common leadership/professional keywords
        leadership_keywords = ['board', 'executive', 'director', 'manager', 'leader', 'ceo', 'president', 'chairman']
        for keyword in leadership_keywords:
            if keyword in question:
                keywords.append(keyword)
        
        # From domain information
        domains = domain_info.get('domains', [])
        if 'professional' in domains:
            keywords.extend(['corporate', 'business', 'company'])
        
        context_type = domain_info.get('context_type', '')
        if context_type:
            keywords.append(context_type.replace('_', ' ').lower())
        
        # Remove duplicates while preserving order
        unique_keywords = []
        for keyword in keywords:
            if keyword not in unique_keywords:
                unique_keywords.append(keyword)
        
        return unique_keywords[:5]  # Limit to top 5 keywords
    
    def _format_results_for_rag(self, sparql_results: List) -> List[Dict]:
        """Format SPARQL results for RAG system compatibility"""
        
        formatted_results = []
        
        for result in sparql_results:
            if len(result) >= 6:  # Full result with all fields
                person_uri, name, occupation, competency, trait, leadership = result[:6]
                
                # Extract readable names from URIs
                competency_name = self._clean_uri_name(str(competency))
                trait_name = self._clean_uri_name(str(trait))
                leadership_name = self._clean_uri_name(str(leadership))
                
                # Create rich text description
                text = self._create_rich_description(
                    str(name), str(occupation), competency_name, trait_name, leadership_name
                )
                
                formatted_result = {
                    'name': str(name),
                    'occupation': str(occupation),
                    'competency': competency_name,
                    'trait': trait_name,
                    'leadership': leadership_name,
                    'text': text,
                    'source': 'semantic_kg'
                }
                formatted_results.append(formatted_result)
            
            elif len(result) >= 3:  # Basic result with name and occupation
                person_uri, name, occupation = result[:3]
                
                formatted_result = {
                    'name': str(name),
                    'occupation': str(occupation),
                    'text': f"{name} is a {occupation} who challenges stereotypes.",
                    'source': 'semantic_kg'
                }
                formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _clean_uri_name(self, uri_string: str) -> str:
        """Clean URI to extract readable name"""
        if '#' in uri_string:
            name = uri_string.split('#')[-1]
        else:
            name = uri_string.split('/')[-1]
        
        # Convert CamelCase to readable format
        import re
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        return name.lower()
    
    def _create_rich_description(self, name: str, occupation: str, competency: str, trait: str, leadership: str) -> str:
        """Create a rich natural language description"""
        
        # Templates for variety
        templates = [
            f"{name} is a {occupation} who excelled in {competency}, demonstrated {trait}, and successfully led {leadership}.",
            f"As a {occupation}, {name} achieved success in {competency}, showing {trait} while taking charge of {leadership}.",
            f"{name}, working as a {occupation}, succeeded in {competency} and demonstrated {trait} when leading {leadership}.",
            f"In their role as a {occupation}, {name} proved their expertise in {competency}, exhibited {trait}, and took responsibility for {leadership}."
        ]
        
        # Choose template based on name hash for consistency
        template_index = hash(name) % len(templates)
        return templates[template_index]
    
    def _map_stereotype_to_bias_type(self, stereotype_type: str) -> str:
        """Map question stereotype types to KG bias types"""
        mapping = {
            # Existing mappings
            'academic_performance_stereotype': 'academic_performance_stereotype',
            'leadership_stereotype': 'leadership_competence', 
            'administrative_role_stereotype': 'administrative_role_stereotype',
            'athletic_competence_stereotype': 'athletic_competence_stereotype',
            
            # NEW COMPREHENSIVE MAPPINGS for Enhanced KG
            'relationship_violence_stereotype': 'relationship_violence_stereotype',
            'professional_competence': 'professional_competence',
            'general_feminine_stereotype': 'general_feminine_stereotype',
            'general_masculine_stereotype': 'general_masculine_stereotype',
            'mental_health_stereotype': 'mental_health_stereotype',
            'technical_competence': 'technical_competence',
            'physical_competence': 'physical_competence',
            'emotional_competence': 'emotional_competence',
            
            # Fallback mappings for variations
            'leadership_competence': 'leadership_competence',
            'aggression_violence': 'relationship_violence_stereotype',
        }
        return mapping.get(stereotype_type, stereotype_type)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        
        if not self.initialized:
            return {}
        
        # Count total persons
        total_persons = len(list(self.kg.graph.subjects(
            self.kg.namespaces['rdf'].type, 
            self.kg.namespaces['targeted'].Person
        )))
        
        # Count by gender
        female_count = len(self.kg.find_persons_by_criteria(gender="female"))
        male_count = len(self.kg.find_persons_by_criteria(gender="male"))
        
        # Count by bias type
        leadership_count = len(self.kg.find_persons_by_criteria(bias_type="leadership_competence"))
        technical_count = len(self.kg.find_persons_by_criteria(bias_type="technical_competence"))
        
        return {
            'total_persons': total_persons,
            'female_persons': female_count,
            'male_persons': male_count,
            'leadership_competence_examples': leadership_count,
            'technical_competence_examples': technical_count,
            'total_triples': len(self.kg.graph)
        }

def test_semantic_retriever():
    """Test the semantic retriever"""
    
    print("Testing Semantic Bias Retriever...")
    
    retriever = SemanticBiasRetriever()
    
    if not retriever.initialize():
        print("Failed to initialize retriever")
        return
    
    # Test data similar to current system
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
    
    print("\nTesting retrieval for leadership question...")
    results = retriever.retrieve_counter_examples(test_question, max_results=3)
    
    print(f"Found {len(results)} counter-examples:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['text']}")
    
    # Test statistics
    print("\nKnowledge Graph Statistics:")
    stats = retriever.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\nSemantic retriever testing complete!")

if __name__ == "__main__":
    test_semantic_retriever() 