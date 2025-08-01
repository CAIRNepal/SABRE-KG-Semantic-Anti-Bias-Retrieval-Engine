#!/usr/bin/env python3
"""
SPARQL Query Engine for Bias Mitigation Retrieval
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.triple_store import BiasKnowledgeGraph

class SPARQLQueryEngine:
    """Engine for executing parameterized SPARQL queries"""
    
    def __init__(self, kg: Optional[BiasKnowledgeGraph] = None):
        self.kg = kg or BiasKnowledgeGraph()
        self.query_templates = {}
        self.load_query_templates()
    
    def load_query_templates(self):
        """Load SPARQL query templates from file"""
        
        template_file = "kg_semantic/queries/retrieval_queries.sparql"
        
        if not os.path.exists(template_file):
            print(f"Query template file not found: {template_file}")
            return False
        
        try:
            with open(template_file, 'r') as f:
                content = f.read()
            
            # Parse templates (split by query comments)
            queries = re.split(r'# Query \d+:', content)
            
            for i, query_block in enumerate(queries[1:], 1):  # Skip first empty split
                # Extract template name from comment
                lines = query_block.strip().split('\n')
                if lines:
                    comment_line = lines[0].strip()
                    # Extract the description as template name (everything after "# Parameters:")
                    if comment_line:
                        template_name = comment_line
                    else:
                        template_name = f"query_{i}"
                    
                    # Extract actual SPARQL query (everything after the comment lines)
                    sparql_lines = []
                    in_query = False
                    for line in lines:
                        if line.strip().startswith('PREFIX') or in_query:
                            in_query = True
                            sparql_lines.append(line)
                    
                    if sparql_lines:
                        sparql_query = '\n'.join(sparql_lines)
                        self.query_templates[template_name] = sparql_query
            
            print(f"Loaded {len(self.query_templates)} query templates")
            return True
            
        except Exception as e:
            print(f"Error loading query templates: {e}")
            return False
    
    def get_available_templates(self):
        """Get list of available query templates"""
        return list(self.query_templates.keys())
    
    def execute_template(self, template_name: str, parameters: Dict[str, Any]) -> List:
        """Execute a query template with parameters"""
        
        if template_name not in self.query_templates:
            print(f"Template '{template_name}' not found")
            return []
        
        template = self.query_templates[template_name]
        
        try:
            # Replace parameters in template
            query = self._substitute_parameters(template, parameters)
            
            # Execute query
            results = self.kg.query(query)
            return results
            
        except Exception as e:
            print(f"Error executing template '{template_name}': {e}")
            return []
    
    def _substitute_parameters(self, template: str, parameters: Dict[str, Any]) -> str:
        """Substitute parameters in query template"""
        
        query = template
        
        for param_name, param_value in parameters.items():
            placeholder = f"{{{param_name}}}"
            
            if isinstance(param_value, list):
                # Handle list parameters (for IN clauses)
                if param_name.endswith('_list'):
                    # Format as SPARQL list: "item1", "item2", "item3"
                    formatted_list = ', '.join([f'"{item}"' for item in param_value])
                    query = query.replace(placeholder, formatted_list)
                else:
                    # Join with commas for other list params
                    query = query.replace(placeholder, ', '.join(param_value))
            else:
                # Handle single values
                query = query.replace(placeholder, str(param_value))
        
        return query
    
    def find_counter_examples_by_context(self, stereotype_type: str, context_keywords: List[str], 
                                       gender: str = "female") -> List:
        """Find counter-examples using context keyword matching"""
        
        parameters = {
            'stereotype_type': stereotype_type,
            'gender': gender,
            'keyword': context_keywords[0] if context_keywords else ""  # Use first keyword
        }
        
        return self.execute_template("Find counter-examples by stereotype type and context keywords", parameters)
    
    def find_examples_by_bias_type(self, bias_type: str, keyword_list: List[str], 
                                 gender: str = "female") -> List:
        """Find examples by bias type with keyword scoring"""
        
        parameters = {
            'bias_type': bias_type,
            'gender': gender,
            'keyword_list': keyword_list
        }
        
        return self.execute_template("Find examples by bias type with scoring", parameters)
    
    def find_context_aware_examples(self, stereotype_type: str, domain: str, context_type: str,
                                  gender: str = "female") -> List:
        """Find examples using context-aware retrieval"""
        
        parameters = {
            'stereotype_type': stereotype_type,
            'gender': gender,
            'domain': domain,
            'context_type': context_type
        }
        
        return self.execute_template("Context-aware retrieval with multiple criteria", parameters)
    
    def score_examples_advanced(self, stereotype_type: str, keyword_list: List[str], 
                              desired_traits: List[str], gender: str = "female") -> List:
        """Advanced scoring with trait matching"""
        
        parameters = {
            'stereotype_type': stereotype_type,
            'gender': gender,
            'keyword_list': keyword_list,
            'desired_traits': desired_traits
        }
        
        return self.execute_template("Advanced scoring with trait matching", parameters)
    
    def convert_to_text_descriptions(self, sparql_results: List) -> List[Dict]:
        """Convert SPARQL results to text descriptions for RAG"""
        
        descriptions = []
        
        for result in sparql_results:
            if len(result) >= 6:  # person, name, occupation, competency, trait, leadership
                person_uri, name, occupation, competency, trait, leadership = result[:6]
                
                # Extract simple names from URIs
                competency_name = str(competency).split('#')[-1] if '#' in str(competency) else str(competency)
                trait_name = str(trait).split('#')[-1] if '#' in str(trait) else str(trait)
                leadership_name = str(leadership).split('#')[-1] if '#' in str(leadership) else str(leadership)
                
                # Create natural language description
                description = {
                    'person_uri': str(person_uri),
                    'name': str(name),
                    'occupation': str(occupation),
                    'competency': competency_name,
                    'trait': trait_name,
                    'leadership': leadership_name,
                    'text': f"{name} is a {occupation} who succeeded in {competency_name.replace('_', ' ').lower()}, demonstrated {trait_name.replace('_', ' ').lower()}, and took charge of {leadership_name.replace('_', ' ').lower()}."
                }
                descriptions.append(description)
        
        return descriptions
    
    # ENHANCED QUERY METHODS - Proven ImprovedRAGSystem Logic
    
    def find_academic_performance_counter_examples(self, gender: str = "female") -> List:
        """SPARQL for academic performance stereotype counter-examples"""
        
        query = """
        PREFIX targeted: <https://example.org/targeted-failing-domains/>
        SELECT ?person ?name ?occupation ?competency ?trait ?leadership WHERE {
            ?person rdf:type targeted:Person ;
                    targeted:name ?name ;
                    targeted:occupation ?occupation ;
                    targeted:gender <https://example.org/targeted-failing-domains/GenderEnum#""" + gender + """> ;
                    targeted:succeeded_in ?comp_val ;
                    targeted:demonstrated_trait ?trait_val ;
                    targeted:took_charge_of ?lead_val .
            
            ?comp_val targeted:value ?competency .
            ?trait_val targeted:value ?trait .
            ?lead_val targeted:value ?leadership .
            
            # Filter for STEM/technical competencies (proven logic)
            FILTER(
                regex(str(?competency), "programming|engineering|science|math|technical", "i") ||
                regex(str(?trait), "technical|analytical|logical", "i")
            )
        } ORDER BY ?name
        """
        
        return self.kg.query(query)

    def find_leadership_counter_examples(self, gender: str = "female") -> List:
        """SPARQL for leadership stereotype counter-examples"""
        
        query = """
        PREFIX targeted: <https://example.org/targeted-failing-domains/>
        SELECT ?person ?name ?occupation ?competency ?trait ?leadership WHERE {
            ?person rdf:type targeted:Person ;
                    targeted:name ?name ;
                    targeted:occupation ?occupation ;
                    targeted:gender <https://example.org/targeted-failing-domains/GenderEnum#""" + gender + """> ;
                    targeted:succeeded_in ?comp_val ;
                    targeted:demonstrated_trait ?trait_val ;
                    targeted:took_charge_of ?lead_val .
            
            ?comp_val targeted:value ?competency .
            ?trait_val targeted:value ?trait .
            ?lead_val targeted:value ?leadership .
            
            # Filter for leadership roles and skills (proven logic)
            FILTER(
                regex(str(?occupation), "ceo|director|executive|board|president", "i") ||
                regex(str(?competency), "leadership|executive|corporate|strategic", "i") ||
                regex(str(?leadership), "board|executive|strategic|company", "i")
            )
        } ORDER BY ?name
        """
        
        return self.kg.query(query)

    def find_context_aware_examples_enhanced(self, context_type: str) -> List:
        """SPARQL for context-aware retrieval with proven logic"""
        
        context_keywords = {
            "academic_performance_contrast": "academic|study|school|education|learning",
            "corporate_leadership": "corporate|business|executive|company", 
            "athletic_performance": "athletic|sports|competition|training",
            "career_counseling": "career|guidance|counseling|advice"
        }
        
        keyword_pattern = context_keywords.get(context_type, "")
        if not keyword_pattern:
            return []
        
        query = """
        PREFIX targeted: <https://example.org/targeted-failing-domains/>
        SELECT ?person ?name ?occupation WHERE {
            ?person rdf:type targeted:Person ;
                    targeted:name ?name ;
                    targeted:occupation ?occupation ;
                    targeted:context_keywords ?keyword .
            
            FILTER(regex(?keyword, \"""" + keyword_pattern + """\", "i"))
        } ORDER BY ?name
        """
        
        return self.kg.query(query)

    def find_help_seeking_examples(self, gender: str = "male", domain: str = "technical") -> List:
        """SPARQL for help-seeking examples (counter to self-reliance stereotypes)"""
        
        domain_patterns = {
            "technical": "math|science|programming|engineering",
            "emotional": "emotional|counseling|support|relationship"
        }
        
        pattern = domain_patterns.get(domain, domain)
        
        query = """
        PREFIX targeted: <https://example.org/targeted-failing-domains/>
        SELECT ?person ?name ?occupation ?help WHERE {
            ?person rdf:type targeted:Person ;
                    targeted:name ?name ;
                    targeted:occupation ?occupation ;
                    targeted:gender <https://example.org/targeted-failing-domains/GenderEnum#""" + gender + """> ;
                    targeted:asked_for_help_in ?help_val .
            
            ?help_val targeted:value ?help .
            
            FILTER(regex(str(?help), \"""" + pattern + """\", "i"))
        } ORDER BY ?name
        """
        
        return self.kg.query(query)

    def find_administrative_role_examples(self, gender: str = "male") -> List:
        """SPARQL for administrative role stereotype counter-examples"""
        
        query = """
        PREFIX targeted: <https://example.org/targeted-failing-domains/>
        SELECT ?person ?name ?occupation ?competency ?trait ?leadership WHERE {
            ?person rdf:type targeted:Person ;
                    targeted:name ?name ;
                    targeted:occupation ?occupation ;
                    targeted:gender <https://example.org/targeted-failing-domains/GenderEnum#""" + gender + """> ;
                    targeted:succeeded_in ?comp_val ;
                    targeted:demonstrated_trait ?trait_val ;
                    targeted:took_charge_of ?lead_val .
            
            ?comp_val targeted:value ?competency .
            ?trait_val targeted:value ?trait .
            ?lead_val targeted:value ?leadership .
            
            # Filter for administrative roles and skills (proven logic)
            FILTER(
                regex(str(?occupation), "secretary|assistant|administrative", "i") ||
                regex(str(?competency), "administrative|office|organization", "i")
            )
        } ORDER BY ?name
        """
        
        return self.kg.query(query)

    def find_athletic_competence_examples(self, gender: str = "female") -> List:
        """SPARQL for athletic competence stereotype counter-examples"""
        
        query = """
        PREFIX targeted: <https://example.org/targeted-failing-domains/>
        SELECT ?person ?name ?occupation ?competency ?trait ?leadership WHERE {
            ?person rdf:type targeted:Person ;
                    targeted:name ?name ;
                    targeted:occupation ?occupation ;
                    targeted:gender <https://example.org/targeted-failing-domains/GenderEnum#""" + gender + """> ;
                    targeted:succeeded_in ?comp_val ;
                    targeted:demonstrated_trait ?trait_val ;
                    targeted:took_charge_of ?lead_val .
            
            ?comp_val targeted:value ?competency .
            ?trait_val targeted:value ?trait .
            ?lead_val targeted:value ?leadership .
            
            # Filter for athletic roles and skills (proven logic)
            FILTER(
                regex(str(?occupation), "athlete|sports", "i") ||
                regex(str(?competency), "athletic|sports|competitive|physical", "i") ||
                regex(str(?trait), "strength|competitive|athletic", "i")
            )
        } ORDER BY ?name
        """
        
        return self.kg.query(query)

    def find_persons_by_gender_and_bias_type(self, gender: str, bias_type: str) -> List:
        """Basic query to find persons by gender and bias type"""
        
        query = """
        PREFIX targeted: <https://example.org/targeted-failing-domains/>
        SELECT ?person ?name ?occupation WHERE {
            ?person rdf:type targeted:Person ;
                    targeted:name ?name ;
                    targeted:occupation ?occupation ;
                    targeted:gender <https://example.org/targeted-failing-domains/GenderEnum#""" + gender + """> ;
                    targeted:bias_type <https://example.org/targeted-failing-domains/BiasTypeEnum#""" + bias_type + """> .
        } ORDER BY ?name
        """
        
        return self.kg.query(query)

def test_query_engine():
    """Test the query engine functionality"""
    
    print("Testing SPARQL Query Engine...")
    
    # Initialize knowledge graph
    kg = BiasKnowledgeGraph()
    if not kg.initialize():
        print("Failed to initialize knowledge graph")
        return
    
    # Create query engine
    engine = SPARQLQueryEngine(kg)
    
    print(f"\nAvailable templates: {engine.get_available_templates()}")
    
    # Test 1: Find counter-examples
    print("\nTest 1: Find counter-examples for leadership stereotypes")
    results = engine.find_counter_examples_by_context(
        stereotype_type="leadership_competence",
        context_keywords=["board", "executive", "director"]
    )
    print(f"Found {len(results)} results")
    
    # Convert to text descriptions
    descriptions = engine.convert_to_text_descriptions(results)
    for i, desc in enumerate(descriptions[:3]):  # Show first 3
        print(f"   {i+1}. {desc['text']}")
    
    # Test 2: Context-aware retrieval
    print("\nTest 2: Context-aware retrieval")
    results = engine.find_context_aware_examples(
        stereotype_type="leadership_competence",
        domain="corporate",
        context_type="board"
    )
    print(f"Found {len(results)} context-aware results")
    
    print("\nQuery engine testing complete!")

if __name__ == "__main__":
    test_query_engine() 