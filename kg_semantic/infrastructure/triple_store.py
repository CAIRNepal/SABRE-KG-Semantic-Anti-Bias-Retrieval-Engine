#!/usr/bin/env python3
"""
Triple Store Management with RDFLib
"""

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS
import os
from pathlib import Path

class BiasKnowledgeGraph:
    """Knowledge Graph for bias mitigation data"""
    
    def __init__(self):
        self.graph = Graph()
        self.namespaces = self._setup_namespaces()
        self._bind_namespaces()
        
        # File paths
        self.ontology_file = "kg_semantic/ontology/bias_mitigation_ontology.owl"
        self.data_file = "kg_semantic/data/persons.ttl"
        
    def _setup_namespaces(self):
        """Define namespaces used in the KG"""
        return {
            'targeted': Namespace("https://example.org/targeted-failing-domains/"),
            'bias': Namespace("https://example.org/bias-mitigation/"),
            'rdf': RDF,
            'rdfs': RDFS
        }
    
    def _bind_namespaces(self):
        """Bind namespaces to the graph for prettier SPARQL queries"""
        for prefix, namespace in self.namespaces.items():
            self.graph.bind(prefix, namespace)
    
    def load_ontology(self):
        """Load the OWL ontology"""
        if not os.path.exists(self.ontology_file):
            print(f"❌ Ontology file not found: {self.ontology_file}")
            return False
        
        try:
            self.graph.parse(self.ontology_file, format='turtle')
            print(f"✅ Loaded ontology from {self.ontology_file}")
            return True
        except Exception as e:
            print(f"❌ Error loading ontology: {e}")
            return False
    
    def load_data(self):
        """Load RDF data from turtle file"""
        
        data_file = "kg_semantic/data/enhanced_persons.ttl"
        
        if not os.path.exists(data_file):
            print(f"Data file not found: {data_file}")
            return False
        
        try:
            # Load the turtle file
            self.graph.parse(data_file, format='turtle')
            print(f"Loaded data from {data_file}")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def initialize(self):
        """Initialize the complete knowledge graph"""
        print("Initializing Bias Knowledge Graph...")
        
        # Load ontology
        ontology_loaded = self.load_ontology()
        
        # Load data  
        data_loaded = self.load_data()
        
        if ontology_loaded and data_loaded:
            print(f"Knowledge Graph initialized successfully")
            print(f"Total triples: {len(self.graph)}")
            self._print_statistics()
            return True
        else:
            print("Failed to initialize Knowledge Graph")
            return False
    
    def _print_statistics(self):
        """Print KG statistics"""
        
        # Count persons
        person_query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX targeted: <https://example.org/targeted-failing-domains/>
        
        SELECT (COUNT(?person) AS ?count) WHERE {
            ?person rdf:type targeted:Person .
        }
        """
        result = list(self.graph.query(person_query))
        person_count = result[0][0] if result else 0
        
        # Count by gender
        gender_query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX targeted: <https://example.org/targeted-failing-domains/>
        
        SELECT ?gender (COUNT(?person) AS ?count) WHERE {
            ?person rdf:type targeted:Person ;
                     targeted:gender ?gender .
        } GROUP BY ?gender
        """
        gender_results = list(self.graph.query(gender_query))
        
        print(f"\nKnowledge Graph Statistics:")
        print(f"   Total persons: {person_count}")
        print(f"   Gender distribution:")
        for gender, count in gender_results:
            gender_name = str(gender).split('#')[-1] if '#' in str(gender) else str(gender)
            print(f"      - {gender_name}: {count}")
    
    def query(self, sparql_query):
        """Execute a SPARQL query"""
        try:
            return list(self.graph.query(sparql_query))
        except Exception as e:
            print(f"❌ SPARQL query error: {e}")
            return []
    
    def find_persons_by_criteria(self, **criteria):
        """Find persons matching specific criteria"""
        
        # Build SPARQL query dynamically
        where_clauses = ["?person rdf:type targeted:Person"]
        
        if 'gender' in criteria:
            where_clauses.append(f"?person targeted:gender <https://example.org/targeted-failing-domains/GenderEnum#{criteria['gender']}>")
        
        if 'bias_type' in criteria:
            where_clauses.append(f"?person targeted:bias_type <https://example.org/targeted-failing-domains/BiasTypeEnum#{criteria['bias_type']}>")
        
        if 'occupation' in criteria:
            where_clauses.append(f'?person targeted:occupation "{criteria["occupation"]}"')
        
        if 'competency' in criteria:
            where_clauses.append(f"""?person targeted:succeeded_in ?comp_val .
                ?comp_val targeted:value <https://example.org/targeted-failing-domains/CompetencyEnum#{criteria['competency']}>""")
        
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX targeted: <https://example.org/targeted-failing-domains/>
        
        SELECT ?person ?name ?occupation WHERE {{
            {' . '.join(where_clauses)} .
            ?person targeted:name ?name ;
                    targeted:occupation ?occupation .
        }}
        """
        
        return self.query(query)
    
    def find_counter_examples(self, stereotype_type, gender="female"):
        """Find counter-stereotypical examples for bias mitigation"""
        
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX targeted: <https://example.org/targeted-failing-domains/>
        
        SELECT ?person ?name ?occupation ?competency ?trait ?leadership WHERE {{
            ?person rdf:type targeted:Person ;
                    targeted:gender <https://example.org/targeted-failing-domains/GenderEnum#{gender}> ;
                    targeted:bias_type <https://example.org/targeted-failing-domains/BiasTypeEnum#{stereotype_type}> ;
                    targeted:name ?name ;
                    targeted:occupation ?occupation .
            
            ?person targeted:succeeded_in ?comp_val .
            ?comp_val targeted:value ?competency .
            
            ?person targeted:demonstrated_trait ?trait_val .
            ?trait_val targeted:value ?trait .
            
            ?person targeted:took_charge_of ?leadership_val .
            ?leadership_val targeted:value ?leadership .
        }}
        """
        
        return self.query(query)
    
    def export_sample_queries(self):
        """Export some sample queries to test the system"""
        
        queries = {
            "all_persons": """
                SELECT ?person ?name ?gender ?occupation WHERE {
                    ?person rdf:type targeted:Person ;
                            targeted:name ?name ;
                            targeted:gender ?gender ;
                            targeted:occupation ?occupation .
                }
            """,
            
            "leadership_competent_females": """
                SELECT ?person ?name ?occupation ?competency WHERE {
                    ?person rdf:type targeted:Person ;
                            targeted:gender targeted:GenderEnum#female ;
                            targeted:bias_type targeted:BiasTypeEnum#leadership_competence ;
                            targeted:name ?name ;
                            targeted:occupation ?occupation .
                    
                    ?person targeted:succeeded_in ?comp_val .
                    ?comp_val targeted:value ?competency .
                }
            """,
            
            "context_keywords": """
                SELECT ?person ?name ?keyword WHERE {
                    ?person rdf:type targeted:Person ;
                            targeted:name ?name ;
                            targeted:context_keywords ?keyword .
                }
            """
        }
        
        # Save queries to file
        queries_file = "kg_semantic/queries/sample_queries.sparql"
        os.makedirs(os.path.dirname(queries_file), exist_ok=True)
        
        with open(queries_file, 'w') as f:
            for name, query in queries.items():
                f.write(f"# {name.replace('_', ' ').title()}\n")
                f.write(query.strip() + "\n\n")
        
        print(f"✅ Sample queries exported to: {queries_file}")
        return queries

if __name__ == "__main__":
    # Test the triple store
    kg = BiasKnowledgeGraph()
    
    if kg.initialize():
        print("\n🧪 Testing SPARQL queries...")
        
        # Test 1: Find all females in leadership roles
        results = kg.find_persons_by_criteria(gender="female", bias_type="leadership_competence")
        print(f"\n👩‍💼 Leadership competent females: {len(results)}")
        for result in results[:3]:  # Show first 3
            print(f"   - {result[1]} ({result[2]})")
        
        # Test 2: Find counter-examples
        counter_examples = kg.find_counter_examples("leadership_competence")
        print(f"\n🔄 Counter-examples for leadership: {len(counter_examples)}")
        
        # Test 3: Export sample queries
        kg.export_sample_queries()
        
        print("\n🎉 Triple store testing complete!")
    else:
        print("❌ Failed to initialize triple store") 