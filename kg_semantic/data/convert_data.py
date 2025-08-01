#!/usr/bin/env python3
"""
Convert LinkML YAML data to RDF Turtle format
"""

import yaml
import json
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD
from pathlib import Path
import os

# Define namespaces
TARGETED = Namespace("https://example.org/targeted-failing-domains/")
BIAS = Namespace("https://example.org/bias-mitigation/")

def load_yaml_data(yaml_file):
    """Load and parse YAML data file"""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data

def create_person_triples(graph, person_data):
    """Convert a person object to RDF triples"""
    
    # Create person URI
    person_id = person_data['id']
    person_uri = URIRef(f"https://example.org/targeted-failing-domains/{person_id}")
    
    # Basic person properties
    graph.add((person_uri, RDF.type, TARGETED.Person))
    graph.add((person_uri, TARGETED.id, Literal(person_id)))
    graph.add((person_uri, TARGETED.name, Literal(person_data['name'])))
    graph.add((person_uri, TARGETED.occupation, Literal(person_data['occupation'])))
    
    # Gender (as object property)
    gender_uri = URIRef(f"https://example.org/targeted-failing-domains/GenderEnum#{person_data['gender']}")
    graph.add((person_uri, TARGETED.gender, gender_uri))
    
    # Bias type
    bias_type_uri = URIRef(f"https://example.org/targeted-failing-domains/BiasTypeEnum#{person_data['bias_type']}")
    graph.add((person_uri, TARGETED.bias_type, bias_type_uri))
    
    # Source
    source_uri = URIRef(f"https://example.org/targeted-failing-domains/SourceEnum#{person_data['source']}")
    graph.add((person_uri, TARGETED.source, source_uri))
    
    # Complex value objects
    if 'succeeded_in' in person_data and person_data['succeeded_in']:
        comp_value = BNode()
        graph.add((person_uri, TARGETED.succeeded_in, comp_value))
        graph.add((comp_value, RDF.type, TARGETED.CompetencyValue))
        comp_uri = URIRef(f"https://example.org/targeted-failing-domains/CompetencyEnum#{person_data['succeeded_in']['value']}")
        graph.add((comp_value, TARGETED.value, comp_uri))
    
    if 'failed_in' in person_data and person_data['failed_in']:
        fail_value = BNode()
        graph.add((person_uri, TARGETED.failed_in, fail_value))
        graph.add((fail_value, RDF.type, TARGETED.FailureValue))
        fail_uri = URIRef(f"https://example.org/targeted-failing-domains/FailureEnum#{person_data['failed_in']['value']}")
        graph.add((fail_value, TARGETED.value, fail_uri))
    
    if 'demonstrated_trait' in person_data and person_data['demonstrated_trait']:
        trait_value = BNode()
        graph.add((person_uri, TARGETED.demonstrated_trait, trait_value))
        graph.add((trait_value, RDF.type, TARGETED.TraitValue))
        trait_uri = URIRef(f"https://example.org/targeted-failing-domains/TraitEnum#{person_data['demonstrated_trait']['value']}")
        graph.add((trait_value, TARGETED.value, trait_uri))
    
    if 'afraid_of' in person_data and person_data['afraid_of']:
        fear_value = BNode()
        graph.add((person_uri, TARGETED.afraid_of, fear_value))
        graph.add((fear_value, RDF.type, TARGETED.FearValue))
        fear_uri = URIRef(f"https://example.org/targeted-failing-domains/FearEnum#{person_data['afraid_of']['value']}")
        graph.add((fear_value, TARGETED.value, fear_uri))
    
    if 'took_charge_of' in person_data and person_data['took_charge_of']:
        leadership_value = BNode()
        graph.add((person_uri, TARGETED.took_charge_of, leadership_value))
        graph.add((leadership_value, RDF.type, TARGETED.LeadershipValue))
        leadership_uri = URIRef(f"https://example.org/targeted-failing-domains/LeadershipEnum#{person_data['took_charge_of']['value']}")
        graph.add((leadership_value, TARGETED.value, leadership_uri))
    
    # Help (optional)
    if 'asked_for_help_in' in person_data and person_data['asked_for_help_in'] and person_data['asked_for_help_in'] != {}:
        if 'value' in person_data['asked_for_help_in']:
            help_value = BNode()
            graph.add((person_uri, TARGETED.asked_for_help_in, help_value))
            graph.add((help_value, RDF.type, TARGETED.HelpValue))
            help_uri = URIRef(f"https://example.org/targeted-failing-domains/HelpEnum#{person_data['asked_for_help_in']['value']}")
            graph.add((help_value, TARGETED.value, help_uri))
    
    # Context keywords (multiple values)
    if 'context_keywords' in person_data:
        for keyword in person_data['context_keywords']:
            graph.add((person_uri, TARGETED.context_keywords, Literal(keyword)))
    
    return person_uri

def convert_yaml_to_rdf(yaml_file, output_file):
    """Main conversion function"""
    
    print(f"Converting {yaml_file} to RDF...")
    
    # Load YAML data
    try:
        data = load_yaml_data(yaml_file)
        print(f"Loaded YAML data with {len(data.get('persons', []))} persons")
    except Exception as e:
        print(f"Error loading YAML data: {e}")
        return False
    
    # Create RDF graph
    graph = Graph()
    
    # Bind namespaces for prettier output
    graph.bind("targeted", TARGETED)
    graph.bind("bias", BIAS)
    
    # Convert each person
    person_uris = []
    for person_data in data.get('persons', []):
        try:
            person_uri = create_person_triples(graph, person_data)
            person_uris.append(person_uri)
        except Exception as e:
            print(f"❌ Error converting person {person_data.get('id', 'unknown')}: {e}")
            continue
    
    print(f"Converted {len(person_uris)} persons to RDF")
    print(f"Total triples: {len(graph)}")
    
    # Save to file
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Serialize to Turtle format
        graph.serialize(destination=output_file, format='turtle')
        print(f"RDF data saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
        
        return True
        
    except Exception as e:
        print(f"Error saving RDF data: {e}")
        return False

def validate_rdf_output(output_file):
    """Validate the generated RDF file"""
    
    if not os.path.exists(output_file):
        print(f"❌ RDF file not found: {output_file}")
        return False
    
    try:
        # Load and validate
        test_graph = Graph()
        test_graph.parse(output_file, format='turtle')
        
        print(f"✅ RDF file validation successful")
        print(f"📊 Total triples loaded: {len(test_graph)}")
        
        # Show some sample triples
        print("\n🔍 Sample triples:")
        for i, (s, p, o) in enumerate(test_graph):
            if i < 5:  # Show first 5 triples
                print(f"   {i+1}. {s} {p} {o}")
            else:
                break
        
        # Count persons
        person_count = len(list(test_graph.subjects(RDF.type, TARGETED.Person)))
        print(f"\n👥 Persons in RDF: {person_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ RDF validation error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 YAML to RDF Conversion Pipeline")
    print("=" * 50)
    
    # Use enhanced data
    yaml_file = "custom_kg/enhanced_linkml_data.yaml"
    output_file = "kg_semantic/data/enhanced_persons.ttl"
    
    # Convert YAML to RDF
    success = convert_yaml_to_rdf(yaml_file, output_file)
    
    if success:
        # Validate the generated RDF
        validate_rdf_output(output_file)
    
    print("\n🎉 Enhanced data conversion complete!") 