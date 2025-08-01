#!/usr/bin/env python3
"""
Convert LinkML schema to OWL ontology
"""

import os
import subprocess
import sys
from pathlib import Path

def convert_linkml_to_owl():
    """Convert LinkML schema to OWL ontology using linkml-convert"""
    
    # Paths
    schema_file = "custom_kg/linkml_schema.yaml"
    output_file = "kg_semantic/ontology/bias_mitigation_ontology.owl"
    
    # Ensure schema file exists
    if not os.path.exists(schema_file):
        print(f"Error: Schema file not found: {schema_file}")
        return False
    
    try:
        # Use gen-owl to generate OWL from LinkML schema
        cmd = [
            "gen-owl", 
            "--output", output_file,
            "--format", "owl",
            schema_file
        ]
        
        print(f"Converting {schema_file} to OWL...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print(f"Successfully created OWL ontology: {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting to OWL:")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def validate_owl_ontology():
    """Basic validation of the generated OWL ontology"""
    
    try:
        from rdflib import Graph
        
        owl_file = "kg_semantic/ontology/bias_mitigation_ontology.owl"
        
        if not os.path.exists(owl_file):
            print(f" OWL file not found: {owl_file}")
            return False
        
        # Load and parse the OWL file
        g = Graph()
        g.parse(owl_file, format="xml")
        
        print(f" OWL ontology validated successfully")
        print(f" Total triples: {len(g)}")
        
        # Print some basic statistics
        from rdflib import RDF, RDFS, OWL
        
        classes = list(g.subjects(RDF.type, RDFS.Class))
        properties = list(g.subjects(RDF.type, RDF.Property))
        
        print(f" Classes found: {len(classes)}")
        print(f" Properties found: {len(properties)}")
        
        # Show first few classes
        print("\n Sample Classes:")
        for i, cls in enumerate(classes[:5]):
            print(f"   {i+1}. {cls}")
        
        return True
        
    except Exception as e:
        print(f" Error validating OWL ontology: {e}")
        return False

if __name__ == "__main__":
    print("LinkML to OWL Conversion Pipeline")
    print("=" * 50)
    
    # Step 1: Convert schema
    if convert_linkml_to_owl():
        # Step 2: Validate result
        validate_owl_ontology()
    else:
        print(" Conversion failed, skipping validation")
        sys.exit(1)
    
    print("\n Ontology generation complete!") 