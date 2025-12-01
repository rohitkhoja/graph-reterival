#!/usr/bin/env python3

import json
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Any, Set
import statistics
import re

def analyze_prime_dataset(file_path: str, sample_size: int = 1000):
    
    

    type_counts = Counter()
    field_stats = defaultdict(list)
    string_lengths = defaultdict(list)
    numeric_stats = defaultdict(list)
    

    type_samples = defaultdict(list)
    

    all_fields = set()
    field_frequency = Counter()
    

    string_fields = set()
    

    source_counts = Counter()
    detail_field_analysis = defaultdict(lambda: defaultdict(list))
    
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_objects = len(data)
        

        for obj_id, obj_data in data.items():
            if not isinstance(obj_data, dict):
                continue
                
            obj_type = obj_data.get('type', 'unknown')
            type_counts[obj_type] += 1
            

            source = obj_data.get('source', 'unknown')
            source_counts[source] += 1
            

            if len(type_samples[obj_type]) < sample_size:
                type_samples[obj_type].append((obj_id, obj_data))
            

            for field_name, field_value in obj_data.items():
                all_fields.add(field_name)
                field_frequency[field_name] += 1
                

                if isinstance(field_value, str):
                    string_fields.add(field_name)
                    string_lengths[field_name].append(len(field_value))
                    field_stats[field_name].append(field_value)
                elif isinstance(field_value, (int, float)):
                    numeric_stats[field_name].append(field_value)
                elif isinstance(field_value, list):
                    field_stats[field_name].append(len(field_value))
                elif isinstance(field_value, dict):

                    if field_name == 'details':
                        for detail_key, detail_value in field_value.items():
                            detail_field_analysis[obj_type][detail_key].append(detail_value)
                else:
                    field_stats[field_name].append(type(field_value).__name__)
        

        print_analysis_results(
            type_counts, all_fields, field_frequency, 
            string_lengths, numeric_stats, type_samples,
            source_counts, detail_field_analysis
        )
        
    except Exception as e:
        return

def print_analysis_results(type_counts, all_fields, field_frequency, 
                          string_lengths, numeric_stats, type_samples,
                          source_counts, detail_field_analysis):
    
    total_objects = sum(type_counts.values())
    for obj_type, count in type_counts.most_common():
        percentage = (count / total_objects) * 100
    
    
    for source, count in source_counts.most_common():
        percentage = (count / total_objects) * 100
    
    for field, count in field_frequency.most_common(15):
        percentage = (count / total_objects) * 100
    
    for field in sorted(string_lengths.keys()):
        lengths = string_lengths[field]
        if lengths:
            avg_length = statistics.mean(lengths)
            median_length = statistics.median(lengths)
            max_length = max(lengths)
            min_length = min(lengths)
    
    for field in sorted(numeric_stats.keys()):
        values = numeric_stats[field]
        if values:
            avg_val = statistics.mean(values)
            median_val = statistics.median(values)
            max_val = max(values)
            min_val = min(values)
    
    for obj_type, detail_fields in detail_field_analysis.items():
        for detail_field, values in detail_fields.items():
            if values:

                non_empty = [v for v in values if v and str(v).strip()]
                if non_empty:
                    

                    if isinstance(non_empty[0], str) and len(non_empty[0]) < 100:
    
    

    object_type_order = [
        'gene/protein', 'disease', 'drug', 'pathway', 'anatomy',
        'biological_process', 'cellular_component', 'molecular_function',
        'effect/phenotype', 'exposure'
    ]
    
    for obj_type in object_type_order:
        if obj_type in type_samples and type_samples[obj_type]:
            samples = type_samples[obj_type]
            

            if samples:
                sample_obj = samples[0][1]
                
                for field_name, field_value in sample_obj.items():
                    if field_name == 'details' and isinstance(field_value, dict):
                        for detail_key, detail_val in field_value.items():
                            if isinstance(detail_val, str):
                                preview = detail_val[:80] + "..." if len(detail_val) > 80 else detail_val
                            elif isinstance(detail_val, list):
                                if detail_val and isinstance(detail_val[0], str):
                                else:
                            elif isinstance(detail_val, dict):
                            else:
                    elif isinstance(field_value, str):
                        preview = field_value[:60] + "..." if len(field_value) > 60 else field_value
                    elif isinstance(field_value, list):
                    else:
                

                for i, (obj_id, obj_data) in enumerate(samples[1:3]):
                    if 'details' in obj_data:
                        details = obj_data['details']
                        key_details = []
                        for key in ['description', 'summary', 'indication', 'definition', 'summation']:
                            if key in details and isinstance(details[key], str):
                                key_details.append(f"{key}: {details[key][:40]}...")
                                break
                        if key_details:
    
    analyze_entity_relationships(type_samples)

def analyze_entity_relationships(type_samples):
    
    

    common_fields = defaultdict(set)
    entity_specific_fields = defaultdict(set)
    
    for obj_type, samples in type_samples.items():
        if not samples:
            continue
            
        sample_obj = samples[0][1]
        

        all_fields = set(sample_obj.keys())
        if 'details' in sample_obj and isinstance(sample_obj['details'], dict):
            all_fields.update(sample_obj['details'].keys())
        
        entity_specific_fields[obj_type] = all_fields
        

        for field in all_fields:
            common_fields[field].add(obj_type)
    
    for field, entity_types in sorted(common_fields.items()):
        if len(entity_types) > 1:
    
    

    relationships = {
        'gene-drug': [],
        'disease-gene': [],
        'pathway-gene': [],
        'drug-pathway': [],
        'disease-drug': [],
        'anatomy-gene': [],
        'process-function': []
    }
    
    for obj_type, samples in type_samples.items():
        if not samples:
            continue
            
        for obj_id, obj_data in samples[:10]:
            details = obj_data.get('details', {})
            

            text_fields = []
            for field, value in details.items():
                if isinstance(value, str):
                    text_fields.append(value.lower())
            
            combined_text = ' '.join(text_fields)
            

            if obj_type == 'drug':
                if any(keyword in combined_text for keyword in ['gene', 'protein', 'target']):
                    relationships['gene-drug'].append(obj_id)
                if any(keyword in combined_text for keyword in ['pathway', 'metabolism']):
                    relationships['drug-pathway'].append(obj_id)
                if any(keyword in combined_text for keyword in ['disease', 'treatment', 'indication']):
                    relationships['disease-drug'].append(obj_id)
            
            elif obj_type == 'disease':
                if any(keyword in combined_text for keyword in ['gene', 'mutation', 'genetic']):
                    relationships['disease-gene'].append(obj_id)
            
            elif obj_type == 'gene/protein':
                if any(keyword in combined_text for keyword in ['pathway', 'signaling']):
                    relationships['pathway-gene'].append(obj_id)
                if any(keyword in combined_text for keyword in ['tissue', 'organ', 'anatomy']):
                    relationships['anatomy-gene'].append(obj_id)
            
            elif obj_type in ['biological_process', 'molecular_function']:
                if any(keyword in combined_text for keyword in ['function', 'process', 'activity']):
                    relationships['process-function'].append(obj_id)
    

    for rel_type, entity_list in relationships.items():
        if entity_list:
            unique_count = len(set(entity_list))
    
    

    domains = {
        'Genomics': ['gene/protein', 'biological_process', 'molecular_function', 'cellular_component'],
        'Clinical': ['disease', 'drug', 'effect/phenotype'],
        'Systems': ['pathway', 'anatomy'],
        'Environmental': ['exposure']
    }
    
    for domain, entity_types in domains.items():
        domain_entities = sum(len(type_samples.get(et, [])) for et in entity_types)
    
    

    indicators = {
        'Genomic Information': 0,
        'Clinical Information': 0,
        'Pathway Information': 0,
        'Anatomical Information': 0,
        'Pharmacological Information': 0
    }
    
    for obj_type, samples in type_samples.items():
        if not samples:
            continue
            
        for obj_id, obj_data in samples[:5]:
            details = obj_data.get('details', {})
            
            if obj_type == 'gene/protein':
                if 'genomic_pos' in details:
                    indicators['Genomic Information'] += 1
            
            if obj_type in ['disease', 'drug']:
                clinical_fields = ['indication', 'symptoms', 'treatment', 'mechanism_of_action']
                if any(field in details for field in clinical_fields):
                    indicators['Clinical Information'] += 1
            
            if obj_type == 'pathway':
                if 'summation' in details or 'goBiologicalProcess' in details:
                    indicators['Pathway Information'] += 1
            
            if obj_type == 'anatomy':
                indicators['Anatomical Information'] += 1
            
            if obj_type == 'drug':
                if any(field in details for field in ['molecular_weight', 'pharmacodynamics', 'protein_binding']):
                    indicators['Pharmacological Information'] += 1
    
    for indicator, count in indicators.items():
        if count > 0:

def analyze_biological_relationships(file_path: str):
    
    
    relationship_patterns = defaultdict(int)
    cross_type_relationships = defaultdict(int)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        

        type_pairs = []
        
        for obj_id, obj_data in data.items():
            if not isinstance(obj_data, dict):
                continue
                
            obj_type = obj_data.get('type', 'unknown')
            

            if 'details' in obj_data and isinstance(obj_data['details'], dict):
                details = obj_data['details']
                

                if 'pathway' in details:
                    relationship_patterns['pathway_relationship'] += 1
                

                if 'protein' in str(details).lower():
                    relationship_patterns['protein_relationship'] += 1
                

                if 'disease' in str(details).lower() or 'mondo' in str(details).lower():
                    relationship_patterns['disease_relationship'] += 1
        
        for pattern, count in relationship_patterns.items():
        

        genomic_entities = 0
        pathway_entities = 0
        
        for obj_id, obj_data in data.items():
            if obj_data.get('type') == 'gene/protein':
                if 'details' in obj_data and 'genomic_pos' in obj_data['details']:
                    genomic_entities += 1
            elif obj_data.get('type') == 'pathway':
                pathway_entities += 1
        
        
    except Exception as e:

def analyze_text_content(file_path: str):
    
    
    text_fields = ['name', 'summary', 'description', 'definition']
    field_stats = defaultdict(list)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for obj_id, obj_data in data.items():
            if not isinstance(obj_data, dict):
                continue
            

            for field in text_fields:
                if field in obj_data and isinstance(obj_data[field], str):
                    text = obj_data[field].strip()
                    if text:
                        field_stats[field].append(text)
            

            if 'details' in obj_data and isinstance(obj_data['details'], dict):
                details = obj_data['details']
                for field in text_fields:
                    if field in details and isinstance(details[field], str):
                        text = details[field].strip()
                        if text:
                            field_stats[f"details.{field}"].append(text)
        
        for field, texts in field_stats.items():
            if texts:
                lengths = [len(text) for text in texts]
                avg_length = statistics.mean(lengths)
                median_length = statistics.median(lengths)
                max_length = max(lengths)
                min_length = min(lengths)
        
    except Exception as e:

def main():
    
    file_path = ""
    

    analyze_prime_dataset(file_path, sample_size=500)
    

    analyze_biological_relationships(file_path)
    

    analyze_text_content(file_path)

if __name__ == "__main__":
    main()
