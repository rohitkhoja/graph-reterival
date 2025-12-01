#!/usr/bin/env python3

import json
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Any, Set
import statistics
import re

def analyze_mag_dataset(file_path: str, sample_size: int = 1000):
    
    
    type_counts = Counter()
    field_stats = defaultdict(list)
    string_lengths = defaultdict(list)
    numeric_stats = defaultdict(list)
    
    type_samples = defaultdict(list)
    
    all_fields = set()
    field_frequency = Counter()
    
    string_fields = set()
    
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_objects = len(data)
        
        for obj_id, obj_data in data.items():
            if not isinstance(obj_data, dict):
                continue
                
            obj_type = obj_data.get('type', 'unknown')
            type_counts[obj_type] += 1
            
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
                else:
                    field_stats[field_name].append(type(field_value).__name__)
        
        print_analysis_results(
            type_counts, all_fields, field_frequency, 
            string_lengths, numeric_stats, type_samples
        )
        
    except Exception as e:
        return

def print_analysis_results(type_counts, all_fields, field_frequency, 
                          string_lengths, numeric_stats, type_samples):
    
    total_objects = sum(type_counts.values())
    for obj_type, count in type_counts.most_common():
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
    
    
    for obj_type, samples in type_samples.items():
        if not samples:
            continue
            
        
        if samples:
            sample_obj = samples[0][1]
            for field_name, field_value in sample_obj.items():
                field_type = type(field_value).__name__
                if isinstance(field_value, str):
                    preview = field_value[:50] + "..." if len(field_value) > 50 else field_value
                elif isinstance(field_value, list):
                else:
        
        for i, (obj_id, obj_data) in enumerate(samples[:3]):
            for field_name, field_value in obj_data.items():
                if isinstance(field_value, str) and len(field_value) > 100:
                    preview = field_value[:100] + "..."
                else:
                    preview = field_value

def analyze_citations_and_relationships(file_path: str):
    
    
    citation_fields = []
    relationship_fields = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sample_count = 0
            for line in f:
                if sample_count >= 1000:
                    break
                
                try:
                    if line.strip().startswith('{'):
                        obj_data = json.loads(line.strip().rstrip(','))
                        
                        for field_name, field_value in obj_data.items():
                            if 'citation' in field_name.lower():
                                citation_fields.append(field_name)
                            elif any(rel_word in field_name.lower() for rel_word in ['author', 'reference', 'cited', 'citing']):
                                relationship_fields.append(field_name)
                        
                        sample_count += 1
                except:
                    continue
        
        
    except Exception as e:

def main():
    
    file_path = ""
    
    analyze_mag_dataset(file_path, sample_size=500)
    
    analyze_citations_and_relationships(file_path)

if __name__ == "__main__":
    main()
