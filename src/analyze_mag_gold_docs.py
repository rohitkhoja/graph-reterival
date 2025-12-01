#!/usr/bin/env python3

import json
import pandas as pd
import ast
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any
import numpy as np

def load_mag_data():
    
    mag_file = ""
    
    with open(mag_file, 'r', encoding='utf-8') as f:
        mag_data = json.load(f)
    

    obj_id_to_info = {}
    type_counts = defaultdict(int)
    type_id_ranges = defaultdict(list)
    
    for obj_id, obj_data in mag_data.items():
        obj_type = obj_data.get('type')
        obj_id_int = int(obj_id)
        
        if obj_type == 'author':
            obj_id_to_info[obj_id_int] = {
                'type': 'author',
                'name': obj_data.get('DisplayName', ''),
                'institution': obj_data.get('institution', ''),
                'paper_count': obj_data.get('PaperCount', 0),
                'citation_count': obj_data.get('CitationCount', 0),
                'rank': obj_data.get('Rank', 0)
            }
            type_counts['author'] += 1
            type_id_ranges['author'].append(obj_id_int)
        
        elif obj_type == 'paper':
            obj_id_to_info[obj_id_int] = {
                'type': 'paper',
                'title': obj_data.get('title', ''),
                'year': obj_data.get('Year', 0),
                'doc_type': obj_data.get('DocType', ''),
                'citation_count': obj_data.get('PaperCitationCount', 0),
                'reference_count': obj_data.get('ReferenceCount', 0),
                'rank': obj_data.get('PaperRank', 0)
            }
            type_counts['paper'] += 1
            type_id_ranges['paper'].append(obj_id_int)
        
        elif obj_type == 'institution':
            obj_id_to_info[obj_id_int] = {
                'type': 'institution',
                'name': obj_data.get('DisplayName', ''),
                'paper_count': obj_data.get('PaperCount', 0),
                'citation_count': obj_data.get('CitationCount', 0),
                'rank': obj_data.get('Rank', 0)
            }
            type_counts['institution'] += 1
            type_id_ranges['institution'].append(obj_id_int)
        
        elif obj_type == 'field_of_study':
            obj_id_to_info[obj_id_int] = {
                'type': 'field_of_study',
                'name': obj_data.get('DisplayName', ''),
                'level': obj_data.get('Level', 0),
                'paper_count': obj_data.get('PaperCount', 0),
                'citation_count': obj_data.get('CitationCount', 0),
                'rank': obj_data.get('Rank', 0)
            }
            type_counts['field_of_study'] += 1
            type_id_ranges['field_of_study'].append(obj_id_int)
    

    id_ranges = {}
    for obj_type, ids in type_id_ranges.items():
        if ids:
            id_ranges[obj_type] = {
                'min': min(ids),
                'max': max(ids),
                'count': len(ids)
            }
    
    
    return obj_id_to_info, id_ranges

def analyze_csv_file(csv_file: str, obj_id_to_info: Dict, id_ranges: Dict):
    

    df = pd.read_csv(csv_file)
    

    gold_doc_stats = defaultdict(list)
    type_distribution = Counter()
    missing_ids = []
    obj_id_type_mapping = {}
    

    for idx, row in df.iterrows():
        gold_docs_str = row['gold_docs']
        query = row['query']
        
        try:

            gold_docs = ast.literal_eval(gold_docs_str)
            
            if not isinstance(gold_docs, list):
                continue
            

            query_types = []
            for obj_id in gold_docs:
                if obj_id in obj_id_to_info:
                    obj_info = obj_id_to_info[obj_id]
                    obj_type = obj_info['type']
                    query_types.append(obj_type)
                    type_distribution[obj_type] += 1
                    obj_id_type_mapping[obj_id] = obj_type
                else:
                    missing_ids.append(obj_id)
            

            if query_types:
                unique_types = list(set(query_types))
                gold_doc_stats['query_types'].append(unique_types)
                gold_doc_stats['num_gold_docs'].append(len(gold_docs))
                gold_doc_stats['num_unique_types'].append(len(unique_types))
                gold_doc_stats['is_mixed_types'].append(len(unique_types) > 1)
        
        except (ValueError, SyntaxError) as e:
            continue
    

    total_gold_docs = sum(type_distribution.values())
    missing_count = len(missing_ids)
    
    if total_gold_docs > 0:
    else:
    
    if total_gold_docs > 0:
        for obj_type, count in type_distribution.most_common():
            percentage = (count / total_gold_docs) * 100
    else:
    

    if gold_doc_stats['query_types']:
        

        type_combinations = Counter()
        for query_types in gold_doc_stats['query_types']:
            if len(query_types) == 1:
                type_combinations[f"Only {query_types[0]}"] += 1
            else:
                type_combinations[f"Mixed: {', '.join(sorted(query_types))}"] += 1
        
        for combo, count in type_combinations.most_common(10):
            percentage = (count / len(gold_doc_stats['query_types'])) * 100
    

    if missing_ids:
        

        missing_by_type = defaultdict(int)
        for obj_id in missing_ids:
            for obj_type, range_info in id_ranges.items():
                if range_info['min'] <= obj_id <= range_info['max']:
                    missing_by_type[obj_type] += 1
                    break
            else:
                missing_by_type['unknown'] += 1
        
        for obj_type, count in missing_by_type.items():
    
    return {
        'type_distribution': type_distribution,
        'missing_ids': missing_ids,
        'obj_id_type_mapping': obj_id_type_mapping,
        'query_stats': gold_doc_stats
    }

def analyze_id_ranges(id_ranges: Dict):
    
    for obj_type, range_info in id_ranges.items():

def main():
    

    obj_id_to_info, id_ranges = load_mag_data()
    

    analyze_id_ranges(id_ranges)
    

    csv_files = [
        "",
        ""
    ]
    
    all_results = {}
    
    for csv_file in csv_files:
        results = analyze_csv_file(csv_file, obj_id_to_info, id_ranges)
        all_results[csv_file.split('/')[-1]] = results
    

    
    combined_type_dist = Counter()
    combined_missing = []
    
    for results in all_results.values():
        combined_type_dist.update(results['type_distribution'])
        combined_missing.extend(results['missing_ids'])
    
    total_combined = sum(combined_type_dist.values())
    if total_combined > 0:
    else:
    
    for obj_type, count in combined_type_dist.most_common():
        percentage = (count / total_combined) * 100 if total_combined > 0 else 0
    

    output_file = ""
    
    analysis_summary = {
        'id_ranges': id_ranges,
        'combined_type_distribution': dict(combined_type_dist),
        'total_gold_docs': total_combined,
        'missing_ids_count': len(combined_missing),
        'coverage_percentage': ((total_combined - len(combined_missing)) / total_combined * 100) if total_combined > 0 else 0,
        'file_results': {}
    }
    
    for filename, results in all_results.items():
        analysis_summary['file_results'][filename] = {
            'type_distribution': dict(results['type_distribution']),
            'missing_ids_count': len(results['missing_ids']),
            'query_stats': {
                'avg_gold_docs': float(np.mean(results['query_stats']['num_gold_docs'])) if results['query_stats']['num_gold_docs'] else 0,
                'avg_unique_types': float(np.mean(results['query_stats']['num_unique_types'])) if results['query_stats']['num_unique_types'] else 0,
                'mixed_type_queries': sum(results['query_stats']['is_mixed_types'])
            }
        }
    
    with open(output_file, 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    

if __name__ == "__main__":
    main()
