#!/usr/bin/env python3

import json
import pandas as pd
import ast
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any
import numpy as np

def load_prime_data():
    
    prime_file = ""
    
    with open(prime_file, 'r', encoding='utf-8') as f:
        prime_data = json.load(f)
    

    obj_id_to_info = {}
    type_counts = defaultdict(int)
    type_id_ranges = defaultdict(list)
    
    for obj_id, obj_data in prime_data.items():
        obj_type = obj_data.get('type')
        obj_id_int = int(obj_id)
        

        if obj_type in ['gene/protein', 'disease', 'drug', 'pathway', 'anatomy', 
                       'biological_process', 'cellular_component', 'molecular_function',
                       'effect/phenotype', 'exposure']:
            
            obj_id_to_info[obj_id_int] = {
                'type': obj_type,
                'name': obj_data.get('name', ''),
                'source': obj_data.get('source', ''),
                'id': obj_data.get('id', obj_id_int)
            }
            

            if obj_type == 'gene/protein':
                details = obj_data.get('details', {})
                obj_id_to_info[obj_id_int].update({
                    'full_name': details.get('name', ''),
                    'summary': details.get('summary', ''),
                    'has_genomic_pos': 'genomic_pos' in details,
                    'aliases': details.get('alias', [])
                })
            
            elif obj_type == 'disease':
                details = obj_data.get('details', {})
                obj_id_to_info[obj_id_int].update({
                    'mondo_name': details.get('mondo_name', ''),
                    'mondo_definition': details.get('mondo_definition', ''),
                })
            
            elif obj_type == 'drug':
                details = obj_data.get('details', {})
                obj_id_to_info[obj_id_int].update({
                    'description': details.get('description', ''),
                    'indication': details.get('indication', ''),
                    'mechanism_of_action': details.get('mechanism_of_action', ''),
                    'molecular_weight': details.get('molecular_weight', 0)
                })
            
            elif obj_type == 'pathway':
                details = obj_data.get('details', {})
                obj_id_to_info[obj_id_int].update({
                    'display_name': details.get('displayName', ''),
                    'summation': details.get('summation', ''),
                    'species_name': details.get('speciesName', ''),
                    'has_diagram': details.get('hasDiagram', False)
                })
            
            type_counts[obj_type] += 1
            type_id_ranges[obj_type].append(obj_id_int)
    

    id_ranges = {}
    for obj_type, ids in type_id_ranges.items():
        if ids:
            id_ranges[obj_type] = {
                'min': min(ids),
                'max': max(ids),
                'count': len(ids)
            }
    
    for obj_type in sorted(type_counts.keys()):
    
    return obj_id_to_info, id_ranges

def analyze_csv_file(csv_file: str, obj_id_to_info: Dict, id_ranges: Dict):
    

    df = pd.read_csv(csv_file)
    

    gold_doc_stats = defaultdict(list)
    type_distribution = Counter()
    missing_ids = []
    obj_id_type_mapping = {}
    

    sample_queries = []
    

    for idx, row in df.iterrows():
        gold_docs_str = row['gold_docs']
        query = row['query']
        query_id = row.get('query_id', idx)
        
        try:

            gold_docs = ast.literal_eval(gold_docs_str)
            
            if not isinstance(gold_docs, list):
                continue
            

            query_types = []
            query_gold_info = []
            
            for obj_id in gold_docs:
                if obj_id in obj_id_to_info:
                    obj_info = obj_id_to_info[obj_id]
                    obj_type = obj_info['type']
                    query_types.append(obj_type)
                    type_distribution[obj_type] += 1
                    obj_id_type_mapping[obj_id] = obj_type
                    

                    query_gold_info.append({
                        'id': obj_id,
                        'type': obj_type,
                        'name': obj_info.get('name', ''),
                    })
                else:
                    missing_ids.append(obj_id)
            

            if query_types:
                unique_types = list(set(query_types))
                gold_doc_stats['query_types'].append(unique_types)
                gold_doc_stats['num_gold_docs'].append(len(gold_docs))
                gold_doc_stats['num_unique_types'].append(len(unique_types))
                gold_doc_stats['is_mixed_types'].append(len(unique_types) > 1)
                

                if len(sample_queries) < 20:
                    sample_queries.append({
                        'query_id': query_id,
                        'query': query,
                        'gold_docs': gold_docs,
                        'gold_types': unique_types,
                        'gold_info': query_gold_info
                    })
        
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
        'query_stats': gold_doc_stats,
        'sample_queries': sample_queries
    }

def analyze_sample_queries(sample_queries: List[Dict], obj_id_to_info: Dict):
    
    for i, query_info in enumerate(sample_queries[:10]):
        
        for gold_info in query_info['gold_info']:
            obj_id = gold_info['id']
            if obj_id in obj_id_to_info:
                obj_data = obj_id_to_info[obj_id]
                

                if gold_info['type'] == 'gene/protein':
                    summary = obj_data.get('summary', '')
                    if summary:
                elif gold_info['type'] == 'disease':
                    mondo_def = obj_data.get('mondo_definition', '')
                    if mondo_def:
                elif gold_info['type'] == 'drug':
                    indication = obj_data.get('indication', '')
                    if indication and isinstance(indication, str):
                elif gold_info['type'] == 'pathway':
                    summation = obj_data.get('summation', '')
                    if summation:
            else:

def analyze_id_ranges(id_ranges: Dict):
    
    for obj_type, range_info in id_ranges.items():

def analyze_query_types(sample_queries: List[Dict]):
    
    query_patterns = Counter()
    question_words = Counter()
    
    for query_info in sample_queries:
        query = query_info['query'].lower()
        

        if 'which' in query:
            query_patterns['Which questions'] += 1
        if 'what' in query:
            query_patterns['What questions'] += 1
        if 'how' in query:
            query_patterns['How questions'] += 1
        if 'why' in query:
            query_patterns['Why questions'] += 1
        if 'can you' in query or 'could you' in query:
            query_patterns['Request questions'] += 1
        

        words = query.split()
        for word in words:
            if word in ['which', 'what', 'how', 'why', 'where', 'when', 'who']:
                question_words[word] += 1
    
    for pattern, count in query_patterns.most_common():
    
    for word, count in question_words.most_common(5):

def main():
    

    obj_id_to_info, id_ranges = load_prime_data()
    

    analyze_id_ranges(id_ranges)
    

    csv_files = [
        "",
        ""
    ]
    
    all_results = {}
    all_sample_queries = []
    
    for csv_file in csv_files:
        results = analyze_csv_file(csv_file, obj_id_to_info, id_ranges)
        all_results[csv_file.split('/')[-1]] = results
        all_sample_queries.extend(results['sample_queries'])
    

    
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
    

    if all_sample_queries:
        analyze_sample_queries(all_sample_queries, obj_id_to_info)
        analyze_query_types(all_sample_queries)
    

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
