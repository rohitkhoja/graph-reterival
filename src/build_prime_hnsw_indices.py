#!/usr/bin/env python3

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
import faiss
import gc

class PRIMEHNSWBuilder:
    
    def __init__(self, embeddings_dir: str, output_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        

        self.common_features = [
            'content_embedding',
            'entity_name_embedding'
        ]
        
        self.gene_features = [
            'gene_summary_embedding',
            'gene_full_name_embedding',
            'gene_alias_embedding'
        ]
        
        self.disease_features = [
            'disease_definition_embedding',
            'disease_clinical_embedding',
            'disease_symptoms_embedding'
        ]
        
        self.drug_features = [
            'drug_description_embedding',
            'drug_indication_embedding',
            'drug_mechanism_embedding'
        ]
        
        self.pathway_features = [
            'pathway_summation_embedding',
            'pathway_go_terms_embedding'
        ]
        

        self.all_features = (
            self.common_features + 
            self.gene_features + 
            self.disease_features + 
            self.drug_features + 
            self.pathway_features
        )
        

        self.embedding_dim = 384
        self.hnsw_params = {
            'M': 64,
            'ef_construction': 2000,
            'ef_search': 1000,
        }
        

        self.stats = {
            'total_chunks': 0,
            'total_nodes': 0,
            'entity_type_counts': defaultdict(int),
            'feature_counts': defaultdict(int),
            'empty_feature_counts': defaultdict(int)
        }
        
    def scan_all_chunks(self) -> Dict[str, Dict[int, np.ndarray]]:
        
        feature_embeddings = {
            feature: {} for feature in self.all_features
        }
        
        chunk_files = sorted(self.embeddings_dir.glob(""))
        self.stats['total_chunks'] = len(chunk_files)
        
        
        for i, chunk_file in enumerate(chunk_files):
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
            
            self.stats['total_nodes'] += len(chunk_data)
            
            for node in chunk_data:
                object_id = node['object_id']
                node_type = node['node_type']
                

                self.stats['entity_type_counts'][node_type] += 1
                

                for feature in self.all_features:
                    if feature in node:
                        embedding = node[feature]
                        

                        if embedding and len(embedding) == self.embedding_dim:
                            feature_embeddings[feature][object_id] = np.array(embedding, dtype=np.float32)
                            self.stats['feature_counts'][feature] += 1
                        else:
                            self.stats['empty_feature_counts'][feature] += 1
                    else:
                        self.stats['empty_feature_counts'][feature] += 1
        
        self._log_statistics()
        
        return feature_embeddings
    
    def _log_statistics(self):
        
        for entity_type in sorted(self.stats['entity_type_counts'].keys()):
            count = self.stats['entity_type_counts'][entity_type]
        
        

        feature_groups = [
            ("Common Features", self.common_features),
            ("Gene/Protein Features", self.gene_features),
            ("Disease Features", self.disease_features),
            ("Drug Features", self.drug_features),
            ("Pathway Features", self.pathway_features)
        ]
        
        for group_name, features in feature_groups:
            for feature in features:
                if feature in self.stats['feature_counts']:
                    valid_count = self.stats['feature_counts'][feature]
                    empty_count = self.stats['empty_feature_counts'][feature]
                    total = valid_count + empty_count
                    pct = (valid_count / total * 100) if total > 0 else 0
    
    def build_hnsw_index(self, feature_name: str, embeddings_dict: Dict[int, np.ndarray]) -> str:
        if not embeddings_dict:
            return None
            
        

        object_ids = list(embeddings_dict.keys())
        embeddings_matrix = np.array([embeddings_dict[oid] for oid in object_ids], dtype=np.float32)
        
        

        hnsw_index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_params['M'])
        hnsw_index.hnsw.efConstruction = self.hnsw_params['ef_construction']
        hnsw_index.hnsw.efSearch = self.hnsw_params['ef_search']
        

        faiss.normalize_L2(embeddings_matrix)
        

        start_time = time.time()
        hnsw_index.add(embeddings_matrix)
        build_time = time.time() - start_time
        

        index_path = self.output_dir / f"{feature_name}_hnsw.faiss"
        faiss.write_index(hnsw_index, str(index_path))
        

        mapping_path = self.output_dir / f"{feature_name}_mapping.pkl"
        with open(mapping_path, 'wb') as f:
            pickle.dump(object_ids, f)
        
        

        del embeddings_matrix
        gc.collect()
        
        return str(index_path)
    
    def build_all_indices(self):
        
        total_start_time = time.time()
        

        feature_embeddings = self.scan_all_chunks()
        

        
        built_indices = {}
        
        for i, feature in enumerate(self.all_features):
            }: {feature} ---")
            
            embeddings_dict = feature_embeddings[feature]
            index_path = self.build_hnsw_index(feature, embeddings_dict)
            
            if index_path:
                built_indices[feature] = {
                    'index_path': index_path,
                    'mapping_path': str(self.output_dir / f"{feature}_mapping.pkl"),
                    'count': len(embeddings_dict)
                }
        

        manifest = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': time.time() - total_start_time,
            'embedding_dimension': self.embedding_dim,
            'hnsw_parameters': self.hnsw_params,
            'statistics': {
                'total_chunks': self.stats['total_chunks'],
                'total_nodes': self.stats['total_nodes'],
                'entity_type_counts': dict(self.stats['entity_type_counts']),
                'feature_counts': dict(self.stats['feature_counts']),
                'empty_feature_counts': dict(self.stats['empty_feature_counts'])
            },
            'indices': built_indices,
            'feature_categories': {
                'common_features': self.common_features,
                'gene_features': self.gene_features,
                'disease_features': self.disease_features,
                'drug_features': self.drug_features,
                'pathway_features': self.pathway_features
            }
        }
        
        manifest_path = self.output_dir / ""
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        total_time = time.time() - total_start_time
        

        for group_name, features in [
            ("Common", self.common_features),
            ("Gene/Protein", self.gene_features),
            ("Disease", self.disease_features),
            ("Drug", self.drug_features),
            ("Pathway", self.pathway_features)
        ]:
            count = sum(1 for f in features if f in built_indices)
            if count > 0:
        
        return manifest_path

def main():
    

    embeddings_dir = ""
    output_dir = ""
    
    

    if not Path(embeddings_dir).exists():
        return False
    

    chunk_files = list(Path(embeddings_dir).glob(""))
    if not chunk_files:
         is complete")
        return False
    
    

    try:
        builder = PRIMEHNSWBuilder(embeddings_dir, output_dir)
        manifest_path = builder.build_all_indices()
        
         to build the graph")
        
        return True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

