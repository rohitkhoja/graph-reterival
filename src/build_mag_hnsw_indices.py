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

class MAGHNSWBuilder:
    
    def __init__(self, embeddings_dir: str, output_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        

        self.paper_features = [
            'content_embedding',
            'original_title_embedding', 
            'abstract_embedding',
            'authors_embedding',
            'fields_of_study_embedding',
            'cites_embedding'
        ]
        
        self.author_features = [
            'content_embedding',
            'display_name_embedding',
            'institution_embedding'
        ]
        

        self.embedding_dim = 384
        self.hnsw_params = {
            'M': 64,
            'ef_construction': 2000,
            'ef_search': 1000,
        }
        

        self.stats = {
            'total_chunks': 0,
            'total_nodes': 0,
            'paper_nodes': 0,
            'author_nodes': 0,
            'feature_counts': defaultdict(int),
            'empty_feature_counts': defaultdict(int)
        }
        
    def scan_all_chunks(self) -> Dict[str, Dict[int, np.ndarray]]:
        
        feature_embeddings = {
            feature: {} for feature in (self.paper_features + self.author_features)
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
                

                if node_type == 'paper':
                    self.stats['paper_nodes'] += 1
                    relevant_features = self.paper_features
                elif node_type == 'author':
                    self.stats['author_nodes'] += 1
                    relevant_features = self.author_features
                else:
                    continue
                

                for feature in relevant_features:
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
        
        for feature in sorted(self.stats['feature_counts'].keys()):
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
        
        all_features = self.paper_features + self.author_features
        for i, feature in enumerate(all_features):
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
            'statistics': dict(self.stats),
            'indices': built_indices
        }
        
        manifest_path = self.output_dir / ""
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        total_time = time.time() - total_start_time
        
        return manifest_path

def main():
    

    embeddings_dir = ""
    output_dir = ""
    
    

    if not Path(embeddings_dir).exists():
        return False
    

    try:
        builder = MAGHNSWBuilder(embeddings_dir, output_dir)
        manifest_path = builder.build_all_indices()
        
        
        return True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

