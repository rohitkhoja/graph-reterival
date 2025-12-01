#!/usr/bin/env python3

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import faiss
import gc

class MAGNeighborGraphBuilder:
    
    def __init__(self, embeddings_dir: str, hnsw_dir: str, output_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.hnsw_dir = Path(hnsw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        

        self.manifest = self._load_manifest()
        self.indices = {}
        self.mappings = {}
        

        self.k_neighbors = 1000
        self.ef_search = 200
        

        self.stats = {
            'total_nodes_processed': 0,
            'paper_nodes_processed': 0,
            'author_nodes_processed': 0,
            'total_connections': 0,
            'cross_type_connections': 0,
            'feature_query_counts': defaultdict(int)
        }
        
    def _load_manifest(self) -> Dict:
        manifest_path = self.hnsw_dir / ""
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    def load_hnsw_indices(self):
        
        for feature, info in self.manifest['indices'].items():
            

            index = faiss.read_index(info['index_path'])
            

            if hasattr(index, 'hnsw'):
                index.hnsw.efSearch = self.ef_search
            

            with open(info['mapping_path'], 'rb') as f:
                object_ids = pickle.load(f)
            
            self.indices[feature] = index
            self.mappings[feature] = object_ids
            
    
    def get_node_embedding(self, node: Dict, feature: str) -> Optional[np.ndarray]:
        if feature in node and node[feature]:
            embedding = node[feature]
            if len(embedding) == self.manifest['embedding_dimension']:
                return np.array(embedding, dtype=np.float32).reshape(1, -1)
        return None
    
    def query_hnsw_neighbors(self, feature: str, query_embedding: np.ndarray, 
                           exclude_id: Optional[int] = None) -> List[Tuple[int, float]]:
        if feature not in self.indices:
            return []
        
        try:

            query_normalized = query_embedding.copy()
            faiss.normalize_L2(query_normalized)
            

            k_search = min(self.k_neighbors + 10, self.indices[feature].ntotal)
            similarities, indices = self.indices[feature].search(query_normalized, k_search)
            

            neighbors = []
            object_ids = self.mappings[feature]
            
            for idx, similarity in zip(indices[0], similarities[0]):
                if idx == -1:
                    continue
                    
                object_id = object_ids[idx]
                

                if exclude_id is not None and object_id == exclude_id:
                    continue
                

                similarity_score = max(0.0, float(similarity))
                neighbors.append((object_id, similarity_score))
            

            neighbors.sort(key=lambda x: x[1], reverse=True)
            return neighbors[:self.k_neighbors]
            
        except Exception as e:
            return []
    
    def find_node_neighbors(self, node: Dict) -> Dict[str, List[Tuple[int, float]]]:
        object_id = node['object_id']
        node_type = node['node_type']
        all_neighbors = {}
        
        if node_type == 'paper':

            paper_features = [
                'content_embedding',
                'original_title_embedding', 
                'abstract_embedding',
                'authors_embedding',
                'fields_of_study_embedding',
                'cites_embedding'
            ]
            
            for feature in paper_features:
                query_embedding = self.get_node_embedding(node, feature)
                if query_embedding is not None:
                    neighbors = self.query_hnsw_neighbors(feature, query_embedding, exclude_id=object_id)
                    if neighbors:
                        all_neighbors[feature] = neighbors
                        self.stats['feature_query_counts'][feature] += 1
            


            authors_embedding = self.get_node_embedding(node, 'authors_embedding')
            if authors_embedding is not None and 'display_name_embedding' in self.indices:
                author_neighbors = self.query_hnsw_neighbors('display_name_embedding', authors_embedding)
                if author_neighbors:
                    all_neighbors['cross_type_authors'] = author_neighbors
                    self.stats['cross_type_connections'] += len(author_neighbors)
                    self.stats['feature_query_counts']['cross_type_authors'] += 1
                    
        elif node_type == 'author':

            author_features = [
                'content_embedding',
                'display_name_embedding',
                'institution_embedding'
            ]
            
            for feature in author_features:
                query_embedding = self.get_node_embedding(node, feature)
                if query_embedding is not None:
                    neighbors = self.query_hnsw_neighbors(feature, query_embedding, exclude_id=object_id)
                    if neighbors:
                        all_neighbors[feature] = neighbors
                        self.stats['feature_query_counts'][feature] += 1
            


            display_name_embedding = self.get_node_embedding(node, 'display_name_embedding')
            if display_name_embedding is not None and 'authors_embedding' in self.indices:
                paper_neighbors = self.query_hnsw_neighbors('authors_embedding', display_name_embedding)
                if paper_neighbors:
                    all_neighbors['cross_type_papers'] = paper_neighbors
                    self.stats['cross_type_connections'] += len(paper_neighbors)
                    self.stats['feature_query_counts']['cross_type_papers'] += 1
        

        total_connections = sum(len(neighbors) for neighbors in all_neighbors.values())
        self.stats['total_connections'] += total_connections
        
        return all_neighbors
    
    def process_all_chunks(self):
        
        chunk_files = sorted(self.embeddings_dir.glob(""))
        
        for i, chunk_file in enumerate(chunk_files):

            chunk_output_path = self.output_dir / f""
            if chunk_output_path.exists():
                }) - already processed")
                continue
            
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
            
            chunk_neighbors = {}
            
            for j, node in enumerate(chunk_data):
                if j % 10000 == 0 and j > 0:
                
                object_id = node['object_id']
                node_type = node['node_type']
                

                neighbors = self.find_node_neighbors(node)
                chunk_neighbors[object_id] = {
                    'node_type': node_type,
                    'neighbors': neighbors
                }
                

                self.stats['total_nodes_processed'] += 1
                if node_type == 'paper':
                    self.stats['paper_nodes_processed'] += 1
                elif node_type == 'author':
                    self.stats['author_nodes_processed'] += 1
            

            with open(chunk_output_path, 'w') as f:
                json.dump(chunk_neighbors, f, indent=2)
            

            del chunk_neighbors
            gc.collect()
        

        all_neighbors = {}
        for chunk_file in sorted(self.output_dir.glob("")):
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
                all_neighbors.update(chunk_data)
        
        return all_neighbors
    
    def save_final_graph(self, all_neighbors: Dict):
        

        graph_path = self.output_dir / ""
        with open(graph_path, 'w') as f:
            json.dump(all_neighbors, f, indent=2)
        

        final_stats = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_nodes': len(all_neighbors),
            'k_neighbors': self.k_neighbors,
            'ef_search': self.ef_search,
            'statistics': dict(self.stats),
            'feature_query_counts': dict(self.stats['feature_query_counts']),
            'hnsw_manifest_used': self.manifest
        }
        
        stats_path = self.output_dir / ""
        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        
        return graph_path, stats_path
    
    def _log_final_statistics(self):
        
        for feature in sorted(self.stats['feature_query_counts'].keys()):
            count = self.stats['feature_query_counts'][feature]
    
    def build_neighbor_graph(self):
        
        total_start_time = time.time()
        
        try:

            self.load_hnsw_indices()
            

            all_neighbors = self.process_all_chunks()
            

            graph_path, stats_path = self.save_final_graph(all_neighbors)
            
            total_time = time.time() - total_start_time
            

            self._log_final_statistics()
            
            
            return graph_path
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

def main():
    

    embeddings_dir = ""
    hnsw_dir = ""
    output_dir = ""
    
    

    for path_name, path in [("Embeddings", embeddings_dir), ("HNSW", hnsw_dir)]:
        if not Path(path).exists():
            return False
    

    try:
        builder = MAGNeighborGraphBuilder(embeddings_dir, hnsw_dir, output_dir)
        graph_path = builder.build_neighbor_graph()
        
        if graph_path:
            return True
        else:
            return False
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False
 
if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)