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

class PRIMENeighborGraphBuilder:
    
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
        

        self.cross_entity_connections = {

            'gene/protein': {
                'to_disease': [('gene_summary_embedding', 'disease_definition_embedding')],
                'to_drug': [('gene_summary_embedding', 'drug_mechanism_embedding')],
                'to_pathway': [('gene_summary_embedding', 'pathway_summation_embedding')]
            },

            'disease': {
                'to_gene': [('disease_definition_embedding', 'gene_summary_embedding'),
                           ('disease_clinical_embedding', 'gene_summary_embedding')],
                'to_drug': [('disease_definition_embedding', 'drug_indication_embedding'),
                           ('disease_symptoms_embedding', 'drug_indication_embedding')],
                'to_pathway': [('disease_definition_embedding', 'pathway_summation_embedding')]
            },

            'drug': {
                'to_gene': [('drug_mechanism_embedding', 'gene_summary_embedding')],
                'to_disease': [('drug_indication_embedding', 'disease_definition_embedding')],
                'to_pathway': [('drug_mechanism_embedding', 'pathway_summation_embedding')]
            },

            'pathway': {
                'to_gene': [('pathway_summation_embedding', 'gene_summary_embedding')],
                'to_disease': [('pathway_summation_embedding', 'disease_definition_embedding')],
                'to_drug': [('pathway_summation_embedding', 'drug_mechanism_embedding')]
            }
        }
        

        self.stats = {
            'total_nodes_processed': 0,
            'entity_type_counts': defaultdict(int),
            'total_connections': 0,
            'cross_type_connections': defaultdict(int),
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
        

        same_type_features = self._get_features_for_type(node_type)
        
        for feature in same_type_features:
            query_embedding = self.get_node_embedding(node, feature)
            if query_embedding is not None:
                neighbors = self.query_hnsw_neighbors(feature, query_embedding, exclude_id=object_id)
                if neighbors:
                    all_neighbors[feature] = neighbors
                    self.stats['feature_query_counts'][feature] += 1
        

        if node_type in self.cross_entity_connections:
            cross_type_neighbors = self._find_cross_type_neighbors(node, node_type)
            all_neighbors.update(cross_type_neighbors)
        

        total_connections = sum(len(neighbors) for neighbors in all_neighbors.values())
        self.stats['total_connections'] += total_connections
        
        return all_neighbors
    
    def _get_features_for_type(self, node_type: str) -> List[str]:
        common_features = ['content_embedding', 'entity_name_embedding']
        
        type_specific = {
            'gene/protein': ['gene_summary_embedding', 'gene_full_name_embedding', 'gene_alias_embedding'],
            'disease': ['disease_definition_embedding', 'disease_clinical_embedding', 'disease_symptoms_embedding'],
            'drug': ['drug_description_embedding', 'drug_indication_embedding', 'drug_mechanism_embedding'],
            'pathway': ['pathway_summation_embedding', 'pathway_go_terms_embedding']
        }
        
        features = common_features.copy()
        if node_type in type_specific:
            features.extend(type_specific[node_type])
        
        return features
    
    def _find_cross_type_neighbors(self, node: Dict, node_type: str) -> Dict[str, List[Tuple[int, float]]]:
        cross_neighbors = {}
        
        connections = self.cross_entity_connections.get(node_type, {})
        
        for target_type, feature_pairs in connections.items():
            for source_feature, target_feature in feature_pairs:

                query_embedding = self.get_node_embedding(node, source_feature)
                
                if query_embedding is not None and target_feature in self.indices:
                    neighbors = self.query_hnsw_neighbors(target_feature, query_embedding)
                    
                    if neighbors:
                        connection_key = f"cross_{target_type}_{source_feature}_to_{target_feature}"
                        cross_neighbors[connection_key] = neighbors
                        self.stats['cross_type_connections'][connection_key] += len(neighbors)
                        self.stats['feature_query_counts'][connection_key] += 1
        
        return cross_neighbors
    
    def process_all_chunks(self):
        
        chunk_files = sorted(self.embeddings_dir.glob(""))
        
        all_neighbors = {}
        
        for i, chunk_file in enumerate(chunk_files):
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
            
            chunk_neighbors = {}
            
            for j, node in enumerate(chunk_data):
                if j % 5000 == 0 and j > 0:
                
                object_id = node['object_id']
                node_type = node['node_type']
                

                neighbors = self.find_node_neighbors(node)
                chunk_neighbors[object_id] = {
                    'node_type': node_type,
                    'neighbors': neighbors
                }
                

                self.stats['total_nodes_processed'] += 1
                self.stats['entity_type_counts'][node_type] += 1
            

            all_neighbors.update(chunk_neighbors)
            
            

            del chunk_data, chunk_neighbors
            gc.collect()
        
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
            'statistics': {
                'total_nodes_processed': self.stats['total_nodes_processed'],
                'entity_type_counts': dict(self.stats['entity_type_counts']),
                'total_connections': self.stats['total_connections'],
                'cross_type_connections': dict(self.stats['cross_type_connections']),
                'feature_query_counts': dict(self.stats['feature_query_counts'])
            },
            'cross_entity_strategies': self.cross_entity_connections,
            'hnsw_manifest_used': self.manifest
        }
        
        stats_path = self.output_dir / ""
        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        
        return graph_path, stats_path
    
    def _log_final_statistics(self):
        
        for entity_type in sorted(self.stats['entity_type_counts'].keys()):
            count = self.stats['entity_type_counts'][entity_type]
        
        
        total_cross = sum(self.stats['cross_type_connections'].values())
        

        cross_by_type = defaultdict(int)
        for key, count in self.stats['cross_type_connections'].items():
            target_type = key.split('_')[1]
            cross_by_type[target_type] += count
        
        for target_type in sorted(cross_by_type.keys()):
        
        sorted_features = sorted(self.stats['feature_query_counts'].items(), 
                                key=lambda x: x[1], reverse=True)[:15]
        for feature, count in sorted_features:
    
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
        builder = PRIMENeighborGraphBuilder(embeddings_dir, hnsw_dir, output_dir)
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

