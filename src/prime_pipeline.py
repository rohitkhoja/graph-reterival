#!/usr/bin/env python3

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
from dataclasses import dataclass
import pickle
import torch
import gc
from tqdm import tqdm
import math
import random


sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.processors.embedding_service import EmbeddingService
from src.core.models import ProcessingConfig

@dataclass
class PRIMENode:
    object_id: int
    node_type: str
    content: str
    

    name: str
    source: str
    

    gene_summary: Optional[str] = None
    gene_full_name: Optional[str] = None
    gene_alias: Optional[str] = None
    

    disease_definition: Optional[str] = None
    disease_clinical: Optional[str] = None
    disease_symptoms: Optional[str] = None
    

    drug_description: Optional[str] = None
    drug_indication: Optional[str] = None
    drug_mechanism: Optional[str] = None
    

    pathway_summation: Optional[str] = None
    pathway_go_terms: Optional[str] = None
    

    content_embedding: Optional[List[float]] = None
    gene_summary_embedding: Optional[List[float]] = None
    gene_full_name_embedding: Optional[List[float]] = None
    gene_alias_embedding: Optional[List[float]] = None
    disease_definition_embedding: Optional[List[float]] = None
    disease_clinical_embedding: Optional[List[float]] = None
    disease_symptoms_embedding: Optional[List[float]] = None
    drug_description_embedding: Optional[List[float]] = None
    drug_indication_embedding: Optional[List[float]] = None
    drug_mechanism_embedding: Optional[List[float]] = None
    pathway_summation_embedding: Optional[List[float]] = None
    pathway_go_terms_embedding: Optional[List[float]] = None
    entity_name_embedding: Optional[List[float]] = None
    

    metadata: Optional[Dict[str, Any]] = None

class PRIMEPipeline:
    
    def __init__(self,
                 prime_dataset_file: str = "",
                 cache_dir: str = "",
                 target_chunks: int = 10,
                 use_gpu: bool = True):
        
        self.prime_dataset_file = Path(prime_dataset_file)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_chunks = target_chunks
        self.use_gpu = use_gpu and torch.cuda.is_available()
        

        self.embeddings_dir = self.cache_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        

        self.config = ProcessingConfig(
            use_faiss=True,
            faiss_use_gpu=self.use_gpu,
            batch_size=512,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_service = EmbeddingService(self.config)
        

        self.stats = {
            'total_nodes': 0,
            'entity_type_counts': {},
            'feature_counts': {},
            'chunks_processed': 0
        }
    
    def run_pipeline(self, max_nodes: Optional[int] = None):
        
        total_start = time.time()
        
        try:

            nodes = self.phase1_load_and_prepare(max_nodes)
            

            self.phase2_generate_embeddings(nodes)
            

            self.phase3_save_embeddings(nodes)
            
            total_time = time.time() - total_start
            
            return self.embeddings_dir
            
        except Exception as e:
            raise
    
    def phase1_load_and_prepare(self, max_nodes: Optional[int] = None) -> List[PRIMENode]:
        start_time = time.time()
        
        with open(self.prime_dataset_file, 'r') as f:
            prime_data = json.load(f)
        
        

        nodes = []
        for obj_id, obj_data in tqdm(prime_data.items(), desc="   Processing entities"):
            if max_nodes and len(nodes) >= max_nodes:
                break
            
            node = self._create_prime_node(int(obj_id), obj_data)
            if node:
                nodes.append(node)
                

                node_type = node.node_type
                if node_type not in self.stats['entity_type_counts']:
                    self.stats['entity_type_counts'][node_type] = 0
                self.stats['entity_type_counts'][node_type] += 1
        
        self.stats['total_nodes'] = len(nodes)
        
        for entity_type, count in sorted(self.stats['entity_type_counts'].items()):
        
        return nodes
    
    def _create_prime_node(self, obj_id: int, obj_data: Dict) -> Optional[PRIMENode]:
        try:
            node_type = obj_data.get('type', 'unknown')
            name = obj_data.get('name', '')
            source = obj_data.get('source', '')
            details = obj_data.get('details', {})
            

            if node_type == 'gene/protein':
                return self._create_gene_node(obj_id, name, source, details)
            elif node_type == 'disease':
                return self._create_disease_node(obj_id, name, source, details)
            elif node_type == 'drug':
                return self._create_drug_node(obj_id, name, source, details)
            elif node_type == 'pathway':
                return self._create_pathway_node(obj_id, name, source, details)
            else:

                return self._create_simple_node(obj_id, node_type, name, source)
        
        except Exception as e:
            return None
    
    def _create_gene_node(self, obj_id: int, name: str, source: str, details: Dict) -> PRIMENode:

        gene_summary = details.get('summary', '')
        gene_full_name = details.get('name', '')
        gene_alias = ' '.join(details.get('alias', [])) if details.get('alias') else ''
        

        content_parts = [name]
        
        if gene_full_name:
            content_parts.append(gene_full_name)
        if gene_summary:
            content_parts.append(gene_summary)
        if gene_alias:
            content_parts.append(gene_alias)
        

        if details.get('_id'):
            content_parts.append(f"NCBI:{details['_id']}")
        

        genomic_pos = details.get('genomic_pos')
        if genomic_pos:
            if isinstance(genomic_pos, dict) and genomic_pos.get('ensemblgene'):
                content_parts.append(f"ENSEMBL:{genomic_pos['ensemblgene']}")
            elif isinstance(genomic_pos, list) and len(genomic_pos) > 0:

                first_pos = genomic_pos[0]
                if isinstance(first_pos, dict) and first_pos.get('ensemblgene'):
                    content_parts.append(f"ENSEMBL:{first_pos['ensemblgene']}")
        
        content = ' '.join(content_parts)
        
        return PRIMENode(
            object_id=obj_id,
            node_type='gene/protein',
            content=content,
            name=name,
            source=source,
            gene_summary=gene_summary if gene_summary else None,
            gene_full_name=gene_full_name if gene_full_name else None,
            gene_alias=gene_alias if gene_alias else None,
            metadata={'genomic_pos': details.get('genomic_pos')}
        )
    
    def _create_disease_node(self, obj_id: int, name: str, source: str, details: Dict) -> PRIMENode:

        def safe_get(dict_obj, key, default=''):
            val = dict_obj.get(key, default)

            if val is None or (isinstance(val, float) and str(val) == 'nan'):
                return ''
            return str(val)
        

        disease_definition = safe_get(details, 'mondo_definition')
        disease_clinical = safe_get(details, 'orphanet_clinical_description') or safe_get(details, 'orphanet_definition')
        disease_symptoms = safe_get(details, 'mayo_symptoms')
        

        content_parts = [name]
        
        if disease_definition:
            content_parts.append(disease_definition)
        if disease_clinical:
            content_parts.append(disease_clinical)
        if disease_symptoms:
            content_parts.append(disease_symptoms)
        

        mondo_id = details.get('mondo_id')
        if mondo_id and not (isinstance(mondo_id, float) and str(mondo_id) == 'nan'):
            content_parts.append(f"MONDO:{mondo_id}")
        
        mondo_name = safe_get(details, 'mondo_name')
        if mondo_name:
            content_parts.append(mondo_name)
        

        mayo_causes = safe_get(details, 'mayo_causes')
        if mayo_causes:
            content_parts.append(mayo_causes)
        
        mayo_complications = safe_get(details, 'mayo_complications')
        if mayo_complications:
            content_parts.append(mayo_complications)
        
        content = ' '.join(content_parts)
        
        return PRIMENode(
            object_id=obj_id,
            node_type='disease',
            content=content,
            name=name,
            source=source,
            disease_definition=disease_definition if disease_definition else None,
            disease_clinical=disease_clinical if disease_clinical else None,
            disease_symptoms=disease_symptoms if disease_symptoms else None,
            metadata={'mondo_id': details.get('mondo_id')}
        )
    
    def _create_drug_node(self, obj_id: int, name: str, source: str, details: Dict) -> PRIMENode:

        def safe_get(dict_obj, key, default=''):
            val = dict_obj.get(key, default)

            if val is None or (isinstance(val, float) and str(val) == 'nan'):
                return ''
            return str(val)
        

        drug_description = safe_get(details, 'description')
        drug_indication = safe_get(details, 'indication')
        drug_mechanism = safe_get(details, 'mechanism_of_action')
        

        content_parts = [name]
        
        if drug_description:
            content_parts.append(drug_description)
        if drug_indication:
            content_parts.append(drug_indication)
        if drug_mechanism:
            content_parts.append(drug_mechanism)
        

        category = safe_get(details, 'category')
        if category:
            content_parts.append(category)
        
        group = safe_get(details, 'group')
        if group:
            content_parts.append(group)
        

        for atc_field in ['atc_1', 'atc_2', 'atc_3', 'atc_4']:
            atc_val = safe_get(details, atc_field)
            if atc_val:
                content_parts.append(atc_val)
        
        content = ' '.join(content_parts)
        
        return PRIMENode(
            object_id=obj_id,
            node_type='drug',
            content=content,
            name=name,
            source=source,
            drug_description=drug_description if drug_description else None,
            drug_indication=drug_indication if drug_indication else None,
            drug_mechanism=drug_mechanism if drug_mechanism else None,
            metadata={'molecular_weight': details.get('molecular_weight')}
        )
    
    def _create_pathway_node(self, obj_id: int, name: str, source: str, details: Dict) -> PRIMENode:

        pathway_summation = ''
        if details.get('summation'):
            summation_data = details['summation']
            if isinstance(summation_data, list):

                texts = []
                for item in summation_data:
                    if isinstance(item, dict) and 'text' in item:
                        texts.append(item['text'])
                    elif isinstance(item, str):
                        texts.append(item)
                pathway_summation = ' '.join(texts)
            elif isinstance(summation_data, str):
                pathway_summation = summation_data
        
        pathway_go_terms = ''
        if details.get('goBiologicalProcess'):
            go_terms = details['goBiologicalProcess']
            if isinstance(go_terms, list):
                pathway_go_terms = ' '.join([str(term) for term in go_terms])
        

        content_parts = [name]
        
        if details.get('displayName'):
            content_parts.append(details['displayName'])
        if pathway_summation:
            content_parts.append(pathway_summation)
        if pathway_go_terms:
            content_parts.append(pathway_go_terms)
        

        if details.get('stId'):
            content_parts.append(f"REACTOME:{details['stId']}")
        if details.get('speciesName'):
            content_parts.append(details['speciesName'])
        
        content = ' '.join(content_parts)
        
        return PRIMENode(
            object_id=obj_id,
            node_type='pathway',
            content=content,
            name=name,
            source=source,
            pathway_summation=pathway_summation if pathway_summation else None,
            pathway_go_terms=pathway_go_terms if pathway_go_terms else None,
            metadata={'stId': details.get('stId')}
        )
    
    def _create_simple_node(self, obj_id: int, node_type: str, name: str, source: str) -> PRIMENode:

        content = name
        
        return PRIMENode(
            object_id=obj_id,
            node_type=node_type,
            content=content,
            name=name,
            source=source
        )
    
    def phase2_generate_embeddings(self, nodes: List[PRIMENode]):
        start_time = time.time()
        

        embedding_tasks = {
            'content': [],
            'gene_summary': [],
            'gene_full_name': [],
            'gene_alias': [],
            'disease_definition': [],
            'disease_clinical': [],
            'disease_symptoms': [],
            'drug_description': [],
            'drug_indication': [],
            'drug_mechanism': [],
            'pathway_summation': [],
            'pathway_go_terms': [],
            'entity_name': []
        }
        

        for node_idx, node in enumerate(nodes):

            if node.content:
                embedding_tasks['content'].append((node_idx, node.content))
            

            if node.name:
                embedding_tasks['entity_name'].append((node_idx, node.name))
            

            if node.node_type == 'gene/protein':
                if node.gene_summary:
                    embedding_tasks['gene_summary'].append((node_idx, node.gene_summary))
                if node.gene_full_name:
                    embedding_tasks['gene_full_name'].append((node_idx, node.gene_full_name))
                if node.gene_alias:
                    embedding_tasks['gene_alias'].append((node_idx, node.gene_alias))
            
            elif node.node_type == 'disease':
                if node.disease_definition:
                    embedding_tasks['disease_definition'].append((node_idx, node.disease_definition))
                if node.disease_clinical:
                    embedding_tasks['disease_clinical'].append((node_idx, node.disease_clinical))
                if node.disease_symptoms:
                    embedding_tasks['disease_symptoms'].append((node_idx, node.disease_symptoms))
            
            elif node.node_type == 'drug':
                if node.drug_description:
                    embedding_tasks['drug_description'].append((node_idx, node.drug_description))
                if node.drug_indication:
                    embedding_tasks['drug_indication'].append((node_idx, node.drug_indication))
                if node.drug_mechanism:
                    embedding_tasks['drug_mechanism'].append((node_idx, node.drug_mechanism))
            
            elif node.node_type == 'pathway':
                if node.pathway_summation:
                    embedding_tasks['pathway_summation'].append((node_idx, node.pathway_summation))
                if node.pathway_go_terms:
                    embedding_tasks['pathway_go_terms'].append((node_idx, node.pathway_go_terms))
        

        for feature_name, feature_data in embedding_tasks.items():
            if not feature_data:
                continue
            
            
            try:

                texts = [text for _, text in feature_data]
                node_indices = [node_idx for node_idx, _ in feature_data]
                

                embeddings = []
                chunk_size = 5000
                
                for i in range(0, len(texts), chunk_size):
                    chunk_texts = texts[i:i+chunk_size]
                    chunk_embeddings = self.embedding_service.generate_embeddings_bulk(chunk_texts)
                    embeddings.extend(chunk_embeddings)
                

                embedding_attr = f"{feature_name}_embedding"
                for node_idx, embedding in zip(node_indices, embeddings):
                    setattr(nodes[node_idx], embedding_attr, embedding)
                
                self.stats['feature_counts'][feature_name] = len(embeddings)
                
            except Exception as e:
            

            self._clear_gpu_memory()
    
    def phase3_save_embeddings(self, nodes: List[PRIMENode]):
        

        chunk_size = math.ceil(len(nodes) / self.target_chunks)
        
        for chunk_id in range(self.target_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, len(nodes))
            chunk_nodes = nodes[start_idx:end_idx]
            
            if not chunk_nodes:
                break
            

            chunk_file = self.embeddings_dir / f""
            self._save_chunk(chunk_nodes, chunk_file)
        

        stats_file = self.embeddings_dir / ""
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
    
    def _save_chunk(self, nodes: List[PRIMENode], file_path: Path):
        chunk_data = []
        
        for node in nodes:
            node_data = {
                'object_id': node.object_id,
                'node_type': node.node_type,
                'name': node.name,
                'source': node.source,
                'content': node.content,
                'content_embedding': node.content_embedding,
                'entity_name_embedding': node.entity_name_embedding,
                'metadata': node.metadata
            }
            

            if node.node_type == 'gene/protein':
                node_data.update({
                    'gene_summary': node.gene_summary,
                    'gene_full_name': node.gene_full_name,
                    'gene_alias': node.gene_alias,
                    'gene_summary_embedding': node.gene_summary_embedding,
                    'gene_full_name_embedding': node.gene_full_name_embedding,
                    'gene_alias_embedding': node.gene_alias_embedding
                })
            elif node.node_type == 'disease':
                node_data.update({
                    'disease_definition': node.disease_definition,
                    'disease_clinical': node.disease_clinical,
                    'disease_symptoms': node.disease_symptoms,
                    'disease_definition_embedding': node.disease_definition_embedding,
                    'disease_clinical_embedding': node.disease_clinical_embedding,
                    'disease_symptoms_embedding': node.disease_symptoms_embedding
                })
            elif node.node_type == 'drug':
                node_data.update({
                    'drug_description': node.drug_description,
                    'drug_indication': node.drug_indication,
                    'drug_mechanism': node.drug_mechanism,
                    'drug_description_embedding': node.drug_description_embedding,
                    'drug_indication_embedding': node.drug_indication_embedding,
                    'drug_mechanism_embedding': node.drug_mechanism_embedding
                })
            elif node.node_type == 'pathway':
                node_data.update({
                    'pathway_summation': node.pathway_summation,
                    'pathway_go_terms': node.pathway_go_terms,
                    'pathway_summation_embedding': node.pathway_summation_embedding,
                    'pathway_go_terms_embedding': node.pathway_go_terms_embedding
                })
            
            chunk_data.append(node_data)
        
        with open(file_path, 'w') as f:
            json.dump(chunk_data, f, indent=2)
    
    def _clear_gpu_memory(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        gc.collect()

def main():
    

    pipeline = PRIMEPipeline()
    
    try:

        embeddings_dir = pipeline.run_pipeline()
        
    except Exception as e:
        raise

if __name__ == "__main__":
    main()

