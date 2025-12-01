#!/usr/bin/env python3

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import pickle
import faiss
import torch
import re
import gc
from tqdm import tqdm
import math


sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.processors.embedding_service import EmbeddingService
from src.core.models import ProcessingConfig

@dataclass
class ChunkInfo:
    chunk_id: int
    start_idx: int
    end_idx: int
    node_count: int
    status: str
    embeddings_file: Optional[str] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class MAGChunkData:
    chunk_id: str
    node_type: str
    object_id: str
    content: str
    content_embedding: Optional[List[float]] = None
    

    original_title: Optional[str] = None
    publisher: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    fields_of_study: Optional[List[str]] = None
    cites: Optional[List[str]] = None
    

    original_title_embedding: Optional[List[float]] = None
    abstract_embedding: Optional[List[float]] = None
    authors_embedding: Optional[List[float]] = None
    fields_of_study_embedding: Optional[List[float]] = None
    cites_embedding: Optional[List[float]] = None
    

    display_name: Optional[str] = None
    institution: Optional[str] = None
    

    display_name_embedding: Optional[List[float]] = None
    institution_embedding: Optional[List[float]] = None
    

    metadata: Optional[Dict[str, Any]] = None

class MAGChunkedPipeline:
    
    def __init__(self, 
                 mag_dataset_file: str = "",
                 cache_dir: str = "",
                 chunk_size: int = 100000,
                 use_gpu: bool = True,
                 num_threads: int = 32):
        
        self.mag_dataset_file = Path(mag_dataset_file)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunk_size = chunk_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.num_threads = num_threads
        self.num_gpus = torch.cuda.device_count() if self.use_gpu else 0
        

        self.chunks_dir = self.cache_dir / "chunks"
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.final_dir = self.cache_dir / "final"
        
        for dir_path in [self.chunks_dir, self.embeddings_dir, self.final_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        

        self.config = ProcessingConfig(
            use_faiss=True, 
            faiss_use_gpu=self.use_gpu,
            batch_size=4096,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_service = EmbeddingService(self.config)
        

        self.chunk_manifest_file = self.cache_dir / ""
        self.chunks_info: List[ChunkInfo] = []
        self.total_nodes = 0
        self.paper_count = 0
        self.author_count = 0
        

    def run_chunked_pipeline(self, max_nodes: Optional[int] = None, start_chunk: Optional[int] = None, end_chunk: Optional[int] = None):
        
        total_start_time = time.time()
        
        try:

            self.phase1_analyze_and_create_chunks(max_nodes)
            

            self.phase2_process_chunks(start_chunk=start_chunk, end_chunk=end_chunk)
            

            self.phase3_merge_and_build_index()
            

            analysis_file = self.phase4_generate_analysis()
            
            total_time = time.time() - total_start_time
            
            
            return analysis_file
            
        except Exception as e:
            raise

    def phase1_analyze_and_create_chunks(self, max_nodes: Optional[int] = None):
        start_time = time.time()
        

        with open(self.mag_dataset_file, 'r') as f:
            mag_data = json.load(f)
        

        relevant_nodes = []
        paper_count = 0
        author_count = 0
        

        for obj_id, obj_data in mag_data.items():
            obj_type = obj_data.get('type')
            if obj_type in ['paper', 'author']:
                relevant_nodes.append((obj_id, obj_data))
                if obj_type == 'paper':
                    paper_count += 1
                elif obj_type == 'author':
                    author_count += 1
        
        self.total_nodes = len(relevant_nodes)
        self.paper_count = paper_count
        self.author_count = author_count
        self.relevant_nodes = relevant_nodes
        
        if max_nodes:
            self.total_nodes = min(self.total_nodes, max_nodes)
            self.relevant_nodes = self.relevant_nodes[:max_nodes]
        
        

        total_chunks = math.ceil(self.total_nodes / self.chunk_size)
        

        self.chunks_info = []
        for chunk_id in range(total_chunks):
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.total_nodes)
            
            chunk_info = ChunkInfo(
                chunk_id=chunk_id,
                start_idx=start_idx,
                end_idx=end_idx,
                node_count=end_idx - start_idx,
                status='pending'
            )
            self.chunks_info.append(chunk_info)
        

        self._save_chunk_manifest()
        
        phase_time = time.time() - start_time
        

        for chunk in self.chunks_info:

    def phase2_process_chunks(self, start_chunk: Optional[int] = None, end_chunk: Optional[int] = None):
        start_time = time.time()
        

        self._load_chunk_manifest()
        

        pending_chunks = [c for c in self.chunks_info if c.status in ['pending', 'failed']]
        completed_chunks = [c for c in self.chunks_info if c.status == 'completed']
        

        if start_chunk is not None:
            pending_chunks = [c for c in pending_chunks if c.chunk_id >= start_chunk]
        
        if end_chunk is not None:
            pending_chunks = [c for c in pending_chunks if c.chunk_id < end_chunk]
        
        
        if not pending_chunks:
            return
        

        for chunk_info in pending_chunks:
            self._process_single_chunk(chunk_info)
            self._save_chunk_manifest()
        
        phase_time = time.time() - start_time
        successful_chunks = len([c for c in self.chunks_info if c.status == 'completed'])

    def _process_single_chunk(self, chunk_info: ChunkInfo):
        chunk_start_time = time.time()
        
        try:
            chunk_info.status = 'processing'
            

            chunk_data = self._load_chunk_data(chunk_info)
            
            if not chunk_data:
                raise ValueError("No valid nodes created from chunk")
            
            

            self._generate_chunk_embeddings(chunk_data)
            

            embeddings_file = self.embeddings_dir / f""
            self._save_chunk_embeddings(chunk_data, embeddings_file)
            

            chunk_info.embeddings_file = str(embeddings_file.name)
            chunk_info.status = 'completed'
            chunk_info.processing_time = time.time() - chunk_start_time
            

            del chunk_data
            self._clear_gpu_memory()
            gc.collect()
            
            
        except Exception as e:
            chunk_info.status = 'failed'
            chunk_info.error_message = str(e)
            chunk_info.processing_time = time.time() - chunk_start_time
            

            self._clear_gpu_memory()
            gc.collect()

    def _load_chunk_data(self, chunk_info: ChunkInfo) -> List[MAGChunkData]:


        node_items = self.relevant_nodes[chunk_info.start_idx:chunk_info.end_idx]
        
        all_chunks = []
        

        batch_size = 5000
        for i in range(0, len(node_items), batch_size):
            batch = node_items[i:i+batch_size]
            

            with ThreadPoolExecutor(max_workers=min(16, len(batch))) as executor:
                future_to_node = {
                    executor.submit(self._process_node_to_chunk, obj_id, obj_data): obj_id
                    for obj_id, obj_data in batch
                }
                
                for future in tqdm(as_completed(future_to_node), total=len(future_to_node), desc=f"   Processing batch"):
                    try:
                        chunk = future.result()
                        if chunk:
                            all_chunks.append(chunk)
                    except Exception as e:
                        obj_id = future_to_node[future]
        
        return all_chunks

    def _process_node_to_chunk(self, obj_id: str, obj_data: Dict[str, Any]) -> Optional[MAGChunkData]:
        try:
            obj_type = obj_data.get('type')
            
            if obj_type == 'paper':
                return self._create_paper_chunk(obj_id, obj_data)
            elif obj_type == 'author':
                return self._create_author_chunk(obj_id, obj_data)
            else:
                return None
                
        except Exception as e:
            return None

    def _create_paper_chunk(self, obj_id: str, obj_data: Dict[str, Any]) -> Optional[MAGChunkData]:
        try:

            original_title = obj_data.get('OriginalTitle', '')
            publisher = obj_data.get('Publisher', '')
            abstract = obj_data.get('abstract', '')
            authors = obj_data.get('authors', [])
            fields_of_study = obj_data.get('fields_of_study', [])
            cites = obj_data.get('cites', [])
            

            content_parts = []
            
            if original_title and original_title.strip():
                content_parts.append(original_title.strip())
            
            if publisher and publisher.strip():
                content_parts.append(publisher.strip())
            
            if abstract and abstract.strip():
                content_parts.append(abstract.strip())
            
            if authors and isinstance(authors, list):
                authors_str = ' '.join([str(a) for a in authors if str(a).strip()])
                if authors_str.strip():
                    content_parts.append(authors_str.strip())
            
            if fields_of_study and isinstance(fields_of_study, list):
                fields_str = ' '.join([str(f) for f in fields_of_study if str(f).strip()])
                if fields_str.strip():
                    content_parts.append(fields_str.strip())
            
            if cites and isinstance(cites, list):
                cites_str = ' '.join([str(c) for c in cites if str(c).strip()])
                if cites_str.strip():
                    content_parts.append(cites_str.strip())
            
            content = ' '.join(content_parts)
            

            chunk_data = MAGChunkData(
                chunk_id=f"paper_{obj_id}",
                node_type="paper",
                object_id=obj_id,
                content=content,
                original_title=original_title,
                publisher=publisher,
                abstract=abstract,
                authors=authors,
                fields_of_study=fields_of_study,
                cites=cites,
                metadata={
                    'original_object_id': obj_id,
                    'year': obj_data.get('Year', ''),
                    'paper_rank': obj_data.get('PaperRank', ''),
                    'citation_count': obj_data.get('PaperCitationCount', 0),
                    'reference_count': obj_data.get('ReferenceCount', 0)
                }
            )
            
            return chunk_data
            
        except Exception as e:
            return None

    def _create_author_chunk(self, obj_id: str, obj_data: Dict[str, Any]) -> Optional[MAGChunkData]:
        try:

            display_name = obj_data.get('DisplayName', '')
            institution = obj_data.get('institution', '')
            

            content_parts = []
            
            if display_name and display_name.strip():
                content_parts.append(display_name.strip())
            
            if institution and institution.strip():
                content_parts.append(institution.strip())
            
            content = ' '.join(content_parts)
            

            chunk_data = MAGChunkData(
                chunk_id=f"author_{obj_id}",
                node_type="author",
                object_id=obj_id,
                content=content,
                display_name=display_name,
                institution=institution,
                metadata={
                    'original_object_id': obj_id,
                    'rank': obj_data.get('Rank', ''),
                    'paper_count': obj_data.get('PaperCount', 0),
                    'citation_count': obj_data.get('CitationCount', 0)
                }
            )
            
            return chunk_data
            
        except Exception as e:
            return None

    def _generate_chunk_embeddings(self, chunk_data: List[MAGChunkData]):

        all_text_data = self._prepare_chunk_texts(chunk_data)
        

        for field_type, data in all_text_data.items():
            if not data['texts']:
                continue
                
            field_start = time.time()
            

            self._check_gpu_memory_before_processing(field_type, len(data['texts']))
            

            if hasattr(self.embedding_service, '_clear_gpu_cache'):
                self.embedding_service._clear_gpu_cache()
            
            try:
                embeddings = self.embedding_service.generate_embeddings_bulk(data['texts'])
                
                field_time = time.time() - field_start
                

                self._assign_embeddings_to_chunks(field_type, data['chunk_indices'], embeddings, chunk_data)
                

                del embeddings
                self._aggressive_gpu_cleanup()
                    
            except Exception as e:

                self._aggressive_gpu_cleanup()
                continue

    def _prepare_chunk_texts(self, chunk_data: List[MAGChunkData]) -> Dict[str, Dict[str, List]]:
        all_text_data = {
            'content': {'texts': [], 'chunk_indices': []},
            'original_title': {'texts': [], 'chunk_indices': []},
            'abstract': {'texts': [], 'chunk_indices': []},
            'authors': {'texts': [], 'chunk_indices': []},
            'fields_of_study': {'texts': [], 'chunk_indices': []},
            'cites': {'texts': [], 'chunk_indices': []},
            'display_name': {'texts': [], 'chunk_indices': []},
            'institution': {'texts': [], 'chunk_indices': []}
        }
        
        for chunk_idx, chunk in enumerate(chunk_data):

            if chunk.content and chunk.content.strip():
                all_text_data['content']['texts'].append(chunk.content.strip())
                all_text_data['content']['chunk_indices'].append(chunk_idx)
            

            if chunk.node_type == 'paper':
                self._collect_paper_field_texts(chunk, chunk_idx, all_text_data)
            elif chunk.node_type == 'author':
                self._collect_author_field_texts(chunk, chunk_idx, all_text_data)
        
        return all_text_data

    def _collect_paper_field_texts(self, chunk: MAGChunkData, chunk_idx: int, all_text_data: Dict):

        if chunk.original_title and chunk.original_title.strip():
            all_text_data['original_title']['texts'].append(chunk.original_title.strip())
            all_text_data['original_title']['chunk_indices'].append(chunk_idx)
        

        if chunk.abstract and chunk.abstract.strip():
            all_text_data['abstract']['texts'].append(chunk.abstract.strip())
            all_text_data['abstract']['chunk_indices'].append(chunk_idx)
        

        if chunk.authors and isinstance(chunk.authors, list):
            authors_str = ' '.join([str(a) for a in chunk.authors if str(a).strip()])
            if authors_str.strip():
                all_text_data['authors']['texts'].append(authors_str.strip())
                all_text_data['authors']['chunk_indices'].append(chunk_idx)
        

        if chunk.fields_of_study and isinstance(chunk.fields_of_study, list):
            fields_str = ' '.join([str(f) for f in chunk.fields_of_study if str(f).strip()])
            if fields_str.strip():
                all_text_data['fields_of_study']['texts'].append(fields_str.strip())
                all_text_data['fields_of_study']['chunk_indices'].append(chunk_idx)
        

        if chunk.cites and isinstance(chunk.cites, list):
            cites_str = ' '.join([str(c) for c in chunk.cites if str(c).strip()])
            if cites_str.strip():
                all_text_data['cites']['texts'].append(cites_str.strip())
                all_text_data['cites']['chunk_indices'].append(chunk_idx)

    def _collect_author_field_texts(self, chunk: MAGChunkData, chunk_idx: int, all_text_data: Dict):

        if chunk.display_name and chunk.display_name.strip():
            all_text_data['display_name']['texts'].append(chunk.display_name.strip())
            all_text_data['display_name']['chunk_indices'].append(chunk_idx)
        

        if chunk.institution and chunk.institution.strip():
            all_text_data['institution']['texts'].append(chunk.institution.strip())
            all_text_data['institution']['chunk_indices'].append(chunk_idx)

    def _assign_embeddings_to_chunks(self, field_type: str, chunk_indices: List[int], embeddings: List[List[float]], chunk_data: List[MAGChunkData]):
        embedding_attr_map = {
            'content': 'content_embedding',
            'original_title': 'original_title_embedding',
            'abstract': 'abstract_embedding',
            'authors': 'authors_embedding',
            'fields_of_study': 'fields_of_study_embedding',
            'cites': 'cites_embedding',
            'display_name': 'display_name_embedding',
            'institution': 'institution_embedding'
        }
        
        embedding_attr = embedding_attr_map[field_type]
        
        for chunk_idx, embedding in zip(chunk_indices, embeddings):
            setattr(chunk_data[chunk_idx], embedding_attr, embedding)

    def _save_chunk_embeddings(self, chunk_data: List[MAGChunkData], embeddings_file: Path):
        embeddings_data = []
        
        for chunk in chunk_data:
            chunk_embeddings = {
                'chunk_id': chunk.chunk_id,
                'node_type': chunk.node_type,
                'object_id': chunk.object_id,
                'content': chunk.content,
                'content_embedding': chunk.content_embedding,
                'metadata': chunk.metadata
            }
            

            if chunk.node_type == 'paper':
                chunk_embeddings.update({
                    'original_title': chunk.original_title,
                    'publisher': chunk.publisher,
                    'abstract': chunk.abstract,
                    'authors': chunk.authors,
                    'fields_of_study': chunk.fields_of_study,
                    'cites': chunk.cites,

                    'original_title_embedding': chunk.original_title_embedding,
                    'abstract_embedding': chunk.abstract_embedding,
                    'authors_embedding': chunk.authors_embedding,
                    'fields_of_study_embedding': chunk.fields_of_study_embedding,
                    'cites_embedding': chunk.cites_embedding
                })
            elif chunk.node_type == 'author':
                chunk_embeddings.update({
                    'display_name': chunk.display_name,
                    'institution': chunk.institution,

                    'display_name_embedding': chunk.display_name_embedding,
                    'institution_embedding': chunk.institution_embedding
                })
            
            embeddings_data.append(chunk_embeddings)
        
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)

    def phase3_merge_and_build_index(self):
        start_time = time.time()
        

        hnsw_index_file = self.final_dir / 'hnsw_index.faiss'
        hnsw_mapping_file = self.final_dir / 'hnsw_mapping.pkl'
        
        if hnsw_index_file.exists() and hnsw_mapping_file.exists():
            return
        

        all_embeddings = []
        object_to_embedding_mapping = []
        
        completed_chunks = [c for c in self.chunks_info if c.status == 'completed']
        
        for chunk_info in tqdm(completed_chunks, desc="   Loading embeddings"):
            embeddings_file = self.embeddings_dir / chunk_info.embeddings_file
            
            with open(embeddings_file, 'r') as f:
                chunk_embeddings = json.load(f)
            
            for chunk_emb in chunk_embeddings:
                if chunk_emb['content_embedding'] and any(x != 0.0 for x in chunk_emb['content_embedding']):
                    all_embeddings.append(chunk_emb['content_embedding'])
                    object_to_embedding_mapping.append({
                        'object_id': chunk_emb['object_id'],
                        'node_type': chunk_emb['node_type'],
                        'chunk_id': chunk_emb['chunk_id']
                    })
        
        

        if not all_embeddings:
            raise ValueError("No valid embeddings found for HNSW index")
        
        embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
        dimension = embeddings_matrix.shape[1]
        
        

        faiss_index = faiss.IndexHNSWFlat(dimension, 64)  
        faiss_index.hnsw.efConstruction = 2000  
        faiss_index.hnsw.efSearch = 1000  
        

        faiss.normalize_L2(embeddings_matrix)
        

        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                gpu_index = faiss.index_cpu_to_all_gpus(faiss_index)
                gpu_index.add(embeddings_matrix)
                faiss_index = faiss.index_gpu_to_cpu(gpu_index)
                del gpu_index
                self._clear_gpu_memory()
            except Exception as e:
                faiss_index.add(embeddings_matrix)
        else:
            faiss_index.add(embeddings_matrix)
        

        faiss.write_index(faiss_index, str(hnsw_index_file))
        with open(hnsw_mapping_file, 'wb') as f:
            pickle.dump(object_to_embedding_mapping, f)
        
        phase_time = time.time() - start_time

    def phase4_generate_analysis(self):
        start_time = time.time()
        

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        analysis_file = self.final_dir / f""
        
        analysis_data = {
            'timestamp': timestamp,
            'total_nodes': self.total_nodes,
            'paper_count': self.paper_count,
            'author_count': self.author_count,
            'total_chunks': len(self.chunks_info),
            'completed_chunks': len([c for c in self.chunks_info if c.status == 'completed']),
            'total_processing_time': sum(c.processing_time or 0 for c in self.chunks_info),
            'chunk_details': [
                {
                    'chunk_id': c.chunk_id,
                    'node_count': c.node_count,
                    'status': c.status,
                    'processing_time': c.processing_time,
                    'embeddings_file': c.embeddings_file
                }
                for c in self.chunks_info
            ]
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        phase_time = time.time() - start_time
        
        return analysis_file

    def _save_chunk_manifest(self):
        manifest_data = {
            'total_nodes': self.total_nodes,
            'paper_count': self.paper_count,
            'author_count': self.author_count,
            'chunk_size': self.chunk_size,
            'total_chunks': len(self.chunks_info),
            'chunks': [
                {
                    'chunk_id': c.chunk_id,
                    'start_idx': c.start_idx,
                    'end_idx': c.end_idx,
                    'node_count': c.node_count,
                    'status': c.status,
                    'embeddings_file': c.embeddings_file,
                    'processing_time': c.processing_time,
                    'error_message': c.error_message
                }
                for c in self.chunks_info
            ]
        }
        
        with open(self.chunk_manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    def _load_chunk_manifest(self):
        if not self.chunk_manifest_file.exists():
            return
        
        with open(self.chunk_manifest_file, 'r') as f:
            manifest_data = json.load(f)
        
        self.total_nodes = manifest_data['total_nodes']
        self.paper_count = manifest_data['paper_count']
        self.author_count = manifest_data['author_count']
        self.chunks_info = []
        
        for chunk_data in manifest_data['chunks']:
            chunk_info = ChunkInfo(
                chunk_id=chunk_data['chunk_id'],
                start_idx=chunk_data['start_idx'],
                end_idx=chunk_data['end_idx'],
                node_count=chunk_data['node_count'],
                status=chunk_data['status'],
                embeddings_file=chunk_data.get('embeddings_file'),
                processing_time=chunk_data.get('processing_time'),
                error_message=chunk_data.get('error_message')
            )
            self.chunks_info.append(chunk_info)

    def _clear_gpu_memory(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        gc.collect()

    def _aggressive_gpu_cleanup(self):
        

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        

        if hasattr(self.embedding_service, '_clear_gpu_cache'):
            self.embedding_service._clear_gpu_cache()
        

        gc.collect()
        

        if hasattr(self.embedding_service, 'model') and hasattr(self.embedding_service.model, 'module'):

            for param in self.embedding_service.model.parameters():
                if param.grad is not None:
                    param.grad = None
        

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        

    def _check_gpu_memory_before_processing(self, field_type: str, num_texts: int):
        if not torch.cuda.is_available():
            return
            

        total_free_memory = 0
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                total_free_memory += free_memory
        

        estimated_memory_needed = num_texts * 100 * 4
        
        
        if estimated_memory_needed > total_free_memory * 0.8:

def main():

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    
    pipeline = MAGChunkedPipeline(chunk_size=100000)
    
    try:
        analysis_file = pipeline.run_chunked_pipeline()
        
        
    except Exception as e:
        raise

if __name__ == "__main__":
    main()
