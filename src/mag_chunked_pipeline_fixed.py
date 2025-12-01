#!/usr/bin/env python3

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import torch
import gc


sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mag_chunked_pipeline import MAGChunkedPipeline

@dataclass
class ChunkProgress:
    completed_chunks: List[int]
    failed_chunks: List[int]
    current_batch: int
    total_chunks: int

class MAGChunkedPipelineFixed:
    
    def __init__(self, cache_dir: str = ""):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.cache_dir / ""
        self.chunks_per_batch = 10
        

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        
    
    def load_progress(self) -> ChunkProgress:
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                return ChunkProgress(**data)
            except Exception as e:
        
        return ChunkProgress(
            completed_chunks=[],
            failed_chunks=[],
            current_batch=0,
            total_chunks=0
        )
    
    def save_progress(self, progress: ChunkProgress):
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(asdict(progress), f, indent=2)
        except Exception as e:
    
    def run_single_batch(self, start_chunk: int, end_chunk: int) -> bool:
        

        pipeline = MAGChunkedPipeline(
            cache_dir=str(self.cache_dir),
            chunk_size=50000,
            use_gpu=True,
            num_threads=8
        )
        
        try:

            success = pipeline.run_chunked_pipeline(
                start_chunk=start_chunk,
                end_chunk=end_chunk
            )
            
            if success:
                return True
            else:
                return False
                
        except Exception as e:
            return False
        finally:

            self._ultra_aggressive_cleanup()
    
    def _ultra_aggressive_cleanup(self):
        

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    torch.cuda.reset_peak_memory_stats()
        

        for _ in range(5):
            gc.collect()
            time.sleep(0.1)
        

        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
        
    
    def run_fixed_pipeline(self, max_chunks: Optional[int] = None):
        

        progress = self.load_progress()
        

        if max_chunks:
            total_chunks = max_chunks
        else:

            pipeline = MAGChunkedPipeline(
                cache_dir=str(self.cache_dir),
                chunk_size=50000,
                use_gpu=True,
                num_threads=8
            )
            total_chunks = pipeline.get_total_chunks()
        
        progress.total_chunks = total_chunks
        
        

        current_chunk = 0
        batch_number = 0
        
        while current_chunk < total_chunks:
            end_chunk = min(current_chunk + self.chunks_per_batch, total_chunks)
            
            

            success = self.run_single_batch(current_chunk, end_chunk)
            
            if success:

                for chunk_id in range(current_chunk, end_chunk):
                    if chunk_id not in progress.completed_chunks:
                        progress.completed_chunks.append(chunk_id)
                
            else:

                for chunk_id in range(current_chunk, end_chunk):
                    if chunk_id not in progress.failed_chunks:
                        progress.failed_chunks.append(chunk_id)
                
            

            progress.current_batch = batch_number + 1
            self.save_progress(progress)
            

            current_chunk = end_chunk
            batch_number += 1
            

            time.sleep(2)
        

        
        if progress.failed_chunks:

def main():
    

    pipeline = MAGChunkedPipelineFixed()
    

    pipeline.run_fixed_pipeline()

if __name__ == "__main__":
    main()

