import torch
import torch.multiprocessing as mp
from queue import Empty
import time
import gc

def texture_worker(rank, task_queue, result_queue, device_tex_base):
    pipeline = None
    task_id = None

    """Each subprocess runs texture generation on a different XPU"""
    try:
        import torch
        
        # Ensure XPU is properly initialized in this process
        if not torch.xpu.is_available():
            raise RuntimeError("XPU not available in worker process")
            
        device = torch.device(f"xpu:{device_tex_base + rank}")
        print(f"Texture worker {rank} starting on device: {device}")
        
        # Critical: Set device before any XPU operations
        torch.xpu.set_device(device)
        
        # Initialize XPU context for this process
        torch.xpu.init()
        torch.xpu.empty_cache()
        
        from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        
        # Create config and pipeline on the specific device
        conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768, device=device)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        
        # Ensure pipeline is created on correct device
        with torch.xpu.device(device):
            pipeline = Hunyuan3DPaintPipeline(conf)
        
        print(f"Texture worker {rank} initialized successfully on {device}")
        
        while True:
            try:
                task = task_queue.get(timeout=1.0)
                
                if task is None:
                    print(f"Texture worker {rank} received exit signal")
                    break
                
                task_id, mesh_path, image_path, output_path = task
                print(f"Worker {rank} processing task {task_id}")
                
                # Ensure we're on the right device for processing
                torch.xpu.set_device(device)
                
                start_time = time.time()
                with torch.xpu.device(device):
                    result_path = pipeline(
                        mesh_path=mesh_path, 
                        image_path=image_path, 
                        output_mesh_path=output_path, 
                        save_glb=False
                    )
                processing_time = time.time() - start_time
                
                result_queue.put((task_id, result_path, processing_time, None))
                print(f"Worker {rank} completed task {task_id} in {processing_time:.2f}s")
                
            except Empty:
                continue
            except Exception as e:
                error_msg = f"Worker {rank} task {task_id} error: {e}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                result_queue.put((task_id, None, 0, error_msg))
                
    except Exception as e:
        error_msg = f"Worker {rank} initialization failed: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        if task_id is not None:
            result_queue.put((task_id, None, 0, error_msg))

class MultiXPUTextureManager:
    def __init__(self, num_workers=2, device_tex_base=0):
        self.num_workers = num_workers
        self.device_tex_base = device_tex_base
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.processes = []
        self.task_counter = 0
        self.running = False
        torch.xpu.synchronize()
        gc.collect()
        torch.xpu.empty_cache()
        torch.xpu._lazy_init()
        torch.xpu._xpu_device_context = None
        gc.collect()
        torch.xpu.init()
    
    def start_workers(self):
        """Start all texture worker processes"""
        if self.running:
            return
            
        print(f"Starting {self.num_workers} texture workers...")
        for rank in range(self.num_workers):
            p = mp.Process(
                target=texture_worker, 
                args=(rank, self.task_queue, self.result_queue, self.device_tex_base)
            )
            p.daemon = True  # Dies with main process
            p.start()
            self.processes.append(p)
        
        self.running = True
        print("All texture workers started")
    
    def submit_task(self, mesh_path, image_path, output_path):
        """Submit a texture generation task"""
        if not self.running:
            raise RuntimeError("Workers not started")
            
        task_id = self.task_counter
        self.task_counter += 1
        self.task_queue.put((task_id, mesh_path, image_path, output_path))
        return task_id
    
    def get_result(self, timeout=300):
        """Get a completed result"""
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def shutdown(self):
        """Shutdown all workers"""
        if not self.running:
            return
            
        print("Shutting down texture workers...")
        # Send sentinel values
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        # Wait for processes to finish
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                print(f"Force terminating worker {p.pid}")
                p.terminate()
                p.join()
        
        self.running = False
        print("All texture workers shut down")