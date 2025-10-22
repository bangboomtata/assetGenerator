import os
import torch
import random
import numpy as np
from PIL import Image
from typing import List
import huggingface_hub
from omegaconf import OmegaConf
from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler, UniPCMultistepScheduler


class multiviewDiffusionNet:
    def __init__(self, config) -> None:
        self.device = config.device

        cfg_path = config.multiview_cfg_path
        custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr")
        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg
        self.mode = self.cfg.model.params.stable_diffusion_config.custom_pipeline[2:]

        model_path = huggingface_hub.snapshot_download(
            repo_id=config.multiview_pretrained_path,
            allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
        )

        model_path = os.path.join(model_path, "hunyuan3d-paintpbr-v2-1")
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            custom_pipeline=custom_pipeline, 
            torch_dtype=torch.float16
        )

        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
        pipeline.set_progress_bar_config(disable=True)
        pipeline.eval()
        setattr(pipeline, "view_size", cfg.model.params.get("view_size", 320))
        
        self.device = torch.device("xpu:1" if torch.xpu.is_available() and torch.xpu.device_count() > 1 else "cpu")
        
        self.pipeline = pipeline.to(self.device)
        
        # Patch the pipeline's convert_pil_list_to_tensor function to use our device
        self._patch_pipeline_device()

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from hunyuanpaintpbr.unet.modules import Dino_v2
            self.dino_v2 = Dino_v2(config.dino_ckpt_path).to(torch.float16)
            self.dino_v2 = self.dino_v2.to(self.device)

    def _patch_pipeline_device(self):
        """Patch the pipeline to use the correct device instead of hardcoded 'cuda'"""
        print("\n" + "="*80)
        print("DEBUG: Starting _patch_pipeline_device()")
        print(f"DEBUG: Target device = {self.device}")
        print(f"DEBUG: Found issue: Line 245 in pipeline.py has hardcoded .to('cuda')")
        print("="*80)
        
        import sys
        device = self.device
        
        # Get the pipeline file path
        module_name = self.pipeline.__class__.__module__
        pipeline_module = sys.modules[module_name]
        pipeline_file = pipeline_module.__file__
        print(f"DEBUG: Pipeline file: {pipeline_file}")
        
        # Wrap the __call__ method to intercept cuda calls
        print("DEBUG: Wrapping pipeline.__call__ to intercept .to('cuda') calls...")
        original_call = self.pipeline.__call__
        
        # Counter for debugging
        cuda_intercept_count = [0]
        
        def patched_call(*args, **kwargs):
            print(f"\nDEBUG: ►►► Pipeline.__call__ invoked ◄◄◄")
            
            # Temporarily replace torch.Tensor.to to intercept cuda calls
            original_to = torch.Tensor.to
            
            def patched_to(self_tensor, *to_args, **to_kwargs):
                # Check if trying to move to 'cuda'
                if len(to_args) > 0 and to_args[0] == "cuda":
                    cuda_intercept_count[0] += 1
                    print(f"DEBUG: ✓✓✓ INTERCEPTED .to('cuda') call #{cuda_intercept_count[0]}!")
                    print(f"DEBUG:     Tensor shape: {self_tensor.shape}, dtype: {self_tensor.dtype}")
                    print(f"DEBUG:     Redirecting from 'cuda' → '{device}'")
                    return original_to(self_tensor, device, *to_args[1:], **to_kwargs)
                elif 'device' in to_kwargs and to_kwargs['device'] == "cuda":
                    cuda_intercept_count[0] += 1
                    print(f"DEBUG: ✓✓✓ INTERCEPTED .to(device='cuda') call #{cuda_intercept_count[0]}!")
                    print(f"DEBUG:     Redirecting from 'cuda' → '{device}'")
                    to_kwargs['device'] = device
                    return original_to(self_tensor, *to_args, **to_kwargs)
                else:
                    # Normal call, not to cuda
                    return original_to(self_tensor, *to_args, **to_kwargs)
            
            # Monkey-patch torch.Tensor.to temporarily
            torch.Tensor.to = patched_to
            
            try:
                print("DEBUG: Executing original pipeline call...")
                result = original_call(*args, **kwargs)
                print(f"DEBUG: ✓✓✓ Pipeline call completed successfully!")
                print(f"DEBUG: Total cuda calls intercepted: {cuda_intercept_count[0]}")
                return result
            except Exception as e:
                print(f"DEBUG: ✗✗✗ Pipeline call failed with error: {e}")
                import traceback
                traceback.print_exc()
                raise
            finally:
                # Restore original method
                torch.Tensor.to = original_to
                print("DEBUG: Restored original torch.Tensor.to method\n")
        
        self.pipeline.__call__ = patched_call
        print("DEBUG: ✓✓✓ Successfully wrapped pipeline.__call__ method")
        print("DEBUG: The wrapper will intercept all .to('cuda') calls during execution")
        
        print("="*80)
        print("DEBUG: Patch setup complete!")
        print("="*80 + "\n")

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    @torch.no_grad()
    def __call__(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False):
        pils = self.forward_one(
            images, conditions, prompt=prompt, custom_view_size=custom_view_size, resize_input=resize_input
        )
        return pils

    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False):
        self.seed_everything(0)
        custom_view_size = custom_view_size if custom_view_size is not None else self.pipeline.view_size
        if not isinstance(input_images, List):
            input_images = [input_images]
        if not resize_input:
            input_images = [
                input_image.resize((self.pipeline.view_size, self.pipeline.view_size)) for input_image in input_images
            ]
        else:
            input_images = [input_image.resize((custom_view_size, custom_view_size)) for input_image in input_images]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((custom_view_size, custom_view_size))
            if control_images[i].mode == "L":
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode="1")
        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        kwargs["width"] = custom_view_size
        kwargs["height"] = custom_view_size
        kwargs["num_in_batch"] = num_view
        kwargs["images_normal"] = normal_image
        kwargs["images_position"] = position_image

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            dino_hidden_states = self.dino_v2(input_images[0])
            kwargs["dino_hidden_states"] = dino_hidden_states

        sync_condition = None

        infer_steps_dict = {
            "EulerAncestralDiscreteScheduler": 30,
            "UniPCMultistepScheduler": 15,
            "DDIMScheduler": 50,
            "ShiftSNRScheduler": 15,
        }

        mvd_image = self.pipeline(
            input_images[0:1],
            num_inference_steps=infer_steps_dict[self.pipeline.scheduler.__class__.__name__],
            prompt=prompt,
            sync_condition=sync_condition,
            guidance_scale=3.0,
            **kwargs,
        ).images

        if "pbr" in self.mode:
            mvd_image = {"albedo": mvd_image[:num_view], "mr": mvd_image[num_view:]}
            # mvd_image = {'albedo':mvd_image[:num_view]}
        else:
            mvd_image = {"hdr": mvd_image}

        return mvd_image