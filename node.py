import torch
import comfy.model_management
import comfy.samplers

class SchedulerMixer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "normal": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "karras": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponential": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sgm_uniform": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "simple": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ddim_uniform": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, denoise, normal, karras, exponential, sgm_uniform, simple, ddim_uniform):
        total_steps = steps
        if denoise < 1.0:
            total_steps = int(steps / denoise)

        comfy.model_management.load_models_gpu([model])

        scheduler_weights = [normal, karras, exponential, sgm_uniform, simple, ddim_uniform]
        scheduler_names = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]

        mixed_sigmas = torch.zeros((steps + 1,), device="cpu", dtype=torch.float)
        for weight, name in zip(scheduler_weights, scheduler_names):                        
            if weight > 0.0:                
                sigmas = comfy.samplers.calculate_sigmas_scheduler(model.model, name, total_steps).cpu()
                sigmas = sigmas[-(steps + 1):]                
                mixed_sigmas += sigmas * weight

        return (mixed_sigmas,)
    
