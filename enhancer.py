import torch
import comfy.model_management as mm
import gc

class ConditioningEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        devices = ["auto", "cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")

        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "enhance_strength": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 2.0, "step": 0.05}),
                "normalize": ("BOOLEAN", {"default": True}),
                "add_self_attention": ("BOOLEAN", {"default": False}),
                "mlp_hidden_mult": ("INT", {"default": 2, "min": 1, "max": 200, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "step": 1}),
                "low_vram": ("BOOLEAN", {"default": False}),
                "device": (devices, {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "enhance"
    CATEGORY = "conditioning/enhance/capitan"

    def enhance(self, conditioning, enhance_strength, normalize, add_self_attention, mlp_hidden_mult, seed, low_vram, device):
        if not conditioning:
            return (conditioning, )

        if device == "auto":
            device = mm.get_torch_device()
        else:
            device = torch.device(device)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        compute_dtype = torch.float16 if low_vram and device.type == "cuda" else torch.float32

        enhanced = []

        for cond, meta in conditioning:
            emb = cond.to(device, dtype=compute_dtype)
            original_dtype = cond.dtype

            if normalize:
                mean = emb.mean(dim=-1, keepdim=True)
                std = emb.std(dim=-1, keepdim=True) + 1e-6
                emb = (emb - mean) / std

            dim = emb.shape[-1]
            hidden_dim = dim * mlp_hidden_mult

            mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, hidden_dim, device=device, dtype=compute_dtype),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, dim, device=device, dtype=compute_dtype)
            )

            torch.nn.init.kaiming_uniform_(mlp[0].weight, nonlinearity='relu')
            torch.nn.init.zeros_(mlp[0].bias)
            torch.nn.init.eye_(mlp[2].weight)
            torch.nn.init.zeros_(mlp[2].bias)

            refined = mlp(emb)
            blended = emb + enhance_strength * (refined - emb)

            if add_self_attention and not low_vram:
                attn = torch.nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=8,
                    batch_first=True,
                    device=device,
                    dtype=compute_dtype
                )
                attn_out, _ = attn(blended, blended, blended)
                blended = blended + 0.3 * attn_out
                del attn, attn_out

            enhanced.append((blended.to("cpu", dtype=original_dtype), meta))

            del mlp, refined, blended, emb

            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        return (enhanced, )

NODE_CLASS_MAPPINGS = {"ConditioningEnhancer": ConditioningEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"ConditioningEnhancer": "Capitan Conditioning Enhancer (basic)"}

