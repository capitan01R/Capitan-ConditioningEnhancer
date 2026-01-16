import torch
import comfy.model_management as mm
import gc

class CapitanAdvancedEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        devices = ["auto", "cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")

        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "enhance_strength": ("FLOAT", {"default": 0.05, "min": -3.0, "max": 2.0, "step": 0.01}),
                "detail_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "preserve_original": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "attention_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "high_pass_filter": ("BOOLEAN", {"default": False}),
                "normalize": ("BOOLEAN", {"default": True}),
                "add_self_attention": ("BOOLEAN", {"default": False}),
                "mlp_hidden_mult": ("INT", {"default": 8, "min": 1, "max": 200, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "step": 1}),
                "low_vram": ("BOOLEAN", {"default": False}),
                "device": (devices, {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "enhance"
    CATEGORY = "conditioning/enhance/capitan"

    def enhance(self, conditioning, enhance_strength, detail_boost, preserve_original,
                attention_strength, high_pass_filter, normalize, add_self_attention,
                mlp_hidden_mult, seed, low_vram, device):

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

            if detail_boost > 1.0:
                detail_map = refined - emb
                detail_map = torch.tanh(detail_map * (detail_boost - 1.0))
                refined = emb + detail_map

            low_pass = None
            if high_pass_filter:
                low_pass = torch.nn.functional.avg_pool1d(
                    refined.transpose(1, 2), kernel_size=3, stride=1, padding=1
                ).transpose(1, 2)
                refined = refined + 0.4 * (refined - low_pass)

            residual_scale = 1.0 / (1.0 + mlp_hidden_mult * 0.05)
            blended = emb + (enhance_strength * residual_scale) * (refined - emb)

            blended = blended * (1 - preserve_original) + emb * preserve_original

            if add_self_attention:
                attn = torch.nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=8,
                    batch_first=True,
                    device=device,
                    dtype=compute_dtype
                )
                attn_out, _ = attn(blended, blended, blended)
                blended = blended + attention_strength * attn_out
                del attn, attn_out

            enhanced.append((blended.to("cpu", dtype=original_dtype), meta))

            # Clean up
            del mlp, refined, blended, emb

            if low_pass is not None:
                del low_pass

            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        return (enhanced, )

NODE_CLASS_MAPPINGS = {"CapitanAdvancedEnhancer": CapitanAdvancedEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"CapitanAdvancedEnhancer": "Capitan Advanced Enhancer"}

