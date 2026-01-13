import torch
import comfy.model_management as mm

class ConditioningEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "enhance_strength": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 2.0, "step": 0.05}),
                "normalize": ("BOOLEAN", {"default": True}),
                "add_self_attention": ("BOOLEAN", {"default": False}),
                "mlp_hidden_mult": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "step": 1}),  # NEW: seed control
            }
        }

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "enhance"
    CATEGORY = "conditioning/enhance"

    def enhance(self, conditioning, enhance_strength, normalize, add_self_attention, mlp_hidden_mult, seed):
        if not conditioning:
            return (conditioning, )

        device = mm.get_torch_device()
        enhanced = []

        # NEW: Set seed for reproducible init
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        for emb_tensor, meta in conditioning:
            emb = emb_tensor.to(device)  # [B, seq, 2560]

            orig_dtype = emb.dtype

            if normalize:
                emb = (emb - emb.mean(dim=-1, keepdim=True)) / (emb.std(dim=-1, keepdim=True) + 1e-6)

            dim = emb.shape[-1]
            hidden_dim = dim * mlp_hidden_mult

            mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, hidden_dim, device=device, dtype=emb.dtype),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, dim, device=device, dtype=emb.dtype)
            ).to(device)

            torch.nn.init.kaiming_uniform_(mlp[0].weight, nonlinearity='relu')
            torch.nn.init.zeros_(mlp[0].bias)
            torch.nn.init.eye_(mlp[2].weight)
            torch.nn.init.zeros_(mlp[2].bias)

            refined = mlp(emb)

            blended = emb + enhance_strength * (refined - emb)

            if add_self_attention:
                attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True, device=device, dtype=emb.dtype)
                attn_out, _ = attn(blended, blended, blended)
                blended = blended + 0.3 * attn_out

            enhanced.append((blended.to("cpu", dtype=orig_dtype), meta))

        return (enhanced, )

NODE_CLASS_MAPPINGS = {"ConditioningEnhancer": ConditioningEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"ConditioningEnhancer": "Conditioning Enhancer (Qwen/Z-Image)"}
