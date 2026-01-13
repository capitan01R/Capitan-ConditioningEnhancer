import torch
import comfy.model_management as mm

class CapitanAdvancedEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "enhance_strength": ("FLOAT", {"default": 0.05, "min": -3.0, "max": 2.0, "step": 0.01}),
                "detail_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),  # NEW: extra sharpness control
                "preserve_original": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),  # NEW: stronger skip
                "attention_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),  # NEW: tunable attention
                "high_pass_filter": ("BOOLEAN", {"default": False}),  # NEW: edge emphasis
                "normalize": ("BOOLEAN", {"default": True}),
                "add_self_attention": ("BOOLEAN", {"default": False}),
                "mlp_hidden_mult": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "step": 1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "enhance"
    CATEGORY = "conditioning/enhance/capitan"

    def enhance(self, conditioning, enhance_strength, detail_boost, preserve_original, attention_strength, high_pass_filter, normalize, add_self_attention, mlp_hidden_mult, seed):
        if not conditioning:
            return (conditioning, )

        device = mm.get_torch_device()
        enhanced = []

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

            # NEW: Gentle detail boost (default 1.0 = no boost)
            if detail_boost > 1.0:
                detail_map = refined - emb
                detail_map = torch.tanh(detail_map * (detail_boost - 1.0))
                refined = emb + detail_map

            # NEW: Gentle high-pass (off by default)
            if high_pass_filter:
                low_pass = torch.nn.functional.avg_pool1d(refined.transpose(1,2), kernel_size=3, stride=1, padding=1).transpose(1,2)
                refined = refined + 0.4 * (refined - low_pass)

            # Blend with light residual dampening (safer than basic)
            residual_scale = 1.0 / (1.0 + mlp_hidden_mult * 0.05)
            blended = emb + (enhance_strength * residual_scale) * (refined - emb)

            # NEW: Preserve original (default 0.0 = full effect)
            blended = blended * (1 - preserve_original) + emb * preserve_original

            # Self-attention (tunable instead of fixed 0.3)
            if add_self_attention:
                attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True, device=device, dtype=emb.dtype)
                attn_out, _ = attn(blended, blended, blended)
                blended = blended + attention_strength * attn_out

            enhanced.append((blended.to("cpu", dtype=orig_dtype), meta))

        return (enhanced, )

NODE_CLASS_MAPPINGS = {"CapitanAdvancedEnhancer": CapitanAdvancedEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"CapitanAdvancedEnhancer": "Capitan Advanced Enhancer"}