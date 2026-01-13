"""
Conditioner Pack for ComfyUI
- Basic Conditioning Enhancer
- Capitan Advanced Enhancer
"""

from .enhancer import ConditioningEnhancer
from .capitan_advanced_enhancer import CapitanAdvancedEnhancer

NODE_CLASS_MAPPINGS = {
    "ConditioningEnhancer": ConditioningEnhancer,
    "CapitanAdvancedEnhancer": CapitanAdvancedEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningEnhancer": "Conditioning Enhancer (Basic)",
    "CapitanAdvancedEnhancer": "Capitan Advanced Enhancer",
}

__version__ = "1.0.1"
