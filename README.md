## Capitan Conditioning Enhancer
Lightweight post-processing node for Qwen3-4B conditioning in Z-Image Turbo workflows.
Refines the 2560-dim CONDITIONING from native Qwen3-4B text encoder with:

## Per-token normalization
Optional 8-head self-attention
2-layer MLP refiner (very wide hidden size supported)
Positive/negative blend strength

Improves coherence, detail retention, mood consistency, and prompt following.


## Installation
extract zip in ComfyUI/custom_nodes 

Or

git clone https://github.com/capitan01R/Capitan-ConditioningEnhancer.git

restart ComfyUI or reload custom nodes

No dependencies.
## Placement
After CLIP Text Encode (Prompt) → before KSampler.
![](images/node.png)

stack method:
![](images/node2.png)

## Grid examples:
![](images/horizontal_tiger_grid.png)


<details>
<summary>Click to view full prompt</summary>
Close-up front-facing portrait of a cute innocent orange domestic house cat's head, perfectly vertically split down the center. Left half is a solid matte black domestic cat with short fur and one glowing yellow eye. Right half is a prehistoric saber-tooth tiger with thick white fur, bold grey stripes, massive thickened jaw, and one extremely long curved ivory saber tooth fully visible. The right half is slowly disintegrating into fine ethereal white dust and smoke particles drifting away, leaving a faint ghostly grey translucent shadow of the tiger shape behind. The cat's mouth is open in a soft mew, with the right-side teeth hazy and fading. Dark charcoal void background to highlight the dust particles and black fur. High detail, sharp contrast, surreal horror style, no blending between halves, tiger side remains fully distinct until the very end.
</details>

### Parameters Explained

| Parameter              | Type    | Default | Range             | What it does                                                                                       | Typical use / effect                                                                 |
|------------------------|---------|---------|-------------------|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| enhance_strength       | FLOAT   | 0.0     | -2.0 → 2.0        | Positive: add refinement, Negative: subtract refinement (sharper, anti-smoothing)                | 0.00–0.15 = subtle polish<br>Negative = crisp/literal<br>>0.5 = high risk            |
| normalize              | BOOLEAN | true    | true/false        | Per-token mean subtraction + unit variance normalization                                          | Almost always true — stability & cleaner generations                                 |
| add_self_attention     | BOOLEAN | false   | true/false        | Light 8-head self-attention across tokens (distant prompt parts influence each other)            | true = better cohesion & complex scene unity<br>false = literal control              |
| mlp_hidden_mult        | INT     | 2       | 1 → 100           | Hidden layer width multiplier for MLP (2560 → 2560×mult → 2560)                                  | 2–10 = balanced<br>20–50 = hyper-literal detail (recommended max)<br>60–100 = extreme (low strength only) |


## MLP note — UI allows up to 100, but recommended max is 50. Higher rarely adds value and mostly increases artifact risk even at tiny strength (0.01–0.05).
Recommended Starting Points
Daily essential / stabilizer

strength: 0.00–0.10
normalize: true
add_self_attention: true
mlp_hidden_mult: 2–4

Hyper-literal detail mode (stacked)

First node: strength 0.05–0.10, normalize true, self-attn true, mult 2–4
Second node: strength 0.05–0.15 (or -0.05 – -0.15), normalize true, self-attn false, mult 40–50

Warnings
High strength (>0.4) + high mlp_hidden_mult + self-attention → rainbow artifacts / noise likely.
Very high mult (60+) needs extremely low strength (≤0.05).
Always test same seed first. Negative strength + high mult usually safer than positive.
