# Checkpoint Summary
**Model:** VideoMAE-S  
**Checkpoint:** vit_s_k400_ft.pth  
**Total Parameters:** **22,033,936 (22.03M)**  
**Classes:** 400 (Kinetics-400)

---

## videomae
### embeddings
#### patch_embeddings
- projection
  - weight `[384, 3, 2, 16, 16]` — 589,824
  - bias `[384]` — 384

---

## encoder
> **Depth:** 12 Transformer layers  
> **Hidden Dim:** 384  
> **MLP Ratio:** 4× (1536)

### layer.0
- attention
  - q_bias `[384]`
  - v_bias `[384]`
  - query.weight `[384, 384]`
  - key.weight `[384, 384]`
  - value.weight `[384, 384]`
  - output.dense
    - weight `[384, 384]`
    - bias `[384]`
- intermediate
  - dense.weight `[1536, 384]`
  - dense.bias `[1536]`
- output
  - dense.weight `[384, 1536]`
  - dense.bias `[384]`
- layernorm_before
  - weight `[384]`
  - bias `[384]`
- layernorm_after
  - weight `[384]`
  - bias `[384]`

### layer.1
*(same structure as layer.0)*

### layer.2
*(same structure as layer.0)*

### layer.3
*(same structure as layer.0)*

### layer.4
*(same structure as layer.0)*

### layer.5
*(same structure as layer.0)*

### layer.6
*(same structure as layer.0)*

### layer.7
*(same structure as layer.0)*

### layer.8
*(same structure as layer.0)*

### layer.9
*(same structure as layer.0)*

### layer.10
*(same structure as layer.0)*

### layer.11
*(same structure as layer.0)*

---

## Head
### fc_norm
- weight `[384]`
- bias `[384]`

### classifier
- weight `[400, 384]` — 153,600
- bias `[400]` — 400

---

## Summary
- **Architecture:** VideoMAE-S (ViT-Small)
- **Temporal Patch Size:** 2 frames
- **Spatial Patch Size:** 16×16
- **Encoder Layers:** 12
- **Hidden Dimension:** 384
- **MLP Dimension:** 1536
- **Pretrained on:** Kinetics-400
- **Task:** Video Classification
