[![arXiv](https://img.shields.io/badge/arXiv-2502.16249-B31B1B?style=for-the-badge&logo=arXiv&logoColor=white)](https://www.arxiv.org/abs/2502.16249)
[![Blog Post](https://img.shields.io/badge/Blog-Post-green?style=for-the-badge&logo=medium&logoColor=white)](https://arshiaafzal.github.io/blog/2025/lion-part1-model/)


------------------

# LION 🦁


Transformers with Linear Attention enable fast and parallel training. Moreover, they can be formulated as Recurrent Neural Networks (RNNs), for efficient linear-time inference. While extensively evaluated in causal sequence modeling, they have yet to be extended to the bi-directional setting. We introduce the **LION** framework, establishing new theoretical foundations for Linear Transformers in bi-directional sequence modeling. **LION** constructs a bi-directional RNN equivalent to full **Linear Attention**. This extends the benefits of Linear Transformers: parallel training and efficient inference into the bi-directional setting.


<div style="display: flex; align-items: flex-start;">
  
  <!-- Left Side: Text & Table -->
  <div style="flex: 1; padding-left: 10px;">

  <p>
  Existing memory-efficient bi-directional models employ more than 2x the training time of a Transformer. Our Linear Attention framework benefits from memory-efficient inference while maintaining the Transformer training speed.
  </p>


<table>
    <tr>
      <th>Task</th>
      <th><span style="background-color: rgb(230, 255, 230); padding: 3px; color:black">🦁-🔥 </span></th>
      <th><span style="background-color: rgb(229, 204, 230); padding: 3px; color:black">🦁-D </span></th>
      <th><span style="background-color: rgb(255, 233, 211); padding: 3px; color:black">🦁-S </span></th>
      <th>Hydra</th>
      <th>Vim</th>
    </tr>
    <tr>
      <td>Vision</td>
      <td>$\times 0.73$</td>
      <td>$\times 1.39$</td>
      <td>$\times 1.46$</td>
      <td>$\times 2.51$</td>
      <td>$\times 10.86$</td>
    </tr>
    <tr>
      <td>MLM </td>
      <td>$\times 0.95$</td>
      <td>$\times 1.10$</td>
      <td>$\times 1.32$</td>
      <td>$\times 3.13$</td>
      <td>✗</td>
    </tr>
</table>

  </div>
</div>

Using **LION**, we cast three Linear Transformers to their bi-directional form:  
- **LION-️‍🔥**, the bi-directional variant corresponding to [LinearTransformer](https://arxiv.org/abs/2006.16236).
- **LION-D**, extending [RetNet](https://arxiv.org/abs/2307.08621).
- **LION-S**, a Linear Transformer with a stable selective mask inspired by selectivity of SSMs like [Mamba🐍](https://arxiv.org/abs/2405.21060).

By replacing the attention block with **LION (-️‍🔥, -D, -S)**, we achieve performance on bi-directional tasks that is comparable to Transformers and State-Space Models (SSMs) while improving training speed.


---

This repository provides the code for the LION model, covering **image classification** and **masked language modeling (MLM)**. Our image classification setup is adapted from [DeiT](https://github.com/facebookresearch/deit), and the MLM implementation builds on [M2-BERT](https://github.com/HazyResearch/m2/tree/main).

## Image Classification

Within the **Image Classification** folder, you’ll find `models_lion.py`, which contains the implementations of **LION-🔥**, **LION-D**, and **LION-S**. We also introduce specialized “curves.py” for processing image patches in **LION-S** and **LION-D**, enhancing spatial representation as discussed in our paper.


Below is an example of how to run **LION-D** for image classification from scratch, followed by a command that demonstrates **LION-S/LION-D** training using “curves” and altered patch orders:

```bash
# Example 1: Train LION-D from scratch
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --model lion_base_patch16_224 \
    --batch-size 256 \
    --data-path /datapath \
    --output_dir /outputpath
```

```bash
# Example 2: Train LION-S (or LION-D) with curves and patch-order changes
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --model lion_base_patch16_224 \
    --batch-size 256 \
    --data-path /datapath \
    --output_dir /outputpath \
    --mask_type Decay \
    --order S \
    --format Attention
```


Inside models_lion, there are 3 sizes defined

- LION in base scale with pathch size of 224
- LION in small scale with pathch size of 224
- LION in tiny scale with pathch size of 224

Below are some of the key arguments you can customize when training LION-based models:

1. **`pos_emb`**  Enables fixed positional embeddings (as in ViT).  
   - Example usage: `--pos_emb True`

2. **`cls_tok`**   Uses an independent classification token if set to `True`; otherwise, classification is based on the average pooling of all tokens (default `False`).  
   - Example usage: `--cls_tok True`

3. **`mask_type`**  Defines how masking or gating is applied. Supported options include `Lit`, `Decay`, and `Selective`.  
   - Example usage: `--mask_type Decay`

4. **`order`**  Specifies the order in which image patches are processed. Options include:
     - `Normal` (default order)
     - `S` (special ordering)  
   - Example usage: `--order S`

5. **`format`**  Controls the internal representation of the sequence. Valid options are:
     - `Attention` (standard attention-like format)
     - `RNN` (recurrent-like format)
     - `Chunk` (chunk-based approach)  
   - Example usage: `--format Attention`

6. **`chunk_size`**   An integer that sets the size of chunks when using chunk-based processing.  
   - Example usage: `--chunk_size 64`

By combining these arguments, you can experiment with different positional embeddings, classification tokens, patch orders, and masking mechanisms to adapt the LION model to your specific tasks and preferences.



**Note:**  
- Replace `lion_base_patch16_224` with any desired **LION** variant (e.g., LION-D or LION-S).  
- Adjust `nproc_per_node`, `batch-size`, `data-path`, and `output_dir` according to your hardware setup and dataset location.  
- The additional flags (`--mask_type`, `--order`, `--format`) control the specific training variations (e.g., applying decaying masks, changing patch-order “S,” and switching to an attention-like format).
- Because our codebase extends [DeiT](https://github.com/facebookresearch/deit), you can easily distill **RegNET** into a **LION** model by following the **same** distillation commands used for DeiT—just swap in the LION model name. This ensures you can leverage the established DeiT distillation process without additional modifications.

Below is the results on Image Classification for LION models vs benchmarks. 

| Model | #Param | Imagenet Top-1 Acc. | Train. time |
|-------|--------|---------------------|------------|
| $\text{ViT}$ | 86M | $77.9$ | $\times 1$ |
| $\text{DeiT}$ | 86M | $\underline{81.8}$ | $\times 1$ |
| $\text{Hydra}$ | 104M | $81.0$ | $\times 2.51$ |
| $\text{Vim}$ | 98M | $\mathbf{81.9}$ | $\times 10.86$ |
| $\text{LION-}\text{🔥}$ | 86M | $74.7$ | $\mathbf{\times 0.73}$ |
| $\text{LION-D}$ | 86M | $77.8$ | $\times \underline{1.39}$ |
| $\text{LION-D}^{\natural}$ | 86M | $80.2$ | $\times 1.48$ |
| $\text{LION-S}$ | 86M | $76.3$ | $\times 1.46$ |
| $\text{LION-S}^{\natural}$ | 86M | $79.9$ | $\times 1.68$ |


-------------------
# Masked Language Modeling (MLM)





-------------------


# Citation
If you use this work, please consider citing the paper:

```
@article{afzal2025linear,
  title={Linear Attention for Efficient Bidirectional Sequence Modeling},
  author={Afzal, Arshia and Abad Rocamora, Elias and Candogan, Leyla Naz and Puigdemont, Pol and Tonin, Francesco and Wu, Yongtao and Shoaran, Mahsa and Cevher, Volkan},
  journal={arXiv preprint arXiv:2502.16249},
  year={2025},
  url={https://arxiv.org/abs/2502.16249},
  doi={10.48550/arXiv.2502.16249}
}
```

----------

# Commercial License


##### © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Laboratory for Information and Inference Systems (LIONS), 2025. See the [LICENSE.TXT](https://github.com/LIONS-EPFL/LION/blob/main/LICENSE.txt) file for more details
