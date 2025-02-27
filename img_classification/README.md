# Image Classification

**Setup**: Please follow the instructions from the [DeiT](https://github.com/facebookresearch/deit) library to configure the environment. 

Within the **Image Classification** folder, you’ll find `models_lion.py`, which contains the implementations of **LION-🔥**, **LION-D**, and **LION-S** in three formats: attention, recurrent and chunk-based. We also introduce specialized “curves.py” for processing image patches in **LION-D** and **LION-S**, enhancing spatial representation as discussed in our paper with notation **LION-D/S<sup>♮</sup>**.


Below is an example of how to run **LION-D** for image classification from scratch, followed by a command that demonstrates **LION-S<sup>♮</sup>** training using “curves” and altered patch orders:

```bash
# Example 1: Train LION-D from scratch
python -m torch.distributed.launch --nproc_per_node=4 --use_env main_lion.py \
    --model lion_base_patch16_224 \
    --batch-size 256 \
    --data-path /datapath \
    --output_dir /outputpath
```

```bash
# Example 2: Train LION-S (or LION-D) with curves and patch-order changes
python -m torch.distributed.launch --nproc_per_node=4 --use_env main_lion.py \
    --model lion_base_patch16_224 \
    --batch-size 256 \
    --data-path /datapath \
    --output_dir /outputpath \
    --mask_type Selective \
    --order S \
    --format Attention
```


Inside models_lion, there are 3 sizes defined as:

- LION in base scale (86M) with an image size of 224, called `lion_base_patch16_224`
- LION in small scale (22M) with an image size of 224, called `lion_small_patch16_224`
- LION in tiny scale (5M) with an image size of 224, called `lion_tiny_patch16_224`

Below are some of the key arguments you can customize when training LION-based models:

1. **`pos_emb`**  Enables fixed positional embeddings (as in ViT) (default `False`).  
   - To set True: `--pos_emb`

2. **`cls_tok`**   Uses an independent classification token if set to `True`; otherwise, classification is based on the average pooling of all tokens (default `False`).  
   - To set True: `--cls_tok`

3. **`mask_type`**  Defines how masking or gating is applied. Supported options include `Lit`, `Decay`, and `Selective` which correspond to **LION-🔥**, **LION-D**, and **LION-S** respectively.  
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



**Notes:**  
- Choose any desired size (e.g., `lion_base_patch16_224`, `lion_small_patch16_224` or `lion_tiny_patch16_224`).  
- By changing the `mask_type`, get different **LION** variants (e.g., LION-🔥, LION-D or LION-S).
- Determine the internal representation format with `format` (e.g., `Attention` for training, `RNN` or `Chunk` for inference).
- Adjust `nproc_per_node`, `batch-size`, `data-path`, and `output_dir` according to your hardware setup and dataset location.  
- The additional flags (`order`, `pos_emb`, `cls_tok`) control the specific training variations (e.g., changing patch-order “S,”, adding positional embeddings and using a classification token).
- As our codebase extends [DeiT](https://github.com/facebookresearch/deit), you can easily distill **RegNET** into a **LION** model by following the **same** distillation commands used for DeiT—just swap in the LION model name. This ensures you can leverage the established DeiT distillation process without additional modifications.
