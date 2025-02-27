# LION

### Dependencies

To install the dependencies, run `pip install -r requirements.txt`.

### Data

In order to download the C4 dataset, run `src/convert_dataset.py`. 

### Pretraining

Select the according `config.yaml` file from the `yamls/pretrain` folder. Make sure to modify the `data_local` path to match the location of the C4 dataset on your machine.

To pretrain the model, run `main.py` with the desired configuration file. For example, to pretrain a Lion-Lit-Large model, run

```bash
composer main.py yamls/pretrain/lion-lit-large.yaml
```

All of our pretrained weights are available  on [Huggingface](https://huggingface.co/collections/LIONS-EPFL/lion-67c0a5f094df709e5e9e7a58).

### Finetuning

Select the according `config.yaml` file from the `yamls/finetune-glue` folder. Make sure to modify the `starting_checkpoint_load_path` to match the location of the checkpoint of the pretraining run you want to finetune.

To finetune the model, run `glue.py` with the desired configuration file. For example, to finetune a Lion-Lit-Large model on GLUE, run 

```bash
python3 glue.py yamls/finetune-glue/lion-lit-large.yaml
```


### Acknowledgements
This repository is built from the [M2-BERT](https://github.com/HazyResearch/m2) and [Hydra](https://github.com/goombalab/hydra) repositories.
