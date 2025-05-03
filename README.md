# ReLIEF

**Code for ReLIEF: Revealing Learned Inequitable Earth Forecasts**

## Set Up

First clone the repo.

Then, set up your environment:

<details>
<summary>Instructions for Brown's OSCAR</summary>

Before doing the below steps, make sure to:
```
module load miniconda3/23.11.0s
conda init bash
```
More instructions available on [CCV's website](https://docs.ccv.brown.edu/oscar/software/miniconda)
</details>
</br>

Use the following steps to set up your python environment:
```
conda create -n relief.env
conda activate relief.env
pip install --file requirements.txt
conda install --channel conda-forge pygmt
```
After the setup, the environment will be active.
Deactivate the environment with `conda deactivate`.
Activate it again at anytime while in the repo with `conda activate relief.env`

Finally, login to your Weights & Biases account with `wandb login`. If you need to set up your account, you can find instructions [here](https://docs.wandb.ai/guides/integrations/lightning/#install-the-wandb-library-and-log-in).

## ReLIEF Evaluation

... TODO

## Training

### Neuralops

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 neuralops/sfno.py
```

Adjust `nnodes` to the number of nodes you have, and `nproc_per_node` to the number of GPUs per node.

--- 

#### Hydra

Example: `torchrun --standalone --nnodes=1 --nproc_per_node=2 neuralops/sfno.py` will use the `conf/neuralops/sfno/configs/default.yaml` configuration. To run multiple jobs using different configurations, run with the flag:

```
--multirun configs=default,<config_filename_2>,...,<config_filename_n>
```

The same principle holds for other models, using their respective subdirectory in `conf/`.

To manually maniupate config values for a single run, you could, for example:

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 neuralops/sfno.py configs.trainer.devices=1
```

More details available on the [Hydra docs](https://hydra.cc/docs/1.3/intro/)

---

## Notes

All data is translated so that the (0,0) index is the gridpoint that is most southern and closest to the meridian on the east. That is, the gridpoint that approaches 0E, 90S. Latitudes thus range from -90 to 90, and longitudes from 0 to 360 (the range 180 to 360 may alternatively be listed as -180 to -0).

## Testing

Run `pytest` in the terminal.
