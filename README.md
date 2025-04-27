# ReLIEF

**Code for ReLIEF: Revealing Learned Inequitable Earth Forecasts**

### Set Up

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

### Training

--- 

#### Neuralops

```
python -m neuralops.<model_name>
```

Example: `python -m neuralops.sfno` will use the `conf/neuralops/sfno/configs/default.yaml` configuration. To run multiple jobs using different configurations, run:

```
python -m neuralops.<model_name> --multirun configs=default,<config_filename_2>,...,<config_filename_n>
```

The same principle holds for other models, using their respective subdirectory in `conf/`

---

### Notes

All data is translated so that the (0,0) index is the gridpoint that is most southern and closest to the meridian on the east. That is, the gridpoint that approaches 0E, 90S. Latitudes thus range from -90 to 90, and longitudes from 0 to 360 (the range 180 to 360 may alternatively be listed as -180 to -0).

### Testing

Run `pytest` in the terminal.
