# ZeoNet: 3D convolutional neural networks for predicting adsorption in nanoporous zeolites
In this work, we present ZeoNet, a representation learning framework using convolutional neural networks (ConvNets) and a novel 3D volumetric representation for predicting
long-chain hydrocarbon adsorption in all-silica zeolites. The best-performing ZeoNet achieves a correlation coefficient r2 = 0.977 and a mean-squared error MSE = 3.8 in lnkH, which corresponds to an error of only 9.3 kJ/mol in adsorption free energy. We also demonstrate that the predictions driven primarily by the accessible pore volume rather than the region occupied by framework atoms.

[[Paper]](https://doi.org/10.1039/D3TA01911J)

## Install Dependencies

```
conda create --name <env_name> --file requirements_conda.txt -c pytorch

```
## Dataset Construction
1.Calculate descriptors. Distance grids and handed-engineered features are calculated according to http://www.zeoplusplus.org/examples.html. The handed-engineered features along with adsorption data are stored in the C18-adsorption/each-zeolite-info.csv file. Some examples of distance grids data are shown in distance-grids-h5 folder.

2.Set up directories to organize the data for distance grids and adsorption. Store the distance grids dataset (h5 files) in the 'distance-grids-h5' directory and the adsorption data (csv file) in the 'C18-adsorption' directory. Ensure that within the 'distance-grids-h5' directory, there are two subfolders named IZASC and PCOD. Place the distance grid files (h5 files) in their respective subfolders.

3.For creating your custom training, validation, and test datasets, list the names of zeolite samples in 'train_set.txt', 'val_set.txt', and 'test_set.txt' files within the 'C18-adsorption' directory.

## Running
```
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
python train.py --epochs=30 --batch_size=16 --lr=0.001 --optimizer='Adam' --model='resnet' --model_hp=18 --grid_resolution=0.45 --grid_size=100
```
or
```
bash train-script.sh <EXP> <EPOCHS> <BS> <LR> <Optimizer> <MODEL> <MODELHP> <DSETROOT> <GRIDRESOLUTION> <GRIDSIZE>
```
e.g. `bash train-script.sh "A001" 30 16 0.001 Adam resnet 18 . 0.45 100`

The code directory needs to be added to the `PYTHONPATH` environment variable, as done in the example above and in `train-script.sh`.


