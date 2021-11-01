

## Creating the dataset
- create two folders A and B with subfolders train and test in each.
- Keep the clean train data in A and the blurred data in B in respective train and test subfolders
- Create the concatenated images using 

```
python scripts/combine_A_and_B.py --fold_ A /path to fold A --fold_B /path to fold B --fold_AB /path to save the catenated image
```

## How to train and test
-Edit the parameters inside train.lua 
- For training follow the command
```
DATA_ROOT=/path_to_catenated_data/  name=expt_name which_direction=BtoA gpu=3 th train.lua

```
-For testing
```
 DATA_ROOT=/path/to/data/ name=expt_name which_direction=BtoA phase=val th test.lua
```


### Prerequisites
- Linux or OSX
- NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)

### Requirements
- Install torch and dependencies from https://github.com/torch/distro
- Install torch packages `nngraph` and `display`
```bash
luarocks install nngraph
luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
```
## Acknowledgments
Code borrows heavily from [DCGAN](https://github.com/soumith/dcgan.torch). The data loader is modified from [DCGAN](https://github.com/soumith/dcgan.torch) and  [Context-Encoder](https://github.com/pathak22/context-encoder).
