# MAMR

Code for multi-agent reinforcement learning to unify order-matching and vehicle-repositioning in ride-hailing services. (https://doi.org/10.1080/13658816.2022.2119477)

## Environment
 
The code is supposed to run in the environment:

python 3.7

CUDA 11.1

Pytorch 1.8.1

numpy 1.20.2

## Instructions

### step 1

Download data from https://outreach.didichuxing.com. Each taxi order consists of an order ID, pick-up/drop-off timestamps, and locations. Each driver track point consists of a driver ID, the timestamp and locations.

### step 2

Preprocess the data.
```
python data/export_neighborhood_data.py
python data/create_city_state.py
```

Files generated including:
```
envs/driver_distribution.csv  
data/neighborhood.dill
data/city_states/city_states.dill
data/hex_bins/hex_bin_attributes.csv
data/hex_bins/hex_distances.csv
```

### step 3

Train the model.
```
python train.py --num_agents ${num_agents}
```

### step 4
Test the model.
```
python train.py --num_agents ${num_agents} --model_dir ${model_dir} --test True
```

## Acknowledgement

Thanks to these repositories:
- [light_mappo](https://github.com/tinyzqh/light_mappo)
- [optimize-ride-sharing-earnings](https://github.com/transparent-framework/optimize-ride-sharing-earnings)

## Cite
@article{xu2022multi,
  title={Multi-agent reinforcement learning to unify order-matching and vehicle-repositioning in ride-hailing services},
  author={Xu, Mingyue and Yue, Peng and Yu, Fan and Yang, Can and Zhang, Mingda and Li, Shangcheng and Li, Hao},
  journal={International Journal of Geographical Information Science},
  pages={1--23},
  year={2022},
  publisher={Taylor \& Francis}
}
