import torch.nn as nn
from .util import init, get_clones
import torch


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func)
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func)
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, hidden_size):
        super(MLPBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N

        self.obs_dim = obs_shape[0]
        if self.obs_dim == 10:
            self.hidden_size = args.hidden_size_a  # 10
        elif self.obs_dim == 145:
            self.hidden_size = args.hidden_size_c  # 145
        elif self.obs_dim == 3:
            self.hidden_size = args.hidden_size_st  # 3

        self.mlp = MLPLayer(self.obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def standardization(self, data, mu, sigma):
        norm = (data - mu) / (sigma + 1e-5)
        return norm

    def compute_norm(self, data, mean=[70.5, 53.5, 3.0, 25, 12], std=[40.99085263811915, 31.175578048637153, 2.0, 60, 30]):
        # obs: hexbinID + timestep + dayofweek + supply/demand(gap) current/neighbor # (109,1,6000,10)
        # share_obs: hexbinID + timestep + dayofweek + supply/demand(gap) all # (109,1,6000,145)
        # share_obs: timestep + dayofweek + supply/demand(gap) all # (109,1,6000,144)
        # st_obs: hexbinID + timestep + dayofweek
        data[:,0] = self.standardization(data[:,0], mean[0], std[0])
        data[:,1] = self.standardization(data[:,1], mean[1], std[1])
        data[:,2] = self.standardization(data[:,2], mean[2], std[2])
        if self.obs_dim == 10: # 3+7=10 # (109,1,6000,3)
            data[:,3:] = self.standardization(data[:,3:], mean[3], std[3])
        elif self.obs_dim == 145: # 3+142=145
            data[:,3:] = self.standardization(data[:,3:], mean[4], std[4])
        elif self.obs_dim == 3:
            pass
        else:
            raise NotImplementedError
        return data

    def forward(self, x):
        x = self.compute_norm(x)

        x = self.mlp(x)  # 6000,10 > 648,10 (bs)
        return x