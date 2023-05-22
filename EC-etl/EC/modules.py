import torch
from torch import nn
import torch.nn.functional as F
import torchvision

def hook_print_grad(grad: torch.Tensor):
    print(grad.abs().mean().item(), grad.abs().max().item())

class SharedGroupMLP(nn.Module):
    # shared mlp over groups splited over the last dim
    # take the last two dims as input dims (while the last one is splitted)
    def __init__(self, groups=8, group_input_dim=320, group_output_dim=32,
            hidden_dims=[128], add_res_block=True, nr_mlps=1, flatten=True,
            shared=True):
        super().__init__()
        # print(groups, group_input_dim, group_output_dim, hidden_dims, add_res_block, nr_mlps, flatten, shared)
        self.shared = shared
        self.groups = groups
        self.group_input_dim = group_input_dim
        self.group_output_dim = group_output_dim
        # mlps indicates different experts
        if shared:
            mlps = [MLPModel(group_input_dim, group_output_dim,
                hidden_dims=hidden_dims) for i in range(nr_mlps)]
        else:
            exit()
        self.mlps = nn.ModuleList(mlps)
        self.FCblocks = None
        if shared and add_res_block:
            FCblocks = [FCResBlock(group_output_dim) for i in range(nr_mlps)]
            self.FCblocks = nn.ModuleList(FCblocks)
        self.flatten = flatten

    def forward(self, x):
        assert x.size(-1) % self.groups == 0
        group_size = x.size(-1) // self.groups
        xs = x.split(group_size, dim=-1)
        new_xs = []
        for i in xs:
            # apply on last two axis, the last axis is splitted
            x = i.flatten(-2, -1)
            # x.shape: (batch, *, group_input_dim)
            new_xs.append(x)
        x = torch.stack(new_xs, dim=-2)
        # x.shape: (batch, *, groups, group_input_dim)
        if not self.shared:
            x = x.flatten(-2, -1)
            # x.shape: (batch, *, groups * group_input_dim)
        ys = []
        for ind, mlp in enumerate(self.mlps):
            y = mlp(x)
            if self.FCblocks:
                y = self.FCblocks[ind](y)
            ys.append(y)
        x = torch.cat(ys, dim=-1)
        # [no-share] x.shape: (batch, *, nr_mlps * groups * group_output_dim)
        # [shared] x.shape: (batch, *, groups, nr_mlps * group_output_dim)
        if self.shared and self.flatten:
            x = x.flatten(-2, -1)
            # x.shape: (batch, *, groups * nr_mlps * group_output_dim)
        return x


class FCResBlock(nn.Module):
    def __init__(self, nn_dim, use_layer_norm=True):
        self.use_layer_norm = use_layer_norm
        super(FCResBlock, self).__init__()
        self.norm_in = nn.LayerNorm(nn_dim)
        self.norm_out = nn.LayerNorm(nn_dim)
        self.transform1 = torch.nn.Linear(nn_dim, nn_dim)
        torch.nn.init.normal_(self.transform1.weight, std=0.005)
        self.transform2 = torch.nn.Linear(nn_dim, nn_dim)
        torch.nn.init.normal_(self.transform2.weight, std=0.005)

    def forward(self, x):
        if self.use_layer_norm:
            x_branch = self.norm_in(x)
        else:
            x_branch = x
        x_branch = self.transform1(F.relu(x_branch))
        if self.use_layer_norm:
            x_branch = self.norm_out(x_branch)
        x_out = x + self.transform2(F.relu(x_branch))
        #x_out = self.transform2(F.relu(x_branch))
        #x_out = F.relu(self.transform2(x_branch))
        #return F.relu(x_out)
        return x_out


class MLPModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, bias=True, act="leaky_relu"):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]
        layers = []
        assert act in ['relu', 'leaky_relu']
        dims = [in_dim] + hidden_dims + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if i != len(dims) - 2:
                if act == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif act == 'leaky_relu':
                    layers.append(nn.LeakyReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

if __name__ == "__main__":
    recon_analogy_shared_mlp_configs = {
        "groups": 4, # self.symbol_attr_dim,
        "group_input_dim": 40, # 2 * self.image_embedding_dim // self.symbol_attr_dim,
        "group_output_dim": 20, # self.image_embedding_dim // self.symbol_attr_dim,
        "hidden_dims": [64, 32], 
        "add_res_block": True, 
        "nr_mlps": 10, # self.symbol_onehot_dim, 
        "flatten": False,
        "shared": True
    }
    dummy_input = torch.Tensor(32, 2, 80)
    me = torch.Tensor(32, 400)

    model = SharedGroupMLP(**recon_analogy_shared_mlp_configs)
    out = model(dummy_input)
    out = out.view(-1, 4, 10, 20) # (-1, self.args.symbol_attr_dim, self.args.symbol_onehot_dim, self.args.image_embedding_dim // self.args.symbol_attr_dim)
    out = out.split(1, dim=1) # self.args.symbol_attr_dim
    print(len(out))
    atten_model = nn.ModuleList([nn.Linear(400, 10) for _ in range(4)])
    new_o = []
    for m, o in zip(atten_model, out):
        print(o.size())
        w = m(me).unsqueeze(1)
        new_o.append(torch.matmul(w, o.squeeze())) 
    new_o = torch.cat(new_o, dim=1)
    print(new_o.size())