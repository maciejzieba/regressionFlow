import torch
from torch import optim
from torch import nn
from models.flow import get_hyper_cnf
from utils import truncated_normal, standard_normal_logprob


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class HyperRegression(nn.Module):
    def __init__(self, args):
        super(HyperRegression, self).__init__()
        self.input_dim = args.input_dim
        self.hyper = HyperFlowNetwork(args)
        self.args = args
        self.point_cnf = get_hyper_cnf(self.args)
        self.gpu = args.gpu

    def make_optimizer(self, args):
        def _get_opt_(params):
            if args.optimizer == 'adam':
                optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                                       weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer
        opt = _get_opt_(list(self.hyper.parameters()) + list(self.point_cnf.parameters()))
        return opt

    def forward(self, x, y, opt, step, writer=None):
        opt.zero_grad()
        batch_size = x.size(0)
        target_networks_weights = self.hyper(x)

        # Loss
        y, delta_log_py = self.point_cnf(y, target_networks_weights, torch.zeros(batch_size, y.size(1), 1).to(y))
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, y.size(1), 1).sum(1)
        log_px = log_py - delta_log_py

        loss = -log_px.mean()

        loss.backward()
        opt.step()
        recon = -log_px.sum()
        recon_nats = recon / float(y.size(0))
        return recon_nats

    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    def decode(self, z, num_points):
        # transform points from the prior to a point cloud, conditioned on a shape code
        target_networks_weights = self.hyper(z)
        y = self.sample_gaussian((z.size(0), num_points, self.input_dim), None, self.gpu)
        x = self.point_cnf(y, target_networks_weights, reverse=True).view(*y.size())
        return y, x


class HyperFlowNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.z_size = args.zdim
        self.use_bias = True
        self.relu_slope = 0.2

        output = []

        dims = tuple(map(int, args.hyper_dims.split("-")))
        self.n_out = dims[-1]
        model = []
        for k in range(len(dims)):
            if k == 0:
                model.append(nn.Linear(in_features=self.z_size, out_features=dims[k], bias=self.use_bias))
            else:
                model.append(nn.Linear(in_features=dims[k-1], out_features=dims[k], bias=self.use_bias))
            if k < len(dims) - 1:
                model.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*model)

        dims = tuple(map(int, args.dims.split("-")))
        for k in range(len(dims)):
            if k == 0:
                output.append(nn.Linear(self.n_out, args.input_dim * dims[k], bias=True))
            else:
                output.append(nn.Linear(self.n_out, dims[k - 1] * dims[k], bias=True))
            #bias
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            #scaling
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            #shift
            output.append(nn.Linear(self.n_out, dims[k], bias=True))

        output.append(nn.Linear(self.n_out, dims[-1] * args.input_dim, bias=True))
        # bias
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))
        # scaling
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))
        # shift
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))

        self.output = ListModule(*output)

    def forward(self, x):
        output = self.model(x)
        multi_outputs = []
        for j, target_network_layer in enumerate(self.output):
            multi_outputs.append(target_network_layer(output))
        multi_outputs = torch.cat(multi_outputs, dim=1)
        return multi_outputs
