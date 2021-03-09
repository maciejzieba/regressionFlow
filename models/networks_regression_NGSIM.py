import torch
import numpy as np
from torch import optim
from torch import nn
from models.flow import get_latent_cnf
from models.flow import get_hyper_cnf
from utils import truncated_normal, standard_normal_logprob, standard_laplace_logprob
from torch.nn import init
from torch.distributions.laplace import Laplace


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
        self.logprob_type = args.logprob_type

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

    def forward(self, hist, nbrs, masks, op_mask, y, opt):
        opt.zero_grad()
        batch_size = y.size(0)
        target_networks_weights = self.hyper(hist, nbrs, masks)

        # Loss
        y, delta_log_py = self.point_cnf(y, target_networks_weights, torch.zeros(batch_size, y.size(1), 1).to(y))
        if self.logprob_type == "Laplace":
            log_py = standard_laplace_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        if self.logprob_type == "Normal":
            log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, y.size(1), 1).sum(1)
        log_px = log_py - delta_log_py
        log_px = log_px*op_mask
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


    @staticmethod
    def sample_laplace(size, gpu=None):
        m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        y = m.sample(sample_shape=torch.Size([size[0], size[1], size[2]])).float().squeeze(3)
        y = y if gpu is None else y.cuda(gpu)
        return y


    def decode(self, hist, nbrs, masks, num_points):
        # transform points from the prior to a point cloud, conditioned on a shape code
        target_networks_weights = self.hyper(hist, nbrs, masks)
        if self.logprob_type == "Laplace":
            y = self.sample_laplace((target_networks_weights.size(0), num_points, self.input_dim), self.gpu)
        if self.logprob_type == "Normal":
            y = self.sample_gaussian((target_networks_weights.size(0), num_points, self.input_dim), None, self.gpu)
        x = self.point_cnf(y, target_networks_weights, reverse=True).view(*y.size())
        return y, x

    def get_logprob(self, hist, nbrs, masks, y_in):
        batch_size = y_in.size(0)
        target_networks_weights = self.hyper(hist, nbrs, masks)

        # Loss
        y, delta_log_py = self.point_cnf(y_in, target_networks_weights, torch.zeros(batch_size, y_in.size(1), 1).to(y_in))
        if self.logprob_type == "Laplace":
            log_py = standard_laplace_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        if self.logprob_type == "Normal":
            log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)

        delta_log_py = delta_log_py.view(batch_size, y.size(1), 1).sum(1)
        log_px = log_py - delta_log_py

        return log_py, log_px


class HyperFlowNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = highwayNet()
        output = []
        self.n_out = 128
        # self.n_out = 46080
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

    def forward(self, hist, nbrs, masks):
        output = self.encoder(hist, nbrs, masks)
        # output = output.view(output.size(0), -1)
        multi_outputs = []
        for j, target_network_layer in enumerate(self.output):
            multi_outputs.append(target_network_layer(output))
        multi_outputs = torch.cat(multi_outputs, dim=1)
        return multi_outputs


class highwayNet(nn.Module):

    ## Initialization
    def __init__(self):
        super(highwayNet, self).__init__()

        ## Sizes of network layers
        self.encoder_size = 64
        self.decoder_size = 128
        self.grid_size = (13, 3)
        self.soc_conv_depth = 64
        self.conv_3x1_depth = 16
        self.dyn_embedding_size = 32
        self.input_embedding_size = 32
        self.soc_embedding_size = (((self.grid_size[0]-4)+1)//2)*self.conv_3x1_depth

        ## Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)

        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)

        # Convolutional social pooling layer and social embedding layer
        self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,1))
        self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0))

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()

        # self.fc1 = nn.Linear(in_features=112, out_features=32, bias=True)
        self.fc1 = nn.Linear(in_features=112, out_features=128, bias=True)
        #self.fc2 = nn.Linear(in_features=512, out_features=1024, bias=True)

    ## Forward Pass
    def forward(self,hist,nbrs,masks):

        ## Forward pass hist:
        _,(hist_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])))

        ## Forward pass nbrs
        _, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        ## Masked scatter
        soc_enc = torch.zeros_like(masks).float()
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)
        soc_enc = soc_enc.permute(0,3,2,1)

        ## Apply convolutional social pooling:
        soc_enc = self.soc_maxpool(self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc)))))
        soc_enc = soc_enc.view(-1,self.soc_embedding_size)

        ## Concatenate encodings:
        enc = nn.functional.relu(torch.cat((soc_enc,hist_enc),1))

        out_fc1 = nn.functional.relu(self.fc1(enc))
        #out_fc2 = nn.functional.relu(self.fc2(out_fc1))

        return out_fc1
