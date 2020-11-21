import sys
import os
import torch
import torch.distributed as dist
import warnings
import torch.distributed
import numpy as np
import random
import faulthandler
import torch.multiprocessing as mp
import time
from models.networks_regression import HyperRegression
from args import get_args
from torch.backends import cudnn
from utils import AverageValueMeter, set_random_seed, resume, save
from data_regression import ExampleData
import matplotlib.pyplot as plt


faulthandler.enable()


def main_worker(gpu, save_dir, args):
    # basic setup
    cudnn.benchmark = True
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = HyperRegression(args)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    start_epoch = 0
    optimizer = model.make_optimizer(args)
    if args.resume_checkpoint is None and os.path.exists(os.path.join(save_dir, 'checkpoint-latest.pt')):
        args.resume_checkpoint = os.path.join(save_dir, 'checkpoint-latest.pt')  # use the latest checkpoint
    if args.resume_checkpoint is not None:
        if args.resume_optimizer:
            model, optimizer, start_epoch = resume(
                args.resume_checkpoint, model, optimizer, strict=(not args.resume_non_strict))
        else:
            model, _, start_epoch = resume(
                args.resume_checkpoint, model, optimizer=None, strict=(not args.resume_non_strict))
        print('Resumed from: ' + args.resume_checkpoint)

    # main training loop
    start_time = time.time()
    point_nats_avg_meter = AverageValueMeter()
    if args.distributed:
        print("[Rank %d] World size : %d" % (args.rank, dist.get_world_size()))

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs):
        print("Epoch starts:")
        data = ExampleData()
        train_loader = torch.utils.data.DataLoader(
            dataset=data, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True)
        for bidx, data in enumerate(train_loader):
            x, y = data
            x = x.float().to(args.gpu).unsqueeze(1)
            y = y.float().to(args.gpu).unsqueeze(1).unsqueeze(2)
            step = bidx + len(train_loader) * epoch
            model.train()
            recon_nats = model(x, y, optimizer, step, None)
            point_nats_avg_meter.update(recon_nats.item())
            if step % args.log_freq == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print("[Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] PointNats %2.5f"
                      % (args.rank, epoch, bidx, len(train_loader),duration, point_nats_avg_meter.avg))
        # save visualizations
        kk = 3
        if (epoch + 1) % args.viz_freq == 0:
            # reconstructions
            model.eval()
            x = torch.from_numpy(np.linspace(0, kk, num=100)).float().to(args.gpu).unsqueeze(1)
            _, y = model.decode(x, 100)
            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            x = np.expand_dims(x, 1).repeat(100, axis=1).flatten()
            y = y.flatten()
            figs, axs = plt.subplots(1, 1, figsize=(12, 12))
            plt.xlim([0, kk])
            plt.ylim([-2, 2])
            plt.scatter(x, y)
            plt.savefig(os.path.join(save_dir, 'images', 'tr_vis_sampled_epoch%d-gpu%s.png' % (epoch, args.gpu)))
            plt.clf()
        if (epoch + 1) % args.save_freq == 0:
            save(model, optimizer, epoch + 1,
                 os.path.join(save_dir, 'checkpoint-%d.pt' % epoch))
            save(model, optimizer, epoch + 1,
                 os.path.join(save_dir, 'checkpoint-latest.pt'))


def main():
    # command line args
    args = get_args()
    save_dir = os.path.join("checkpoints", args.log_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'images'))

    with open(os.path.join(save_dir, 'command.sh'), 'w') as f:
        f.write('python -X faulthandler ' + ' '.join(sys.argv))
        f.write('\n')

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.sync_bn:
        assert args.distributed

    print("Arguments:")
    print(args)
    main_worker(args.gpu, save_dir, args)


if __name__ == '__main__':
    main()
