import sys
import os
import torch
import cv2
import torch.distributed as dist
import warnings
import torch.distributed
import numpy as np
import random
import faulthandler
import torch.multiprocessing as mp
import time
from models.networks_regression_NGSIM import HyperRegression
from torch import optim
from args import get_args
from torch.backends import cudnn
from utils import AverageValueMeter, set_random_seed, resume, save
from data_regression_NGSIM import ngsimDataset

from utils import draw_hyps

faulthandler.enable()


def main_worker(gpu, save_dir, ngpus_per_node, args):
    # basic setup
    cudnn.benchmark = True
    normalize = False
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

    # initialize datasets and loaders


    # initialize the learning rate scheduler
    if args.scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.exp_decay)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 2, gamma=0.1)
    elif args.scheduler == 'linear':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - 0.5 * args.epochs) / float(0.5 * args.epochs)
            return lr_l
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        assert 0, "args.schedulers should be either 'exponential' or 'linear'"

    # main training loop
    start_time = time.time()
    entropy_avg_meter = AverageValueMeter()
    latent_nats_avg_meter = AverageValueMeter()
    point_nats_avg_meter = AverageValueMeter()
    if args.distributed:
        print("[Rank %d] World size : %d" % (args.rank, dist.get_world_size()))

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    data = ngsimDataset(os.path.join(args.data_dir, 'TrainSet.mat'))
    train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                               shuffle=True,num_workers=8, collate_fn=data.collate_fn)
    for epoch in range(start_epoch, args.epochs):
        # adjust the learning rate
        if (epoch + 1) % args.exp_decay_freq == 0:
            scheduler.step(epoch=epoch)

        # train for one epoch
        print("Epoch starts:")
        for bidx, data in enumerate(train_loader):
            if bidx < 1000:
                hist, nbrs, mask, _, _, fut, op_mask = data
                hist = hist.float().to(args.gpu)
                nbrs = nbrs.float().to(args.gpu)
                op_mask = op_mask.float().to(args.gpu)
                mask = mask.type(torch.ByteTensor).to(args.gpu)
                fut = fut.float().to(args.gpu)
                y = fut[-1].float().to(args.gpu).unsqueeze(1)
                op_mask = op_mask[-1, :, 0].unsqueeze(1)
                #y += 0.01*torch.randn(y.shape[0], y.shape[1], y.shape[2]).to(args.gpu)
                step = bidx + len(train_loader) * epoch
                model.train()
                recon_nats = model(hist, nbrs, mask, op_mask, y, optimizer)
                point_nats_avg_meter.update(recon_nats.item())
                if step % args.log_freq == 0:
                    duration = time.time() - start_time
                    start_time = time.time()
                    print("[Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] PointNats %2.5f"
                          % (args.rank, epoch, bidx, len(train_loader),duration, point_nats_avg_meter.avg))
                                # print("Memory")
                                # print(process.memory_info().rss / (1024.0 ** 3))
            else:
                break
        # save visualizations
        # if (epoch + 1) % args.viz_freq == 0:
        #     # reconstructions
        #     model.eval()
        #     for bidx, data in enumerate(test_loader):
        #         x, _ = data
        #         x = x.float().to(args.gpu)
        #         _, y_pred = model.decode(x, 100)
        #         y_pred = y_pred.cpu().detach().numpy().squeeze()
        #         # y_pred[y_pred < 0] = 0
        #         # y_pred[y_pred >= 0.98] = 0.98
        #         testing_sequence = data_test.dataset.scenes[data_test.test_id].sequences[bidx]
        #         objects_list = []
        #         for k in range(3):
        #             objects_list.append(decode_obj(testing_sequence.objects[k], testing_sequence.id))
        #         objects = np.stack(objects_list, axis=0)
        #         gt_object = decode_obj(testing_sequence.objects[-1], testing_sequence.id)
        #         drawn_img_hyps = draw_hyps(testing_sequence.imgs[-1], y_pred, gt_object, objects, normalize)
        #         cv2.imwrite(os.path.join(save_dir, 'images', str(bidx) + '-' + str(epoch) + '-hyps.jpg'), drawn_img_hyps)
        if (epoch + 1) % args.save_freq == 0:
            save(model, optimizer, epoch + 1,
                 os.path.join(save_dir, 'checkpoint-%d.pt' % epoch))
            save(model, optimizer, epoch + 1,
                 os.path.join(save_dir, 'checkpoint-latest.pt'))


def main():
    # command line args
    args = get_args()
    if args.root_dir is None:
        save_dir = os.path.join("checkpoints", args.log_name)
    else:
        save_dir = os.path.join(args.root_dir,"checkpoints", args.log_name)
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

    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(save_dir, ngpus_per_node, args))
    else:
        main_worker(args.gpu, save_dir, ngpus_per_node, args)


if __name__ == '__main__':
    main()
