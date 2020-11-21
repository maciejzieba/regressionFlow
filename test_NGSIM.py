from models.networks_regression_NGSIM import HyperRegression
from args import get_args
import torch
import os

from data_regression_NGSIM import ngsimDataset
import numpy as np


def main(args):
    model = HyperRegression(args)
    model = model.to(args.gpu)
    resume_checkpoint = args.resume_checkpoint
    print("Resume Path:%s" % resume_checkpoint)
    checkpoint = torch.load(resume_checkpoint)
    model_serialize = checkpoint['model']
    model.load_state_dict(model_serialize)
    model.eval()
    save_path = os.path.join(os.path.split(resume_checkpoint)[0], 'results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    tsSet = ngsimDataset(os.path.join(args.data_dir, 'TestSet.mat'))
    test_loader = torch.utils.data.DataLoader(tsSet, batch_size=10000, shuffle=True, num_workers=8, collate_fn=tsSet.collate_fn)
    nll_px_sum = 0
    nll_py_sum = 0
    counter = 0.0
    for bidx, data in enumerate(test_loader):
        hist, nbrs, mask, _, _, fut, op_mask = data
        hist = hist.float().to(args.gpu)
        nbrs = nbrs.float().to(args.gpu)
        op_mask = op_mask.float().to(args.gpu)
        op_mask = op_mask[-1, :, 0]
        mask = mask.type(torch.ByteTensor).to(args.gpu)
        fut = fut.float().to(args.gpu)
        y_gt = fut[-1].float().to(args.gpu).unsqueeze(1)
        log_py, log_px = model.get_logprob(hist, nbrs, mask, y_gt)
        log_py = log_py.squeeze()*op_mask
        log_px = log_px.squeeze()*op_mask
        log_py = log_py.cpu().detach().numpy().squeeze()
        log_px = log_px.cpu().detach().numpy().squeeze()
        counter = counter + torch.sum(op_mask).cpu().detach().numpy()
        nll_px_sum = nll_px_sum + -1.0 * np.sum(log_px)
        nll_py_sum = nll_py_sum + -1.0 * np.sum(log_py)
        print(str(-1.0 * np.sum(log_px)/torch.sum(op_mask).cpu().detach().numpy()))
        print(str(-1.0 * np.sum(log_py)/torch.sum(op_mask).cpu().detach().numpy()))
        print(str(np.sum(log_py-log_px)/torch.sum(op_mask).cpu().detach().numpy()))
        print("Batch [%2d/%2d]" % (bidx, len(test_loader)))
    print("Sum log_p_x: " + str(nll_px_sum/counter))
    print("Sum log_p_y: " + str(nll_py_sum/counter))


if __name__ == '__main__':
    args = get_args()
    main(args)
