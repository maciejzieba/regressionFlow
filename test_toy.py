import torch
from models.networks_regression import HyperRegression
import os
import numpy as np
import matplotlib.pyplot as plt
from args import get_args


def main(args):
    save_dir = os.path.dirname(args.resume_checkpoint)
    model = HyperRegression(args)
    model = model.cuda()
    resume_checkpoint = args.resume_checkpoint
    print("Resume Path:%s" % resume_checkpoint)
    checkpoint = torch.load(resume_checkpoint)
    model_serialize = checkpoint['model']
    model.load_state_dict(model_serialize)
    # reconstructions
    kk = 2.1
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
    plt.savefig(os.path.join(save_dir, 'image.png'))
    plt.clf()


if __name__ == '__main__':
    args = get_args()
    main(args)