from pprint import pprint
from sklearn.svm import LinearSVC
from math import log, pi
import os
import torch
import torch.distributed as dist
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import io
from PIL import Image
from sklearn.mixture import GaussianMixture as GMM
# from wemd import computeWEMD

class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def gaussian_log_likelihood(x, mean, logvar, clip=True):
    if clip:
        logvar = torch.clamp(logvar, min=-4, max=3)
    a = log(2 * pi)
    b = logvar
    c = (x - mean) ** 2 / torch.exp(logvar)
    return -0.5 * torch.sum(a + b + c)


def bernoulli_log_likelihood(x, p, clip=True, eps=1e-6):
    if clip:
        p = torch.clamp(p, min=eps, max=1 - eps)
    return torch.sum((x * torch.log(p)) + ((1 - x) * torch.log(1 - p)))


def kl_diagnormal_stdnormal(mean, logvar):
    a = mean ** 2
    b = torch.exp(logvar)
    c = -1
    d = -logvar
    return 0.5 * torch.sum(a + b + c + d)


def kl_diagnormal_diagnormal(q_mean, q_logvar, p_mean, p_logvar):
    # Ensure correct shapes since no numpy broadcasting yet
    p_mean = p_mean.expand_as(q_mean)
    p_logvar = p_logvar.expand_as(q_logvar)

    a = p_logvar
    b = - 1
    c = - q_logvar
    d = ((q_mean - p_mean) ** 2 + torch.exp(q_logvar)) / torch.exp(p_logvar)
    return 0.5 * torch.sum(a + b + c + d)


# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
def truncated_normal(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()

    rt /= world_size
    return rt


def standard_normal_logprob(z):
    log_z = -0.5 * log(2 * pi)
    return log_z - z.pow(2) / 2


def standard_laplace_logprob(z):
    log_z = - 1.0*log(2)
    return log_z - torch.abs(z)


def log_normal_logprob(z, mu, var):
     log_norm = torch.log(torch.norm(z, dim=2))
     logz = -1.0 * log(2) - 1.5 * log(2 * pi) - 0.5 * log(var)
     return logz - 3.0 * log_norm - (log_norm - mu).pow(2) / (2 * var)


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Visualization
def visualize_point_clouds(pts, gtr, idx, pert_order=[0, 1, 2]):
    pts = pts.cpu().detach().numpy()[:, pert_order]
    gtr = gtr.cpu().detach().numpy()[:, pert_order]

    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Sample:%s" % idx)
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Ground Truth:%s" % idx)
    ax2.scatter(gtr[:, 0], gtr[:, 1], gtr[:, 2], s=5)

    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res


# Augmentation
def apply_random_rotation(pc, rot_axis=1):
    B = pc.shape[0]

    theta = np.random.rand(B) * 2 * np.pi
    zeros = np.zeros(B)
    ones = np.ones(B)
    cos = np.cos(theta)
    sin = np.sin(theta)

    if rot_axis == 0:
        rot = np.stack([
            cos, -sin, zeros,
            sin, cos, zeros,
            zeros, zeros, ones
        ]).T.reshape(B, 3, 3)
    elif rot_axis == 1:
        rot = np.stack([
            cos, zeros, -sin,
            zeros, ones, zeros,
            sin, zeros, cos
        ]).T.reshape(B, 3, 3)
    elif rot_axis == 2:
        rot = np.stack([
            ones, zeros, zeros,
            zeros, cos, -sin,
            zeros, sin, cos
        ]).T.reshape(B, 3, 3)
    else:
        raise Exception("Invalid rotation axis")
    rot = torch.from_numpy(rot).to(pc)

    # (B, N, 3) mul (B, 3, 3) -> (B, N, 3)
    pc_rotated = torch.bmm(pc, rot)
    return pc_rotated, rot, theta


def validate_classification(loaders, model, args):
    train_loader, test_loader = loaders

    def _make_iter_(loader):
        iterator = iter(loader)
        return iterator

    tr_latent = []
    tr_label = []
    for data in _make_iter_(train_loader):
        tr_pc = data['train_points']
        tr_pc = tr_pc.cuda() if args.gpu is None else tr_pc.cuda(args.gpu)
        latent = model.encode(tr_pc)
        label = data['cate_idx']
        tr_latent.append(latent.cpu().detach().numpy())
        tr_label.append(label.cpu().detach().numpy())
    tr_label = np.concatenate(tr_label)
    tr_latent = np.concatenate(tr_latent)

    te_latent = []
    te_label = []
    for data in _make_iter_(test_loader):
        tr_pc = data['train_points']
        tr_pc = tr_pc.cuda() if args.gpu is None else tr_pc.cuda(args.gpu)
        latent = model.encode(tr_pc)
        label = data['cate_idx']
        te_latent.append(latent.cpu().detach().numpy())
        te_label.append(label.cpu().detach().numpy())
    te_label = np.concatenate(te_label)
    te_latent = np.concatenate(te_latent)

    clf = LinearSVC(random_state=0)
    clf.fit(tr_latent, tr_label)
    test_pred = clf.predict(te_latent)
    test_gt = te_label.flatten()
    acc = np.mean((test_pred == test_gt).astype(float)) * 100.
    res = {'acc': acc}
    print("Acc:%s" % acc)
    return res


def validate_conditioned(loader, model, args, max_samples=None, save_dir=None):
    from metrics.evaluation_metrics import EMD_CD
    all_idx = []
    all_sample = []
    all_ref = []
    ttl_samples = 0
    iterator = iter(loader)

    for data in iterator:
        # idx_b, tr_pc, te_pc = data[:3]
        idx_b, tr_pc, te_pc = data['idx'], data['train_points'], data['test_points']
        tr_pc = tr_pc.cuda() if args.gpu is None else tr_pc.cuda(args.gpu)
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)

        if tr_pc.size(1) > te_pc.size(1):
            tr_pc = tr_pc[:, :te_pc.size(1), :]
        out_pc = model.reconstruct(tr_pc, num_points=te_pc.size(1))

        # denormalize
        m, s = data['mean'].float(), data['std'].float()
        m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
        s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)
        all_idx.append(idx_b)

        ttl_samples += int(te_pc.size(0))
        if max_samples is not None and ttl_samples >= max_samples:
            break

    # Compute MMD and CD
    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    print("[rank %s] Recon Sample size:%s Ref size: %s" % (args.rank, sample_pcs.size(), ref_pcs.size()))

    if save_dir is not None and args.save_val_results:
        smp_pcs_save_name = os.path.join(save_dir, "smp_recon_pcls_gpu%s.npy" % args.gpu)
        ref_pcs_save_name = os.path.join(save_dir, "ref_recon_pcls_gpu%s.npy" % args.gpu)
        np.save(smp_pcs_save_name, sample_pcs.cpu().detach().numpy())
        np.save(ref_pcs_save_name, ref_pcs.cpu().detach().numpy())
        print("Saving file:%s %s" % (smp_pcs_save_name, ref_pcs_save_name))

    res = EMD_CD(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)
    mmd_cd = res['MMD-CD'] if 'MMD-CD' in res else None
    mmd_emd = res['MMD-EMD'] if 'MMD-EMD' in res else None

    print("MMD-CD  :%s" % mmd_cd)
    print("MMD-EMD :%s" % mmd_emd)

    return res


def validate_sample(loader, model, args, max_samples=None, save_dir=None):
    from metrics.evaluation_metrics import compute_all_metrics, jsd_between_point_cloud_sets as JSD
    all_sample = []
    all_ref = []
    ttl_samples = 0

    iterator = iter(loader)

    for data in iterator:
        idx_b, te_pc = data['idx'], data['test_points']
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
        _, out_pc = model.sample(te_pc.size(0), te_pc.size(1), gpu=args.gpu)

        # denormalize
        m, s = data['mean'].float(), data['std'].float()
        m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
        s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)

        ttl_samples += int(te_pc.size(0))
        if max_samples is not None and ttl_samples >= max_samples:
            break

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    print("[rank %s] Generation Sample size:%s Ref size: %s"
          % (args.rank, sample_pcs.size(), ref_pcs.size()))

    if save_dir is not None and args.save_val_results:
        smp_pcs_save_name = os.path.join(save_dir, "smp_syn_pcls_gpu%s.npy" % args.gpu)
        ref_pcs_save_name = os.path.join(save_dir, "ref_syn_pcls_gpu%s.npy" % args.gpu)
        np.save(smp_pcs_save_name, sample_pcs.cpu().detach().numpy())
        np.save(ref_pcs_save_name, ref_pcs.cpu().detach().numpy())
        print("Saving file:%s %s" % (smp_pcs_save_name, ref_pcs_save_name))

    res = compute_all_metrics(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)
    pprint(res)

    sample_pcs = sample_pcs.cpu().detach().numpy()
    ref_pcs = ref_pcs.cpu().detach().numpy()
    jsd = JSD(sample_pcs, ref_pcs)
    jsd = torch.tensor(jsd).cuda() if args.gpu is None else torch.tensor(jsd).cuda(args.gpu)
    res.update({"JSD": jsd})
    print("JSD     :%s" % jsd)
    return res


def save(model, optimizer, epoch, path):
    d = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(d, path)


def resume(path, model, optimizer=None, strict=True):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'], strict=strict)
    start_epoch = ckpt['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, start_epoch


def validate(test_loader, model, epoch, writer, save_dir, args, clf_loaders=None):
    model.eval()

    # Make epoch wise save directory
    if writer is not None and args.save_val_results:
        save_dir = os.path.join(save_dir, 'epoch-%d' % epoch)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None

    # classification
    if args.eval_classification and clf_loaders is not None:
        for clf_expr, loaders in clf_loaders.items():
            with torch.no_grad():
                clf_val_res = validate_classification(loaders, model, args)

            for k, v in clf_val_res.items():
                if writer is not None and v is not None:
                    writer.add_scalar('val_%s/%s' % (clf_expr, k), v, epoch)

    # samples
    if args.use_latent_flow:
        with torch.no_grad():
            val_sample_res = validate_sample(
                test_loader, model, args, max_samples=args.max_validate_shapes,
                save_dir=save_dir)

        for k, v in val_sample_res.items():
            if not isinstance(v, float):
                v = v.cpu().detach().item()
            if writer is not None and v is not None:
                writer.add_scalar('val_sample/%s' % k, v, epoch)

    # reconstructions
    with torch.no_grad():
        val_res = validate_conditioned(
            test_loader, model, args, max_samples=args.max_validate_shapes,
            save_dir=save_dir)
    for k, v in val_res.items():
        if not isinstance(v, float):
            v = v.cpu().detach().item()
        if writer is not None and v is not None:
            writer.add_scalar('val_conditioned/%s' % k, v, epoch)


def visualize_2Dimage(image, samples, idx):
    if not isinstance(image, np.ndarray):
        image = image.cpu().detach().numpy()
    if not isinstance(samples, np.ndarray):
        samples = samples.cpu().detach().numpy()
    generated_image = np.zeros(image.shape)
    coordinates = np.round(samples[:, 0:2]*np.array([image.shape[0], image.shape[1]]))
    for i in range(generated_image.shape[0]):
        for j in range(generated_image.shape[1]):
            generated_image[i, j] = samples
    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(121)
    ax1.set_title("Sample:%s" % idx)
    ax1.imshow(image)

    ax2 = fig.add_subplot(122)
    ax2.set_title("Ground Truth:%s" % idx)
    ax2.imshow(samples)

    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    # res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res


def draw_hyps(img_path, hyps, gt_object, objects, normalize=True, hist_rects_color=(0, 0, 255), cvt_color: bool = False):
    img = cv2.imread(img_path)
    if cvt_color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # draw object history
    tranparency = {0: 0.2, 1: 0.5, 2: 1.0}
    for i in range(3):
        overlay = img.copy()
        cv2.rectangle(overlay, (int(objects[i, 0, 0, 0, 0]), int(objects[i, 0, 0, 1, 0])),
                      (int(objects[i, 0, 0, 2, 0]), int(objects[i, 0, 0, 3, 0])), hist_rects_color, -1)
        img = cv2.addWeighted(overlay, tranparency[i], img, 1 - tranparency[i], 0)

    # draw the ground truth future
    cv2.rectangle(img, (int(gt_object[0, 0, 0, 0]), int(gt_object[0, 0, 1, 0])),
                  (int(gt_object[0, 0, 2, 0]), int(gt_object[0, 0, 3, 0])), (255, 0, 255), -1)
    gt_width = gt_object[0, 0, 2, 0] - gt_object[0, 0, 0, 0]
    gt_height = gt_object[0, 0, 3, 0] - gt_object[0, 0, 1, 0]
    #
    # # draw hypotheses (different futures)
#     for h in hyps:
#         print(h)
#         x1 = int(h[0, 0, 0, 0])
#         y1 = int(h[0, 1, 0, 0])
#
#         color = (0, 255, 0)
    for k in range(hyps.shape[0]):
        if normalize:
            x1 = int(img.shape[1] * hyps[k, 0])
            y1 = int(img.shape[0] * hyps[k, 1])
        else:
            x1 = int(hyps[k, 0])
            y1 = int(hyps[k, 1])
        color = (0, 255, 0)
        x2 = int(x1 + gt_width)
        y2 = int(y1 + gt_height)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    return img


def draw_ngsim_plots(hist, y_gt, y_pred, op_mask, dir):
    hist_np = hist.cpu().detach().numpy()
    y_gt = y_gt.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    op_mask = op_mask.cpu().detach().numpy()
    for k in range(hist.shape[1]):
        plt.scatter(hist_np[:, k, 1], hist_np[:, k, 0], marker='^')
        plt.scatter(y_pred[k, :, 1], y_pred[k, :, 0], marker='*', c='#ff7f0e')
        plt.scatter(y_gt[k, 0, 1], y_gt[k, 0, 0], marker='o', c='#bcbd22')
        plt.savefig(os.path.join(dir, str(k) + '_' + str(op_mask[k]) + '.png'))
        plt.close()


def draw_sdd_heatmap(
        objects,
        gt_object,
        testing_sequence,
        log_px_pred,
        X, Y,
        save_path
):
    def transparent_cmap(cmap, N=255):
        "Copy colormap and set alpha values"
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.clip(np.linspace(0, 1.0, N + 4), 0, 1.0)
        return mycmap
    img = draw_hyps(testing_sequence.imgs[-1], np.empty((0,2)), gt_object, np.array(objects), normalize=False, hist_rects_color=(255, 0,0), cvt_color=True)

    Z = log_px_pred.reshape(-1)
    Z = np.exp(Z)
    vmax = np.max(Z)
    vmin = np.min(Z)
    h, w, _ = img.shape
    plt.figure(figsize=(w // 25, h // 25,))
    plt.imshow(img)
    plt.contourf(X, Y,Z.reshape(X.shape), vmin=vmin , vmax=vmax, cmap=transparent_cmap(plt.cm.jet), levels=20)

    plt.axis("off")
    plt.savefig(save_path,format='png', bbox_inches='tight', pad_inches=0 )

