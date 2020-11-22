import os
from collections import defaultdict

import cv2
import numpy as np
import torch

import mmfp_utils
from args import get_args
from data_regression_SDD import SDDData, decode_obj
from models.networks_regression_SDD import HyperRegression
from utils import draw_hyps
from wemd import computeWEMD

CPI_WIDTH = 512
CPI_HEIGHT = 512


def main(args):
    model = HyperRegression(args, input_width=CPI_WIDTH, input_height=CPI_HEIGHT)
    model = model.cuda()
    resume_checkpoint = args.resume_checkpoint
    print(f"Resume Path: {resume_checkpoint}")
    checkpoint = torch.load(resume_checkpoint)
    model_serialize = checkpoint['model']
    model.load_state_dict(model_serialize)
    model.eval()
    save_path = os.path.join(os.path.split(resume_checkpoint)[0], 'results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    data_test = SDDData(width=CPI_WIDTH, height=CPI_HEIGHT, split='test', normalize=False, root=args.data_dir)

    metrics = {
        "car": defaultdict(list),
        "ped": defaultdict(list)
    }

    for scene_id in range(len(data_test.dataset.scenes)):
        data_test.test_id = scene_id
        print("scene", scene_id, "n_datas", len(data_test))

        test_loader = torch.utils.data.DataLoader(
            dataset=data_test, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True)

        pedestrian_gt = []
        car_gt = []

        pedestrian_x = None
        car_x = None
        for bidx, data in enumerate(test_loader):
            x, y_gt = data

            if bidx % 2 == 0:
                pedestrian_x = x
                pedestrian_gt.append(y_gt)
            else:
                car_x = x
                car_gt.append(y_gt)

        for bidx, (obj_name, x, y_gt) in enumerate([
            ("ped", pedestrian_x, pedestrian_gt),
            ("car", car_x, car_gt)
        ]):

            y_gt = torch.stack(y_gt).float().to(args.gpu)
            x = x.float().to(args.gpu)

            _, y_pred = model.decode(x, 100)

            log_py, log_px, _ = model.get_logprob(
                x.repeat(len(y_gt), 1, 1, 1),
                y_gt
            )
            log_py = log_py.cpu().detach().numpy().squeeze()
            log_px = log_px.cpu().detach().numpy().squeeze()

            metrics[obj_name]["nll_px"].extend(-1.0 * log_px)
            metrics[obj_name]["nll_py"].extend(-1.0 * log_py)

            y_gt_np = y_gt.detach().cpu().numpy().reshape((-1, 2))

            y_pred = y_pred.cpu().detach().numpy().squeeze()

            oracle_err = np.array([
                mmfp_utils.compute_oracle_FDE(
                    y_pred.reshape(1, *y_pred.shape, 1, 1),
                    yg.reshape(1, 1, 2, 1, )
                )
                for yg in y_gt_np

            ])
            metrics[obj_name]["oracle_err"].append(oracle_err.mean())

            hist_gt, *_ = np.histogram2d(y_gt_np[:, 0], y_gt_np[:, 1], bins=np.linspace(0, 512, 512))
            hist_pred, *_ = np.histogram2d(y_pred[:, 0], y_pred[:, 1], bins=np.linspace(0, 512, 512))

            wemd = computeWEMD(hist_pred, hist_gt)
            metrics[obj_name]["wemd"].append(wemd)

            log_metrics = {
                "oracle_err": oracle_err.mean(),
                "wemd": wemd,
                "nll_px": (-1 * log_px).mean(),
                "nll_py": (-1 * log_py).mean()
            }
            print(f"scene {scene_id}", obj_name, log_metrics)
            testing_sequence = data_test.dataset.scenes[data_test.test_id].sequences[bidx]
            objects_list = []
            for k in range(3):
                objects_list.append(decode_obj(testing_sequence.objects[k], testing_sequence.id))
            objects = np.stack(objects_list, axis=0)
            gt_object = np.array([[[[0], [0], [0], [0], [bidx]]]]).astype(float)  # mock it and draw dots instead
            drawn_img_hyps = draw_hyps(testing_sequence.imgs[2], y_pred, gt_object, objects, normalize=False)

            for (x1, y1) in y_gt_np:
                color = (255, 0, 0)
                cv2.circle(drawn_img_hyps, (x1, y1), 3, color, -1)
            cv2.imwrite(os.path.join(save_path, f"{scene_id}-{bidx}-{obj_name}-hyps.jpg"), drawn_img_hyps)

    total_metrics = defaultdict(list)

    for k, mets in metrics.items():
        for obj_name, nums in mets.items():
            print(f"Mean {k} {obj_name}: ", np.array(nums).mean())
            total_metrics[obj_name].extend(nums)

    for obj_name, nums in mets.items():
        print(f"Total mean {obj_name}: ", np.array(nums).mean())


if __name__ == '__main__':
    args = get_args()
    main(args)
