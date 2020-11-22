import json
import os

import cv2
import numpy as np
import torch

import mmfp_utils
from args import get_args
from data_regression_SDD import SDDData, decode_obj
from models.networks_regression_SDD import HyperRegression
from utils import draw_hyps, draw_sdd_heatmap


def get_grid_logprob(
        height, width, x, model
):
    x_sp = np.linspace(0, width - 1, width // 1)
    y = np.linspace(0, height - 1, height // 1)
    X, Y = np.meshgrid(x_sp, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    _, _, (log_py_grid, log_px_grid) = model.get_logprob(
        x,
        torch.tensor(XX).unsqueeze(0).to(args.gpu)
    )
    return (X, Y), (log_px_grid.detach().cpu().numpy(), log_py_grid.detach().cpu().numpy())


def main(args):
    model = HyperRegression(args)
    model = model.cuda()
    resume_checkpoint = args.resume_checkpoint
    print(f"Resume Path: {resume_checkpoint}")
    checkpoint = torch.load(resume_checkpoint, map_location="cpu")
    model_serialize = checkpoint['model']
    model.load_state_dict(model_serialize)
    model.eval()
    save_path = os.path.join(os.path.split(resume_checkpoint)[0], 'results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    data_test = SDDData(split='test', normalize=False, root=args.data_dir)
    nll_px_sum = 0
    nll_py_sum = 0

    multimod_emd_sum = 0

    counter = 0.0

    results = []
    for session_id in range(len(data_test.dataset.scenes)):
        data_test.test_id = session_id
        test_loader = torch.utils.data.DataLoader(
            dataset=data_test, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True)
        for bidx, data in enumerate(test_loader):

            x, y_gt = data
            x = x.float().to(args.gpu)
            y_gt = y_gt.float().to(args.gpu).unsqueeze(1)
            _, y_pred = model.decode(x, 1000)

            log_py, log_px, _ = model.get_logprob(x, y_gt)

            log_py = log_py.cpu().detach().numpy().squeeze()
            log_px = log_px.cpu().detach().numpy().squeeze()

            hyps_name = f"{session_id}-{bidx}-hyps.jpg"
            print(hyps_name)
            print("nll_x", str(-1.0 * log_px))
            print("nll_y", str(-1.0 * log_py))
            print("nll_(x+y)", str(-1.0 * log_px + log_py))

            nll_px_sum = nll_px_sum + -1.0 * log_px
            nll_py_sum = nll_py_sum + -1.0 * log_py
            counter = counter + 1.0
            y_pred = y_pred.cpu().detach().numpy().squeeze()

            testing_sequence = data_test.dataset.scenes[data_test.test_id].sequences[bidx]
            objects_list = []
            for k in range(3):
                objects_list.append(decode_obj(testing_sequence.objects[k], testing_sequence.id))
            objects = np.stack(objects_list, axis=0)
            gt_object = decode_obj(testing_sequence.objects[-1], testing_sequence.id)
            drawn_img_hyps = draw_hyps(testing_sequence.imgs[-1], y_pred, gt_object, objects, normalize=False)
            cv2.imwrite(os.path.join(save_path, hyps_name), drawn_img_hyps)

            multimod_emd = mmfp_utils.wemd_from_pred_samples(y_pred)
            multimod_emd_sum += multimod_emd
            print("multimod_emd", multimod_emd)

            _, _, height, width = x.shape

            (X, Y), (log_px_grid, log_py_grid) = get_grid_logprob(height, width, x, model)

            draw_sdd_heatmap(
                objects=objects_list,
                gt_object=gt_object,
                testing_sequence=testing_sequence,
                log_px_pred=log_px_grid,
                X=X, Y=Y,
                save_path=os.path.join(save_path, f"{session_id}-{bidx}-heatmap.png")
            )

            result_row = {
                "session_id": session_id,
                "bidx": bidx,
                "nll_x": float(-1.0 * log_px),
                "nll_y": float(-1.0 * log_py),
                "multimod_emd": float(multimod_emd)
            }
            results.append(result_row)

    print("Mean log_p_x: ", nll_px_sum / counter)
    print("Mean log_p_y: ", nll_py_sum / counter)
    print("Mean multimod_emd:", multimod_emd_sum / counter)

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(results, f)

if __name__ == '__main__':
    args = get_args()
    main(args)
