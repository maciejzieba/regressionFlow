from models.networks_regression_SDD import HyperRegression
from args import get_args
import torch
import os
import cv2

from data_regression_SDD import SDDData, decode_obj
import numpy as np
from utils import draw_hyps


def main(args):
    model = HyperRegression(args)
    model = model.cuda()
    resume_checkpoint = args.resume_checkpoint
    print("Resume Path:%s" % resume_checkpoint)
    checkpoint = torch.load(resume_checkpoint)
    model_serialize = checkpoint['model']
    model.load_state_dict(model_serialize)
    model.eval()
    save_path = os.path.join(os.path.split(resume_checkpoint)[0], 'results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    data_test = SDDData(split='test')
    nll_px_sum = 0
    nll_py_sum = 0
    counter = 0.0
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
            log_py, log_px = model.get_logprob(x, y_gt)
            log_py = log_py.cpu().detach().numpy().squeeze()
            log_px = log_px.cpu().detach().numpy().squeeze()
            nll_px_sum = nll_px_sum + -1.0 * log_px
            nll_py_sum = nll_py_sum + -1.0 * log_py
            counter = counter + 1.0
            y_pred = y_pred.cpu().detach().numpy().squeeze()
            # y_pred[y_pred < 0] = 0
            # y_pred[y_pred >= 0.98] = 0.98
            testing_sequence = data_test.dataset.scenes[data_test.test_id].sequences[bidx]
            objects_list = []
            for k in range(3):
                objects_list.append(decode_obj(testing_sequence.objects[k], testing_sequence.id))
            objects = np.stack(objects_list, axis=0)
            gt_object = decode_obj(testing_sequence.objects[-1], testing_sequence.id)
            drawn_img_hyps = draw_hyps(testing_sequence.imgs[-1], y_pred, gt_object, objects, normalize=False)
            cv2.imwrite(os.path.join(save_path, str(session_id) + '-' + str(bidx) + '-hyps.jpg'), drawn_img_hyps)
    print("Sum log_p_x: " + str(nll_px_sum/counter))
    print("Sum log_p_y: " + str(nll_py_sum/counter))
    #evaluate_gen_2(args)
    #evaluate_recon_3(args)


if __name__ == '__main__':
    args = get_args()
    main(args)
