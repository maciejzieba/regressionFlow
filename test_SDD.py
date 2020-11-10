from models.networks_regression_SDD import HyperRegression
from args import get_args
import torch
import os
import cv2

from data_regression_SDD import SDDData, decode_obj
import numpy as np
from utils import draw_hyps, draw_sdd_heatmap
import mmfp_utils
import pickle

def main(args):
    model = HyperRegression(args)
    model = model.cuda()
    resume_checkpoint = args.resume_checkpoint
    print("Resume Path:%s" % resume_checkpoint)
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
    oracle_err_sum = 0
    
    counter = 0.0
#     rows = []
    for session_id in range(len(data_test.dataset.scenes)):
        data_test.test_id = session_id
        test_loader = torch.utils.data.DataLoader(
            dataset=data_test, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True)
        for bidx, data in enumerate(test_loader):
#             if bidx not in [0,1]:
#                 continue
            x, y_gt = data
            x = x.float().to(args.gpu)
            y_gt = y_gt.float().to(args.gpu).unsqueeze(1)
            _, y_pred = model.decode(x, 10)
            
            log_py, log_px, _ = model.get_logprob(x, y_gt)
            
            _, _, height, width = x.shape
            
            #####
            x_sp = np.linspace(0, width - 1, width // 5)
            y = np.linspace(0, height - 1, height // 5)
            X, Y = np.meshgrid(x_sp, y)
            XX = np.array([X.ravel(), Y.ravel()]).T
            print("xx", XX.shape, y_pred.shape)
            _, _, (log_py_pred, log_px_pred)  = model.get_logprob(
                x, 
                torch.tensor(XX).unsqueeze(0).to(args.gpu)
            )
            #####
            
            print(log_py_pred.shape)
            log_py = log_py.cpu().detach().numpy().squeeze()
            log_px = log_px.cpu().detach().numpy().squeeze()
            print(str(session_id) + '-' + str(bidx) + '-hyps.jpg')
            print("nll_x", str(-1.0 * log_px))
            print("nll_y", str(-1.0 * log_py))
            print("nll_(x+y)",str(-1.0 * log_px + log_py))


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
            
#             semd = mmfp_utils.wemd_from_pred_samples(y_pred)
#             semd_sum += semd
            
            y_gt_np = y_gt.cpu().numpy()
            oracle_err = mmfp_utils.compute_oracle_FDE(
                y_pred.reshape(1, *y_pred.shape,1,1 ), 
                y_gt_np.reshape(1,1,2,1))
    
            oracle_err_sum +=oracle_err
            
            # print("EMD (Przemek multimodality measure)", multimod_emd)
            print("oracle err", oracle_err)
            row = {
                "session_id": session_id,
                "bidx": bidx,
                "nll_x": (-1 * log_px),
                "nll_y": (-1 * log_py),
                 "x": x.cpu().numpy(),
                "y_gt": y_gt.cpu().numpy(),
                "y_pred": y_pred,
                "objects": objects_list,
                "gt_object": gt_object,
                "drawn_img_hyps": drawn_img_hyps,
                "testing_sequence": testing_sequence,
                "log_px_pred": log_px_pred.detach().cpu().numpy(),
                "log_py_pred": log_py_pred.detach().cpu().numpy(),
                "xx": XX,
                "XY": (X,Y),

            }
            ht_img = draw_sdd_heatmap(**row)
            ht_img.save(os.path.join(save_path, str(session_id) + '-' + str(bidx) + '-heatmap.png'))
    
#             if bidx in [0,1]:
#                 rows.append(row)
            
    print("Mean log_p_x: " + str(nll_px_sum/counter))
    print("Mean log_p_y: " + str(nll_py_sum/counter))
    print("Mean multimod_emd:", multimod_emd_sum/counter)
    print("Mean Oracle Error:", oracle_err_sum / counter)
    #evaluate_gen_2(args)
    #evaluate_recon_3(args)
#     with open("sdd_results.pkl", "wb") as f:
#         pickle.dump(rows, f)
#     print(len(rows))



if __name__ == '__main__':
    args = get_args()
    main(args)
