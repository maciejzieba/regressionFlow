from models.networks_regression_SDD import HyperRegression
from args import get_args
import torch
import os
import cv2
import numpy as np

from data_regression_SDD import SDDData, decode_obj
import numpy as np
from utils import draw_hyps
from scipy.stats import wasserstein_distance as emd_distance
from pyemd import emd_samples
from collections import defaultdict
from wemd import computeWEMD
import mmfp_utils

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
        
    # hardcoded data path
    data_test = SDDData(width=512, height=512, split='test_small', normalize=False, root="cpi_generation/cpi/")
    
    nll_px_sum = 0
    nll_py_sum = 0
    counter = 0.0
    
    metrics = {
        "car": defaultdict(list),
        "ped": defaultdict(list)
    }
    
    for session_id in range(len(data_test.dataset.scenes)):
        data_test.test_id = session_id
        print("scene", session_id, "n_datas", len(data_test))

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
        
        
        for bidx, (name, x, y_gt) in enumerate([
            ("ped", pedestrian_x, pedestrian_gt),
            ("car", car_x, car_gt)
        ]):
            
            y_gt = torch.stack(y_gt).float().to(args.gpu)
            x = x.float().to(args.gpu)

            _, y_pred = model.decode(x, 100 )

            log_py, log_px, _ = model.get_logprob(
                x.repeat(len(y_gt), 1, 1, 1), 
                y_gt
            )
            log_py = log_py.cpu().detach().numpy().squeeze()
            log_px = log_px.cpu().detach().numpy().squeeze()
            
            metrics[name]["nll_px"].extend(-1.0 * log_px)
            metrics[name]["nll_py"].extend(-1.0 * log_py)
            
            y_gt_np = y_gt.detach().cpu().numpy().reshape((-1, 2))
            
            y_pred = y_pred.cpu().detach().numpy().squeeze()
            
            
            oracle_err = np.array([
                mmfp_utils.compute_oracle_FDE(
                    y_pred.reshape(1, *y_pred.shape,1,1), 
                    yg.reshape(1,1, 2, 1,)
                )
                for yg in y_gt_np
                
            ])
            metrics[name]["oracle_err"].append(oracle_err.mean())
            print("oracle_err", oracle_err.mean())
            
#             emd = emd_samples(y_gt_np, y_pred)
            print(name)
            print(str(-1.0 * log_px.mean()))
            print(str(-1.0 * log_py.mean()))
#             print("emd", emd)
#             metrics[name]["emd"].append(emd)
            
            
            hist_gt, *_ = np.histogram2d(y_gt_np[:, 0], y_gt_np[:, 1], bins=np.linspace(0, 512, 512))
            hist_pred, *_ = np.histogram2d(y_pred[:, 0], y_pred[:, 1], bins=np.linspace(0, 512, 512))

            
            wemd = computeWEMD(hist_pred, hist_gt)
            print("wemd", wemd)
            metrics[name]["wemd"].append(wemd)
        
            
            testing_sequence = data_test.dataset.scenes[data_test.test_id].sequences[bidx]
            objects_list = []
            for k in range(3):
                objects_list.append(decode_obj(testing_sequence.objects[k], testing_sequence.id))
            objects = np.stack(objects_list, axis=0)
#             gt_object = decode_obj(testing_sequence.objects[-1], testing_sequence.id)
#             print(gt_object)
            gt_object = np.array([[[[0],[0], [0], [0], [bidx]]]]).astype(float) # mock it and draw dots instead
#             print(gt_object)
            drawn_img_hyps = draw_hyps(testing_sequence.imgs[2], y_pred, gt_object, objects, normalize=False)
            
            # draw ground truth samples in blue
            for (x1, y1) in y_gt_np:
                color = (255, 0, 0)
                cv2.circle(drawn_img_hyps, (x1, y1), 3, color, -1)
            cv2.imwrite(os.path.join(save_path, f"{session_id}-{bidx}-{name}-hyps.jpg"), drawn_img_hyps)

    
    total_mets = defaultdict(list)
    
    for k, mets in metrics.items():
        for name, nums in mets.items():
            print(f"Mean {k} {name}: ", np.array(nums).mean())
            total_mets[name].extend(nums)
    
    for name, nums in mets.items():
        print(f"Total mean {name}: ", np.array(nums).mean())
    


if __name__ == '__main__':
    args = get_args()
    main(args)
