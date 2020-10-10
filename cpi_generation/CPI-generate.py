import argparse
import os

import numpy as np
from Environment import Environment
from imageio import imwrite
from utils import writeFloat, locs_to_sdd_features

parser = argparse.ArgumentParser()
parser.add_argument("output_folder", help='destination folder for the produced data')
parser.add_argument("n_scenes", help='number of scenes to produce')
parser.add_argument("history", help='history (length of the image sequence to store)')
parser.add_argument("n_gts", help='number of gts per image sequence')
parser.add_argument("dist", help='prediction horizon')

args = parser.parse_args()

hist = int(args.history)


# def get_mask(objects):
#     indices0 = objects[0, 0, 0:3].astype(int)  # shape (3)
#     indices1 = objects[0, 0, 3:6].astype(int)  # shape (3)
#     ind_row0 = [indices0[0], 512 - indices0[0] - indices0[2]]
#     ind_row1 = [indices1[0], 512 - indices1[0] - indices1[2]]
#     ind_col0 = [indices0[1], 512 - indices0[1] - indices0[2]]
#     ind_col1 = [indices1[1], 512 - indices1[1] - indices1[2]]
#     padding0 = np.stack([ind_col0, ind_row0])
#     padding1 = np.stack([ind_col1, ind_row1])
#     input0 = np.ones([indices0[2], indices0[2]])
#     input1 = np.ones([indices1[2], indices1[2]])
#     padded0 = np.pad(input0, padding0, 'constant')
#     padded1 = np.pad(input1, padding1, 'constant')
#     padded = (padded0 + padded1) * 100
#     return padded


for i in range(int(args.n_scenes)):
    print('processing scene_%07d' % i)
    scene_path = os.path.join(args.output_folder, 'scene_%07d' % i)
    os.makedirs(scene_path, exist_ok=True)
    os.makedirs(f"{scene_path}/imgs", exist_ok=True)
    os.makedirs(f"{scene_path}/floats", exist_ok=True)

    env = Environment(512, 512)
    env.draw_cross_road()
    env.init_pedestrian()
    env.init_vehicle()
    env.draw_cross_road()
    env.next_state()

    for j in range(hist):
        env.next_state()
        env.draw_cross_road()
        env.draw_objects()
        sample = env.get_image()
        locs = env.get_objects_locations()
        prefix = f"hist_{j}"
        imwrite(os.path.join(scene_path, f'imgs/{prefix}-img-resized.jpg'), np.array(sample))
        sdd_feats = locs_to_sdd_features(locs)

        writeFloat(os.path.join(scene_path, f'floats/{prefix}-features.float3'), sdd_feats)

        # imwrite(os.path.join(scene_path, '-sample%03d.png' % (j)), np.array(sample))
        # writeFloat(os.path.join(scene_path, '%03d-%06d-%03d-objects.float3' % (hist - 1, k, l)), locs)

    scene_lines = []

    # multiple ground truths for the same input sequence
    for k in range(int(args.n_gts)):
        current_env = env.get_copy()
        current_env.draw_cross_road()
        current_env.draw_objects()
        for l in range(40):
            current_env.next_state()
            current_env.draw_cross_road()
            current_env.draw_objects()
            if (l + 1) == int(args.dist):
                locs = current_env.get_objects_locations()
                sample = current_env.get_image()

                prefix = f"future_{k}"
                sdd_feats = locs_to_sdd_features(locs)
                writeFloat(os.path.join(scene_path, f'floats/{prefix}-features.float3'), sdd_feats)
                imwrite(os.path.join(scene_path, f'imgs/{prefix}-img-resized.jpg'), np.array(sample))
                for i in range(len(sdd_feats)):
                    scene_lines.append(f"{i} hist_0,hist_1,hist_2,{prefix}\n")

                # writeFloat(os.path.join(scene_path, '%03d-%06d-%03d-objects.float3' % (hist - 1, k, l)), locs)
                break
    with open(os.path.join(scene_path, "scene.txt"), "w") as f:
        f.writelines(scene_lines)