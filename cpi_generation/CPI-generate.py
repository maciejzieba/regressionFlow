import argparse
import os

import numpy as np
from Environment import Environment
from imageio import imwrite
from utils import writeFloat, locs_to_sdd_features

parser = argparse.ArgumentParser(
    description="Script for generating CPI data in format compatible with SDD file format.")
parser.add_argument("--output_folder", help='destination folder for the produced data',
                    default="cpi/train")  # test: cpi/test
parser.add_argument("--n_scenes", help='number of scenes to produce', default=20000)  # test: 54
parser.add_argument("--history", help='history (length of the image sequence to store)', default=3)
parser.add_argument("--n_gts", help='number of gts per image sequence', default=20)  # test: 1000
parser.add_argument("--dist", help='prediction horizon.', default=20)

args = parser.parse_args()

hist = int(args.history)

for i in range(int(args.n_scenes)):

    scene_name = f"scene_{i}"
    print(f"processing {scene_name}")
    scene_path = os.path.join(args.output_folder, scene_name)
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

                break
    with open(os.path.join(scene_path, "scene.txt"), "w") as f:
        f.writelines(scene_lines)
