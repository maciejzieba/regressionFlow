import os
import cv2

import numpy as np
import imgaug.augmenters as iaa
from torch.utils.data import Dataset


def readFloat(name):
    f = open(name, 'rb')

    if(f.readline().decode("utf-8"))  != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())
    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))
    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim == 2:
        data = np.transpose(data, (0, 1))
    elif dim == 3:
        data = np.transpose(data, (1, 2, 0))
    elif dim == 4:
        data = np.transpose(data, (2, 3, 1, 0))
    else:
        raise Exception('bad float file dimension: %d' % dim)

    return data


# read an image amd resize when needed
def decode_img(file_path, width=None, height=None):
    img = cv2.imread(file_path)

    img = img / 255.0
    img = np.subtract(img, 0.4)
    if width is not None and height is not None:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (0, 3, 1, 2))
    return img


# read the float file containing the object information
def decode_obj(file_path, id, coeff_x=1.0, coeff_y=1.0):
    object = np.expand_dims(np.expand_dims(np.expand_dims(readFloat(file_path)[id], 0), 0), 3).astype(np.float32)
    x_tl = object[:, :, 0:1, :] / coeff_x
    y_tl = object[:, :, 1:2, :] / coeff_y
    x_br = object[:, :, 2:3, :] / coeff_x
    y_br = object[:, :, 3:4, :] / coeff_y
    object = np.concatenate((x_tl, y_tl, x_br, y_br, object[:, :, 4:6, :]), axis=2)
    return object


class DataLoader():
    def __init__(self, path):
        self.path = path
        self.scenes = []
        self.load_scenes()

    def load_scenes(self):
        scenes_names = sorted(os.listdir(self.path))
        print(self.path, "scenes_names", len(scenes_names))
        for scene_name in scenes_names:
            if os.path.exists(os.path.join(self.path, scene_name, 'scene.txt')):
                    self.scenes.append(Scene(os.path.join(self.path, scene_name)))

        print("n_Seqs", sum([len(s.sequences) for s in self.scenes]))

class Scene():
    def __init__(self, scene_path):
        self.scene_path = scene_path
        self.img_ext = '-img-resized.jpg'
        self.sequences = []
        self.load_sequences()

    def parse_string(self, c):
        id, ss = c.split(' ')
        img_0, img_1, img_2, img_f = ss.split(',')
        return int(id), str(img_0).strip(), str(img_1).strip(), str(img_2).strip(), str(img_f).strip()

    def load_sequences(self):
        with open(os.path.join(self.scene_path, 'scene.txt')) as f:
            content = f.readlines()
            for c in content:
                id, img_0, img_1, img_2, img_f = self.parse_string(c)
                img_0_path = os.path.join(self.scene_path, 'imgs', '%s-img-resized.jpg' % img_0)
                img_1_path = os.path.join(self.scene_path, 'imgs', '%s-img-resized.jpg' % img_1)
                img_2_path = os.path.join(self.scene_path, 'imgs', '%s-img-resized.jpg' % img_2)
                img_f_path = os.path.join(self.scene_path, 'imgs', '%s-img-resized.jpg' % img_f)

                obj_0_path = os.path.join(self.scene_path, 'floats', '%s-features.float3' % img_0)
                obj_1_path = os.path.join(self.scene_path, 'floats', '%s-features.float3' % img_1)
                obj_2_path = os.path.join(self.scene_path, 'floats', '%s-features.float3' % img_2)
                obj_f_path = os.path.join(self.scene_path, 'floats', '%s-features.float3' % img_f)

                self.sequences.append(Sequence(id, [img_0_path, img_1_path, img_2_path, img_f_path],
                                               [obj_0_path, obj_1_path, obj_2_path, obj_f_path]))


class Sequence():
    def __init__(self, id, imgs, objects):
        self.id = id
        self.imgs = imgs
        self.objects = objects


class SDDData(Dataset):
    def __init__(self, width=320, height=576, split='train', test_id=0, normalize=True, root=None):

        self.split = split
        if self.split == 'train':
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, split)
        self.width = width
        self.height = height
        self.dataset = DataLoader(root)
        self.test_id = test_id
        self.normalize = normalize

    def __len__(self):
        if self.split == 'train':
            return len(self.dataset.scenes)
        else:
            return len(self.dataset.scenes[self.test_id].sequences)

    def __getitem__(self, idx):
        if self.split == 'train':
            j = np.random.choice(len(self.dataset.scenes), 1)[0]
            scene = self.dataset.scenes[j]
            i = np.random.choice(len(scene.sequences), 1)[0]
            testing_sequence = scene.sequences[i]
        else:
            scene = self.dataset.scenes[self.test_id]
            testing_sequence = scene.sequences[idx]
        objects_list = []
        imgs_list = []
        for k in range(3):
            objects_list.append(decode_obj(testing_sequence.objects[k], testing_sequence.id))
            imgs_list.append(decode_img(testing_sequence.imgs[k], width=self.width, height=self.height))
        objects = np.stack(objects_list, axis=0)
        imgs = np.stack(imgs_list, axis=0)
        gt_object = decode_obj(testing_sequence.objects[-1], testing_sequence.id)
        input = []
        for i in range(2, -1, -1):
            object = objects[i]
            mask = get_mask(object[0, 0, :, 0], self.width, self.height, fill_value=object[0, 0, 4, 0])
            input.append(imgs[i])
            input.append(mask)
        input = np.squeeze(np.concatenate(input, axis=1))
        output = get_avg_output(gt_object[0, 0, :, 0], self.width, self.height, self.normalize)
        # (1,1) (2, 0)

        if self.split == 'train':
            s = np.random.uniform(0, 1)
            if s > 0.5:
                input = np.flip(input, 2).copy()
                output[0] = self.width - output[0]
            s = np.random.uniform(0, 1)
            if s > 0.5:
                input = np.flip(input, 1).copy()
                output[1] = self.height - output[1]
        return input, output


def get_mask(indices, width, height, fill_value=1.0):# shape of indices is 5
    indices = indices.astype(int)
    tl_x = indices[0]
    tl_y = indices[1]
    bbox_width = indices[2]-indices[0]
    bbox_height = indices[3]-indices[1]
    ind_row = [tl_y, height - tl_y - bbox_height]
    ind_col = [tl_x, width - tl_x - bbox_width]
    padding = np.stack([ind_row, ind_col])
    input = np.ones([bbox_height, bbox_width]) * (fill_value + 1)
    padded = np.expand_dims(np.expand_dims(np.pad(input, padding, mode='constant'), axis=0), axis=0)
    return padded


def get_avg_output(indices, width, height, normalize=True):
    if normalize:
        return np.array([0.5*(indices[2] + indices[0])/width, 0.5*(indices[3] + indices[1])/height])
    else:
        return np.array([0.5*(indices[2] + indices[0]), 0.5*(indices[3] + indices[1])])


def main():
    root = 'data/SDD/test'
    width = 320
    height = 576
    dataset = DataLoader(root)
    for scene_index in range(len(dataset.scenes)):
        scene = dataset.scenes[scene_index]
        scene_name = scene.scene_path.split('/')[-1]
        print('---------------- Scene %s ---------------------' % scene_name)
        print(len(scene.sequences))
        for i in range(1):# range(len(scene.sequences)):
            testing_sequence = scene.sequences[i]
            objects_list = []
            imgs_list = []
            for k in range(3):
                objects_list.append(decode_obj(testing_sequence.objects[k], testing_sequence.id))
                imgs_list.append(decode_img(testing_sequence.imgs[k], width=width, height=height))
            objects = np.stack(objects_list, axis=0)
            imgs = np.stack(imgs_list, axis=0)
            gt_object = decode_obj(testing_sequence.objects[-1], testing_sequence.id)
            input = []
            for i in range(2, -1, -1):
                object = objects[i]
                mask = get_mask(object[0, 0, :, 0], width, height, fill_value=object[0, 0, 4, 0])
                input.append(imgs[i])
                input.append(mask)
            input = np.concatenate(input, axis=1)
            input = np.flip(input, 2)
            cv2.imwrite('color_img_1_f.jpg', np.transpose(255*input[0,0:3,:,:], [1, 2, 0]))
            cv2.imwrite('color_img_2_f.jpg', np.transpose(255*np.expand_dims(input[0,3,:,:],axis=0), [1, 2, 0]))
            cv2.imwrite('color_img_3_f.jpg', np.transpose(255*input[0,4:7,:,:], [1, 2, 0]))
            cv2.imwrite('color_img_4_f.jpg', np.transpose(255*np.expand_dims(input[0,7,:,:],axis=0), [1, 2, 0]))
            cv2.imwrite('color_img_5_f.jpg', np.transpose(255*input[0,8:11,:,:], [1, 2, 0]))
            cv2.imwrite('color_img_6_f.jpg', np.transpose(255*np.expand_dims(input[0,11,:,:], axis=0), [1, 2, 0]))
            output = get_avg_output(gt_object[0, 0, :, 0], width, height, False)
            output[1] = width - output[1]


if __name__ == '__main__':
    main()