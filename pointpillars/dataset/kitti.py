import numpy as np
import os
import torch
from torch.utils.data import Dataset

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from pointpillars.utils import read_pickle, read_points
from pointpillars.dataset import point_range_filter, data_augment


def camera2radar(box, transforms):
    from vod.frame import homogeneous_transformation
    import numpy as np

    x, y, z, h, w, l, ry = box

    # Convert center from center to bottom
    y_center = y+(h/2)
    center_camera = np.array([[x, y_center, z, 1]])

    # Apply transformation
    center_radar = homogeneous_transformation(center_camera, transforms.t_radar_camera)[:, :3].flatten()

    # Rotate yaw vector
    R_cam_to_radar = transforms.t_radar_camera[:3, :3]
    yaw_vec_camera = np.array([np.cos(ry), 0, np.sin(ry)])
    yaw_vec_radar = R_cam_to_radar @ yaw_vec_camera
    yaw_radar = np.arctan2(yaw_vec_radar[1], yaw_vec_radar[0])

    # Return as NumPy array in format [x, y, z, w, l, h, yaw]
    return np.array([center_radar[0], center_radar[1], center_radar[2],
                     w, l, h, yaw_radar], dtype=np.float32)


class BaseSampler():
    def __init__(self, sampled_list, shuffle=True):
        self.total_num = len(sampled_list)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0
        #print(f"Initialized sampler with {self.total_num} samples")

    def available_samples(self):
        return self.total_num - self.idx

    def sample(self, num):
        #print(f"Attempting to sample {num} samples from index {self.idx}")
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx:self.idx+num]]
            self.idx += num
            #print(f"Samples retrieved: {ret.shape[0]}")
        else:
            ret = self.sampled_list[self.indices[self.idx:]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
            #print(f"Samples retrieved (wrapped around): {ret.shape[0]}")
        return ret


class Kitti(Dataset):

    #CLASSES = {
    #    'Pedestrian': 0, 
    #    'Cyclist': 1,
    #    'Car': 2
    #}
    CLASSES = {
        'Car': 0,
        'Pedestrian': 1, 
        'Cyclist': 2, 
        'rider': 3,
        'bicycle': 4,
        'moped_scooter': 5,
        'motor': 6,
        'ride_other': 7,
        'bicycle_rack': 8    
        }

    def __init__(self, data_root, split, pts_prefix='velodyne_reduced'):
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.pts_prefix = pts_prefix
        self.data_infos = read_pickle(os.path.join(data_root, f'kitti_infos_{split}.pkl'))
        self.sorted_ids = list(self.data_infos.keys())
        #print(f"self_sorted_ids ={self.sorted_ids}")
        db_infos = read_pickle(os.path.join(data_root, 'kitti_dbinfos_train.pkl'))
        # Find all unique class names across the dataset annotations
        all_class_names = set()

        for data_info in self.data_infos.values():
            annos = data_info['annos']
            if 'name' in annos:
                all_class_names.update(annos['name'])

# Compare against known classes
        known_class_names = set(self.CLASSES.keys())
        unknown_class_names = all_class_names - known_class_names

        #print(f"\n📦 All classes in dataset: {sorted(all_class_names)}")
        #print(f"✅ Known classes (used): {sorted(known_class_names)}")
        #print(f"❌ Unknown (unused) classes: {sorted(unknown_class_names)}\n")
        #print(f"Number of samples for Pedestrian BEFORE: {len(db_infos.get('Pedestrian', []))}")
        db_infos = self.filter_db(db_infos)
        #print(f"Number of samples for Pedestrian AFTER: {len(db_infos.get('Pedestrian', []))}")
        #print(f"Keys in dbinfos{db_infos.keys()}")
        #print(f"db_info{db_infos}")
        db_sampler = {}
        for cat_name in self.CLASSES:
            
            db_sampler[cat_name] = BaseSampler(db_infos[cat_name], shuffle=True)
        self.data_aug_config=dict(
            db_sampler=dict(
                db_sampler=db_sampler,
                sample_groups=dict(Car=1, Pedestrian=2, Cyclist=2, rider=2, bicycle=2, moped_scooter=2, motor=2, ride_other=1, bicycle_rack=1)
                #sample_groups=dict(Car=5, Pedestrian=5, bicycle=3)
                ),
            object_noise=dict(
                num_try=100,
                translation_std=[0.25, 0.25, 0],
                rot_range=[-0.15707963267, 0.15707963267]
                ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
                ), 
            point_range_filter=[0, -39.68, -3, 69.12, 39.68, 1],
            object_range_filter=[0, -39.68, -3, 69.12, 39.68, 1]              
        )

    def remove_dont_care(self, annos_info):
        keep_ids = [i for i, name in enumerate(annos_info['name']) if name != 'DontCare']
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info

    def filter_db(self, db_infos):
        # 1. filter_by_difficulty
        for k, v in db_infos.items():
            if k != "Cyclist":  # Don't filter Cyclist based on difficulty
                #db_infos[k] = [item for item in v if item['difficulty'] != -1]
                db_infos[k] = [item for item in v]
        else:
            # For Cyclist, allow difficulty -1
            db_infos[k] = [item for item in v]

        # 2. filter_by_min_points, dict(Car=5, Pedestrian=10, Cyclist=10)
        
        filter_thrs = dict(Car=3, Pedestrian=3, Cyclist=3, rider=3, bicycle=3, moped_scooter=3, motor=3, ride_other=3, bicycle_rack=3)
        #filter_thrs = dict(Car=3, Pedestrian=2, bicycle=2)
        for cat in self.CLASSES:
            filter_thr = filter_thrs[cat]
            #print(f"Filtering {cat} with threshold {filter_thr}")
            # Debugging: Print number of points for Cyclist
            #if cat == "Cyclist":
                #for item in db_infos[cat]:
                    #print(f"Cyclist item points: {item['num_points_in_gt']}")
            db_infos[cat] = [item for item in db_infos[cat] if item['num_points_in_gt'] >= filter_thr]
            #print(f"Number of {cat} after filtering: {len(db_infos[cat])}")

        return db_infos

    def __getitem__(self, index):
        data_info = self.data_infos[self.sorted_ids[index]]
        #print(f"index", index)
        #print(f"self.sorted_ids", self.sorted_ids[index])
        image_info, calib_info, annos_info = \
            data_info['image'], data_info['calib'], data_info['annos']
        #print(f"annos_info = {annos_info}")
        # point cloud input
        velodyne_path = data_info['velodyne_path'].replace('velodyne', self.pts_prefix)
        pts_path = os.path.join(self.data_root, velodyne_path)
        pts = read_points(pts_path)
        # Extract x, y, and z coordinates
        x = pts[:, 0]  # All rows, first column (x-coordinate)
        y = pts[:, 1]  # All rows, second column (y-coordinate)
        z = pts[:, 2]  # All rows, third column (z-coordinate)

# Calculate min and max for x, y, z
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)

# Print the ranges
        #print(f"X range: {x_min:.2f} to {x_max:.2f}")
        #print(f"Y range: {y_min:.2f} to {y_max:.2f}")
        #print(f"Z range: {z_min:.2f} to {z_max:.2f}")
        #print(f"pts = {pts}")
        # calib input: for bbox coordinates transformation between Camera and Lidar.
        # because
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)

        # annotations input
        annos_info = self.remove_dont_care(annos_info)
        annos_name = annos_info['name']
        annos_location = annos_info['location']
        annos_dimension = annos_info['dimensions']
        rotation_y = annos_info['rotation_y']
        gt_bboxes = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
        #print("index = ", index)
        #print("gt_bboxes = ", gt_bboxes)

        from vod.frame import KittiLocations, FrameDataLoader, FrameTransformMatrix, homogeneous_transformation
        from scipy.spatial.transform import Rotation as R
        root_dir = "/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC"
        kitti_locations = KittiLocations(root_dir=root_dir)
        
        gt_bboxes_3d = np.zeros_like(gt_bboxes)  # Same shape and dtype as original
        #print(f"index = {index}")
        frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number=str(self.sorted_ids[index]).zfill(5))
        transforms = FrameTransformMatrix(frame_data)
        for id in range(len(gt_bboxes)):
            gt_bboxes_3d[id] = camera2radar(gt_bboxes[id], transforms)
        
        gt_labels = [self.CLASSES.get(name, -1) for name in annos_name]
        #print(f"gt_labels = {gt_labels}")
        data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': np.array(gt_labels), 
            'gt_names': annos_name,
            'difficulty': annos_info['difficulty'],
            'image_info': image_info,
            'calib_info': calib_info
        }
        if self.split in ['train', 'trainval']:
            #print("Before filtering GT boxes:", data_dict['gt_bboxes_3d'].shape)
            data_dict = data_augment(self.CLASSES, self.data_root, data_dict, self.data_aug_config)
            #print("After filtering GT boxes:", data_dict['gt_bboxes_3d'].shape)
            #data_dict = point_range_filter(data_dict, point_range=self.data_aug_config['point_range_filter'])
        else:
            data_dict = point_range_filter(data_dict, point_range=self.data_aug_config['point_range_filter'])

        return data_dict

    def __len__(self):
        return len(self.data_infos)
 

if __name__ == '__main__':
    import numpy as np

    kitti_data = Kitti(data_root='/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC/radar', split='train')

    # Initialize global min/max
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])

    for idx in range(len(kitti_data)):
        print(idx)
        data_dict = kitti_data.__getitem__(idx)
        pts = data_dict['pts']  # shape: [N, D]
        coords = pts[:, :3]     # x, y, z only

        current_min = coords.min(axis=0)
        current_max = coords.max(axis=0)

        global_min = np.minimum(global_min, current_min)
        global_max = np.maximum(global_max, current_max)

        if idx % 100 == 0:
            print(f"[{idx}] So far global min: {global_min}, max: {global_max}")

    print("\n=== Final Global Point Cloud Range ===")
    print(f"X range: {global_min[0]:.2f} to {global_max[0]:.2f}")
    print(f"Y range: {global_min[1]:.2f} to {global_max[1]:.2f}")
    print(f"Z range: {global_min[2]:.2f} to {global_max[2]:.2f}")


    #kitti_data = Kitti(data_root='/mnt/ssd1/lifa_rdata/det/kitti', 
    ##                   split='train')
    #kitti_data.__getitem__(9)
