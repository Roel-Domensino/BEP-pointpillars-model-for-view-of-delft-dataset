import os
import numpy as np
from vod.frame import KittiLocations, FrameDataLoader, FrameTransformMatrix, homogeneous_transformation
from scipy.spatial.transform import Rotation as R

# Paths
source_folder = "/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC/radar/training/label_2"
target_folder = "/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC/radar/training/label_2_radar_transformed"
root_dir = "/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC"

# Ensure the target folder exists
os.makedirs(target_folder, exist_ok=True)

# Classes to keep
valid_classes = {'Car', 'Pedestrian', 'bicycle', 'Cyclist', 'moped_scooter', 'motor', 'ride_other', 'rider', 'bicycle_rack'}

def transform_kitti_box_to_radar(parts, transforms):
    # Parse fields
    class_name = parts[0]
    h, w, l = map(float, parts[8:11])  # KITTI: h w l
    x, y, z = map(float, parts[11:14])  # In camera coords
    ry = float(parts[14])  # Rotation around Y-axis in camera

    # Adjust bottom-centered z to center z
    y_center = y - (h / 2)
    center_camera = np.array([[x, y_center, z, 1]])

    # Transform center to radar frame
    center_radar = homogeneous_transformation(center_camera, transforms.t_radar_camera)[:, :3].flatten()

    # Transform yaw vector from camera to radar frame
    R_cam_to_radar = transforms.t_radar_camera[:3, :3]
    yaw_vec_camera = np.array([np.cos(ry), 0, np.sin(ry)])
    yaw_vec_radar = R_cam_to_radar @ yaw_vec_camera
    yaw_radar = np.arctan2(yaw_vec_radar[1], yaw_vec_radar[0])

    # Dimensions to w, l, h format
    dims_radar = [w, l, h]

    # Rebuild line with updated dims, location and yaw at correct indices:
    # Original KITTI line layout:
    # 0: class, 1-7: other info, 8-10: h,w,l, 11-13: x,y,z, 14: rotation_y
    # We'll replace dims, location and rotation_y accordingly.
    new_parts = parts[:]

    new_parts[8] = f"{dims_radar[0]:.2f}"  # w
    new_parts[9] = f"{dims_radar[1]:.2f}"  # l
    new_parts[10] = f"{dims_radar[2]:.2f}"  # h

    new_parts[11] = f"{center_radar[0]:.2f}"
    new_parts[12] = f"{center_radar[1]:.2f}"
    new_parts[13] = f"{center_radar[2]:.2f}"

    new_parts[14] = f"{yaw_radar:.2f}"

    return new_parts

def process_all_files(source_folder, target_folder, root_dir):
    # Initialize KittiLocations once
    kitti_locations = KittiLocations(root_dir=root_dir)

    for filename in os.listdir(source_folder):
        if not filename.endswith('.txt'):
            continue

        frame_id = filename.split('.')[0]
        output_lines = []

        # Load frame data and transforms for this frame
        frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number=str(frame_id).zfill(5))
        transforms = FrameTransformMatrix(frame_data)

        # Process each line
        with open(os.path.join(source_folder, filename), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] not in valid_classes:
                    continue
                radar_parts = transform_kitti_box_to_radar(parts, transforms)
                output_lines.append(' '.join(radar_parts))

        # Write transformed file
        with open(os.path.join(target_folder, filename), 'w') as out_file:
            for line in output_lines:
                out_file.write(line + '\n')

        print(f"Transformed: {filename}")

# Run the transformation
process_all_files(source_folder, target_folder, root_dir)
