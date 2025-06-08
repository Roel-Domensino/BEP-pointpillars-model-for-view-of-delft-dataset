import argparse
import os
import torch
from tqdm import tqdm
import pdb
import torch.nn.functional as F
import numpy as np

from torch.amp import autocast, GradScaler
from pointpillars.utils import setup_seed
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillars
from pointpillars.loss import Loss
from torch.utils.tensorboard import SummaryWriter


from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader
from vod.frame import FrameTransformMatrix

def filter_within_fov(xyz, max_distance=50.0, max_angle_deg=32.0):
    """
    Filters points within radial distance and angular FoV.
    xyz: Tensor of shape (N, 3) or (N, 7), in radar or LiDAR coordinate frame.
    Returns mask (bool tensor) of shape (N,).
    """
    x, y = xyz[:, 0], xyz[:, 1]
    r = torch.sqrt(x ** 2 + y ** 2)
    angle = torch.atan2(y, x) * 180 / np.pi
    mask = (r <= max_distance) & (angle >= -max_angle_deg) & (angle <= max_angle_deg)
    return mask

def compute_accuracy(predictions, ground_truth, nclasses):
    """
    Computes the accuracy for multi-class classification.
    predictions: Tensor of predicted class labels.
    ground_truth: Tensor of ground truth class labels.
    nclasses: Number of classes in the classification task.
    """
    # Get the predicted class labels (assuming softmax is already applied)
    pred_labels = predictions.argmax(dim=-1)
    
    # Ignore the negative class (background)
    correct = (pred_labels == ground_truth) & (ground_truth != nclasses)  # Ignore background class
    accuracy = correct.sum().item() / (ground_truth != nclasses).sum().item()
    
    return accuracy

def radar_to_camera_coords(radar_coords, frame_data):
    import numpy as np
    from vod.frame import homogeneous_transformation, FrameTransformMatrix

    radar_coords = np.asarray(radar_coords)
    if radar_coords.shape[1] == 3:
        ones = np.ones((radar_coords.shape[0], 1))
        radar_coords = np.hstack((radar_coords, ones))

    transforms = FrameTransformMatrix(frame_data)
    return homogeneous_transformation(radar_coords, transforms.t_camera_radar)


def transform_bbox_7dof_radar_to_camera(bbox_7dof, transform_matrix):
    """
    Transforms a 3D bounding box from radar frame to camera frame.
    bbox_7dof: (7,) [x, y, z, w, h, l, yaw]
    transform_matrix: (4, 4) homogenous matrix t_camera_radar
    Returns:
        Transformed bbox_7dof in camera frame.
    """
    import numpy as np

    # 1. Transform the center point
    center = np.array([*bbox_7dof[:3], 1.0])  # [x, y, z, 1]
    transformed_center = transform_matrix @ center
    x, y, z = transformed_center[:3]
    
    # 2. Extract rotation part (3x3) of the transform
    R = transform_matrix[:3, :3]

    # 3. Compute yaw in camera frame
    # The yaw is around Z in radar; get direction vector
    yaw = bbox_7dof[6]
    direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    new_direction = R @ direction
    new_yaw = np.arctan2(new_direction[1], new_direction[0])  # camera yaw

    # 4. Dimensions: Usually stay same unless axis flips (check dataset specs)
    w, l, h = bbox_7dof[3:6]
    y=y- h / 2  # Adjust y from bottom to center
    return np.array([x, y, z, h, w, l, new_yaw], dtype=np.float32)

def bbox_lidar2camera(bboxes, tr_velo_to_cam, r0_rect):
    '''
    bboxes: shape=(N, 7)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 7)
    '''
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([y_size, z_size, x_size], axis=1)
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = r0_rect @ tr_velo_to_cam
    xyz = extended_xyz @ rt_mat.T
    bboxes_camera = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return bboxes_camera



def save_predictions(result, all_val_ids, frame_ptr, args):
    # Directory to save predictions
    root_dir = '/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC'
    pred_save_dir = os.path.join('/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC', 'predictions')
    pred_save_dir_2 = os.path.join('/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC', 'predictions_6000')
    os.makedirs(pred_save_dir, exist_ok=True)
    os.makedirs(pred_save_dir_2, exist_ok=True)

    # Label ID to class name mapping
    label_id_to_name = {
        0: 'Car',
        1: 'Pedestrian', 
        2: 'Cyclist', 
        3: 'rider',
        4: 'bicycle',
        5: 'moped_scooter',
        6: 'motor',
        7: 'ride_other',
        8: 'bicycle_rack'
    }

    # Get prediction results from the result dictionary
    print(f"result_type = {type(result)}")
    print(f"result = {result}")

    #radar_bboxes = result['lidar_bboxes']
    #labels = result['labels']
    #scores = result['scores']

    # Number of predictions
    #total_preds = radar_bboxes.shape[0]  # shape [B*N, 7]
    batch_size = args.batch_size
    batch_frame_ids = all_val_ids[frame_ptr: frame_ptr + batch_size]

    frame_ptr += batch_size  # Update pointer

    # Start iterating over each frame in the batch
    start_idx = 0
    for batch_idx in range(len(batch_frame_ids)):
        frame_id = batch_frame_ids[batch_idx]  # Use the correct frame ID
        file_path = os.path.join(pred_save_dir_2, f"{frame_id}.txt")

        # Get the current frame's prediction results
        current_frame_result = result[batch_idx]

        #print(f"current_frame_result_type = {type(current_frame_result)}")
        #print(f"current_frame_result = {current_frame_result}")

        if isinstance(current_frame_result, tuple):
            print(f"No predictions for frame {frame_id}")
            continue
        # Extract predictions for the current frame
        radar_bboxes = current_frame_result['lidar_bboxes']
        labels = current_frame_result['labels']
        scores = current_frame_result['scores']

        # Number of predictions for this frame
        num_preds = radar_bboxes.shape[0]
        #end_idx = start_idx + num_preds

        # Slice this frame's data
        #boxes = radar_bboxes[start_idx:end_idx]  # Shape: [N, 7]
        #frame_labels = labels[start_idx:end_idx]  # Shape: [N]
        #frame_scores = scores[start_idx:end_idx]  # Shape: [N]

        # Apply score thresholding (confidence threshold)
        conf_thresh = 0.15
        keep = scores > conf_thresh
        boxes = radar_bboxes[keep]
        frame_labels = labels[keep]
        frame_scores = scores[keep]

        kitti_locations = KittiLocations(root_dir=root_dir)
        frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number=str(frame_id).zfill(5))
        transforms = FrameTransformMatrix(frame_data)

        # Extract center coordinates
        centers_radar = boxes[:, :3]  # (x, y, z)
        #centers_camera = radar_to_camera_coords(centers_radar, frame_data)
        
        boxes = np.array([
            transform_bbox_7dof_radar_to_camera(box, transforms.t_camera_radar)
            for box in boxes
        ])
        #boxes=transform_bbox_7dof_radar_to_camera(boxes, transforms.t_camera_radar)
        #print(f"boxes = {boxes}")

        
        # Save predictions to text file
        with open(file_path, 'w') as f:
            for i in range(boxes.shape[0]):
                cls_name = label_id_to_name[frame_labels[i].item()]
                score = frame_scores[i].item()
                x, y, z, h, w, l, yaw = boxes[i].tolist()
                #x=centers_camera[i][0]
                #y=centers_camera[i][1]
                #z=centers_camera[i][2]
                #h, w, l = h, w, l  # Assuming dz, dy, dx are height, width, length
                f.write(f"{cls_name} 0 0 -1 0 0 0 0 {h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {yaw:.2f} {score:.4f}\n")

        #start_idx = end_idx  # Update index for the next frame

    return frame_ptr  # Return the updated frame pointer to continue processing the next batch

def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)


def main(args):
    setup_seed()
    loss_log={}
    class_counter = torch.zeros(args.nclasses + 1, dtype=torch.int32).cuda()
    scaler = GradScaler()

    train_dataset = Kitti(data_root=args.data_root,
                          split='train')
    val_dataset = Kitti(data_root=args.data_root,
                        split='val')
    train_dataloader = get_dataloader(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers,
                                      shuffle=True)
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)

    if not args.no_cuda:
        pointpillars = PointPillars(nclasses=args.nclasses).cuda()
    else:
        pointpillars = PointPillars(nclasses=args.nclasses)
    loss_func = Loss()
    
    monitor_val_loss=False


    # Load frame ID list from file (assuming one ID per line, e.g., '00000')
    with open('/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/PointPillars-main_dup_less_classes/Ids_val.txt', 'r') as f:
        all_val_ids = [line.strip() for line in f.readlines()]
    

    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr
    optimizer = torch.optim.AdamW(params=pointpillars.parameters(), 
                                  lr=init_lr, 
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                    max_lr=init_lr*10, 
                                                    total_steps=max_iters, 
                                                    pct_start=0.4, 
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True, 
                                                    base_momentum=0.95*0.895, 
                                                    max_momentum=0.95,
                                                    div_factor=10)
    saved_logs_path = os.path.join(args.saved_path, 'summary')
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints_less_classes')
    os.makedirs(saved_ckpt_path, exist_ok=True)

    for epoch in range(args.max_epoch):
        print('=' * 20, epoch, '=' * 20)
        train_step, val_step = 0, 0
        for i, data_dict in enumerate(tqdm(train_dataloader)):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            optimizer.zero_grad()

            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']

            
            #print(f"batched_labels = {batched_labels}")
            #print(batched_labels) 
            #batched_labels_tensor = torch.tensor(batched_labels)
            #print(f"distribution = {torch.bincount(batched_labels_tensor)}")
            #print(f"batched_pts = {batched_pts}")

           

            #print(f"Min value: {min_val.item()}")
            #print(f"Max value: {max_val.item()}")
            #print(f"batched_gt_bboxes = {batched_gt_bboxes}")
            #print(f"batched_labels = {batched_labels}")
            batched_difficulty = data_dict['batched_difficulty']
            polar_coords_list = []
            for pts in batched_pts:
                coords = pts[:, :3]  # (x, y, z)
                x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
                r = torch.sqrt(x**2 + y**2)
                theta = torch.atan2(y, x)  # angle in radians
                polar_coords = torch.stack((r, theta, z), dim=1)
                polar_coords_list.append(polar_coords)
            #batched_pts = [
            #torch.cat([polar_coords_list[i], batched_pts[i][:, 3:]], dim=1)  # Concatenate other features if any
            #for i in range(len(batched_pts))
            #]

            with autocast(device_type='cuda'):
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                    pointpillars(batched_pts=batched_pts, 
                                mode='train',
                                batched_gt_bboxes=batched_gt_bboxes, 
                                batched_gt_labels=batched_labels)
                


                anchors_per_position = 63  # (7 rotations × 9 sizes or classes, depending how you built anchors)
                nclasses = 9


                #print("bbox_cls_pred.shape before reshape:", bbox_cls_pred.shape)
                #print("args.nclasses =", args.nclasses)
                #print("anchors_per_position =", bbox_cls_pred.shape[1] // args.nclasses)

                                # Classification prediction
                #bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1)  # (B, H, W, C)
                #bbox_cls_pred = bbox_cls_pred.reshape(-1, anchors_per_position, args.nclasses)

                # Box regression prediction
                #bbox_pred = bbox_pred.permute(0, 2, 3, 1)
                #bbox_pred = bbox_pred.reshape(-1, anchors_per_position, 7)

                # Direction classification
                #bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1)
                #bbox_dir_cls_pred = bbox_dir_cls_pred.reshape(-1, anchors_per_position, 2)


                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
                #print(f"bbox_pred.shape = {bbox_pred.shape}")
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
                
                #print(f"Shape of bbox_pred after: {bbox_pred.shape}")          # Should be [batch_size * grid_size, 7]
                #print(f"Shape of bbox_dir_cls_pred: {bbox_dir_cls_pred.shape}")  # Should be [batch_size * grid_size, 2]
                
                batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                #print(f"batched_bbox_labels.shape before = {batched_bbox_labels.shape}")
                batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
                batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
                if torch.any((batched_bbox_labels >= args.nclasses) & (batched_label_weights > 0)):
                    invalid_labels = batched_bbox_labels[(batched_bbox_labels >= args.nclasses) & (batched_label_weights > 0)]
                    #print(f"Found invalid labels: {invalid_labels.unique()} (nclasses = {args.nclasses})")
                    #raise ValueError("Some labels are out of bounds for the number of classes.")
                
                # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)

                #print(f"Shape of bbox_cls_pred: {bbox_cls_pred.shape}")  # Should be [batch_size * grid_size, args.nclasses]
                #print(f"batched_bbox_labels.shape = {batched_bbox_labels.shape}")

                #print(f"bbox_cls_pred = {bbox_cls_pred}")
                #print(f"batched_bbox_labels = {batched_bbox_labels}")

                #print(f"Shape of bbox_pred: {bbox_pred.shape}")  # Should be [batch_size * grid_size, args.nclasses]
                #print(f"batched_bbox_reg.shape = {batched_bbox_reg.shape}")

                #print(f"Shape of bbox_dir_cls_pred: {bbox_dir_cls_pred.shape}")  # Should be [batch_size * grid_size, args.nclasses]
                #print(f"batched_dir_labels.shape = {batched_dir_labels.shape}")



                unique_values = torch.unique(batched_bbox_labels)
                
                
                #print(f"batched_bbox_labels = {batched_bbox_labels}")
                #print(f"Unique values in batched_bbox_labels: {unique_values}")
                #print(f"args.nclasses = {args.nclasses}")
                pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
                #print(f"Shape of pos_idx: {pos_idx.shape}")  # Should match batch_size * grid_size

                #print(f"pos_idx = {pos_idx}")
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
                bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
                batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]

                num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]
                #if 'class_counter' not in globals():
                #    class_counter = torch.zeros(args.nclasses + 1, dtype=torch.int32).cuda()

                # Clone to avoid modifying original labels
                labels_used = batched_bbox_labels.clone()
                # Cap values at args.nclasses (background is assigned this number)
                labels_used[labels_used >= args.nclasses] = args.nclasses
                counts = torch.bincount(labels_used, minlength=args.nclasses + 1)
                class_counter += counts
                #print("Positive samples:", (batched_bbox_labels < args.nclasses).sum().item())
                #print("Negative samples:", (batched_bbox_labels == args.nclasses).sum().item())

                loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                    bbox_pred=bbox_pred,
                                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                                    batched_labels=batched_bbox_labels, 
                                    num_cls_pos=num_cls_pos, 
                                    batched_bbox_reg=batched_bbox_reg, 
                                    batched_dir_labels=batched_dir_labels)
                
                probabilities = F.softmax(bbox_cls_pred, dim=-1)
                #print(bbox_cls_pred[:10]) 
                # Print the probabilities of the first 10 examples
                #print(probabilities[:10])
                loss = loss_dict['total_loss']
                print(f"loss train = {loss}")
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            #loss.backward()
            #print(f"loss_after = {loss}")
            # torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=35)
            #optimizer.step()
            #scheduler.step()

            global_step = epoch * len(train_dataloader) + train_step + 1

            if global_step % args.log_freq == 0:
                save_summary(writer, loss_dict, global_step, 'train',
                             lr=optimizer.param_groups[0]['lr'], 
                             momentum=optimizer.param_groups[0]['betas'][0])
            train_step += 1
        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            torch.save(pointpillars.state_dict(), os.path.join(saved_ckpt_path, f'epoch_{epoch+1}.pth'))

        #if epoch % 2 == 0:
        #    continue
        frame_ptr = 0  # A pointer to keep track of which IDs we’re using
        pointpillars.eval()
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(val_dataloader)):
                if not args.no_cuda:
                    # move the tensors to the cuda
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = data_dict[key][j].cuda()
                
                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                
                batched_difficulty = data_dict['batched_difficulty']
                with autocast(device_type='cuda'):
                    if monitor_val_loss:
                        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                            pointpillars(batched_pts=batched_pts, 
                                    mode='train',
                                    batched_gt_bboxes=batched_gt_bboxes, 
                                    batched_gt_labels=batched_labels)
                        batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                        batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                        batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                        batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)

                        pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
                        bbox_pred = bbox_pred[pos_idx]
                        batched_bbox_reg = batched_bbox_reg[pos_idx]
                        bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1]) * torch.cos(batched_bbox_reg[:, -1])
                        batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1]) * torch.sin(batched_bbox_reg[:, -1])
                        bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                        batched_dir_labels = batched_dir_labels[pos_idx]

                        num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                        bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                        batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                        batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                        loss_dict = loss_func(
                            bbox_cls_pred=bbox_cls_pred,
                            bbox_pred=bbox_pred,
                            bbox_dir_cls_pred=bbox_dir_cls_pred,
                            batched_labels=batched_bbox_labels,
                            num_cls_pos=num_cls_pos,
                            batched_bbox_reg=batched_bbox_reg,
                            batched_dir_labels=batched_dir_labels
                        )

                        loss = loss_dict['total_loss']
                        print(f"loss val = {loss}")

                        global_step = epoch * len(val_dataloader) + val_step + 1
                        if global_step % args.log_freq == 0:
                            save_summary(writer, loss_dict, global_step, 'val')
                    else:
                        results = \
                        pointpillars(batched_pts=batched_pts, 
                                    mode='val',
                                    batched_gt_bboxes=batched_gt_bboxes, 
                                    batched_gt_labels=batched_labels)
                        frame_ptr = save_predictions(results, all_val_ids, frame_ptr, args)
                # Apply softmax to logits
                probabilities = F.softmax(bbox_cls_pred, dim=-1)
                #print(bbox_cls_pred[:10]) 
                # Print the probabilities of the first 10 examples
                #print(probabilities[:10])
                # Print the first 10 predicted class probabilities
                #print(torch.argmax(bbox_cls_pred, dim=-1)[:10])  # Print the top predicted class labels
                #probabilities = F.softmax(bbox_cls_pred, dim=-1)
                #print(f"accuracy = {compute_accuracy(probabilities,batched_bbox_labels,args.nclasses)}")

                

                


                
                global_step = epoch * len(val_dataloader) + val_step + 1
                if global_step % args.log_freq == 0:
                    save_summary(writer, loss_dict, global_step, 'val')
                val_step += 1
        loss_log[epoch]=loss
        print("Final class counts used in training (excluding background):")
        for cls_id in range(args.nclasses):
            print(f"Class {cls_id}: {class_counter[cls_id].item()} samples")
        print(f"Background (ignored class): {class_counter[args.nclasses].item()} samples")        
        pointpillars.train()
    print("Training complete. Log of losses per epoch:")
    for epoch, loss in loss_log.items():
        print(f"Epoch {epoch}: Loss = {loss.item()}")
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC/radar', 
                        help='your data root for kitti')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=9)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=2)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
