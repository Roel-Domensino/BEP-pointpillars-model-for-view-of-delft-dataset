import pdb
import numpy as np
import torch
from pointpillars.utils import limit_period, iou2d_nearest


class Anchors():
    def __init__(self, ranges, sizes, rotations):
        assert len(ranges) == len(sizes)
        self.ranges = ranges
        self.sizes = sizes
        self.rotations = rotations

    def get_anchors(self, feature_map_size, anchor_range, anchor_size, rotations):
        device = feature_map_size.device
        x_centers = torch.linspace(anchor_range[0], anchor_range[3], feature_map_size[1] + 1, device=device)
        y_centers = torch.linspace(anchor_range[1], anchor_range[4], feature_map_size[0] + 1, device=device)
        z_centers = torch.linspace(anchor_range[2], anchor_range[5], 1 + 1, device=device)

        x_shift = (x_centers[1] - x_centers[0]) / 2
        y_shift = (y_centers[1] - y_centers[0]) / 2
        z_shift = (z_centers[1] - z_centers[0]) / 2

        x_centers = x_centers[:feature_map_size[1]] + x_shift  # (W,)
        y_centers = y_centers[:feature_map_size[0]] + y_shift  # (H,)
        z_centers = z_centers[:1] + z_shift                    # (1,)

        # Make sure rotations is a tensor
        if not torch.is_tensor(rotations):
            rotations = torch.tensor(rotations, device=device)

        # Create meshgrid
        mesh_x, mesh_y, mesh_z, mesh_rot = torch.meshgrid(x_centers, y_centers, z_centers, rotations, indexing='ij')
        mesh_x = mesh_x.unsqueeze(-1)
        mesh_y = mesh_y.unsqueeze(-1)
        mesh_z = mesh_z.unsqueeze(-1)
        mesh_rot = mesh_rot.unsqueeze(-1)

        # Shape: (W, H, 1, num_rot, 1)
        anchor_size_tensor = torch.tensor(anchor_size, device=device).view(1, 1, 1, 1, 3)
        anchor_size_tensor = anchor_size_tensor.repeat(mesh_x.shape[0], mesh_x.shape[1], 1, rotations.shape[0], 1)

        anchors = torch.cat([mesh_x, mesh_y, mesh_z, anchor_size_tensor, mesh_rot], dim=-1)
        anchors = anchors.permute(1, 0, 3, 2, 4).contiguous()  # (H, W, num_rot, 1, 7)
        anchors = anchors.squeeze(3)  # remove the z dim singleton
        return anchors  # (H, W, num_rot, 7)


    def get_multi_anchors(self, feature_map_size):
        '''
        feature_map_size: (y_l, x_l)
        ranges: [[x1, y1, z1, x2, y2, z2], [x1, y1, z1, x2, y2, z2], [x1, y1, z1, x2, y2, z2]]
        sizes: [[w, l, h], [w, l, h], [w, l, h]]
        rotations: [0, 1.57]
        return: shape=(y_l, x_l, 3, 2, 7)
        '''
        device = feature_map_size.device
        ranges = torch.tensor(self.ranges, device=device)  
        sizes = torch.tensor(self.sizes, device=device) 
        rotations = torch.tensor(self.rotations, device=device)
        multi_anchors = []
        for i in range(len(ranges)):
            anchors = self.get_anchors(feature_map_size=feature_map_size, 
                                       anchor_range=ranges[i], 
                                       anchor_size=sizes[i], 
                                       rotations=rotations)
            multi_anchors.append(anchors[:, :, None, :, :])
        multi_anchors = torch.cat(multi_anchors, dim=2)

        return multi_anchors


def anchors2bboxes(anchors, deltas):
    '''
    anchors: (M, 7),  (x, y, z, w, l, h, theta)
    deltas: (M, 7)
    return: (M, 7)
    '''
    da = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
    x = deltas[:, 0] * da + anchors[:, 0]
    y = deltas[:, 1] * da + anchors[:, 1]
    z = deltas[:, 2] * anchors[:, 5] + anchors[:, 2] + anchors[:, 5] / 2

    w = anchors[:, 3] * torch.exp(deltas[:, 3])
    l = anchors[:, 4] * torch.exp(deltas[:, 4])
    h = anchors[:, 5] * torch.exp(deltas[:, 5])

    z = z - h / 2

    theta = anchors[:, 6] + deltas[:, 6]
    
    bboxes = torch.stack([x, y, z, w, l, h, theta], dim=1)
    return bboxes


def bboxes2deltas(bboxes, anchors):
    '''
    bboxes: (M, 7), (x, y, z, w, l, h, theta)
    anchors: (M, 7)
    return: (M, 7)
    '''
    da = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

    dx = (bboxes[:, 0] - anchors[:, 0]) / da
    dy = (bboxes[:, 1] - anchors[:, 1]) / da

    zb = bboxes[:, 2] + bboxes[:, 5] / 2  # bottom center
    za = anchors[:, 2] + anchors[:, 5] / 2 # bottom center
    dz = (zb - za) / anchors[:, 5] # bottom center

    dw = torch.log(bboxes[:, 3] / anchors[:, 3])
    dl = torch.log(bboxes[:, 4] / anchors[:, 4])
    dh = torch.log(bboxes[:, 5] / anchors[:, 5])
    dtheta = bboxes[:, 6] - anchors[:, 6]

    deltas = torch.stack([dx, dy, dz, dw, dl, dh, dtheta], dim=1)
    return deltas


def anchor_target(batched_anchors, batched_gt_bboxes, batched_gt_labels, assigners, nclasses):
    '''
    batched_anchors: [(y_l, x_l, 3, 2, 7), (y_l, x_l, 3, 2, 7), ... ]
    batched_gt_bboxes: [(n1, 7), (n2, 7), ...]
    batched_gt_labels: [(n1, ), (n2, ), ...]
    return: 
           dict = {batched_anchors_labels: (bs, n_anchors),
                   batched_labels_weights: (bs, n_anchors),
                   batched_anchors_reg: (bs, n_anchors, 7),
                   batched_reg_weights: (bs, n_anchors),
                   batched_anchors_dir: (bs, n_anchors),
                   batched_dir_weights: (bs, n_anchors)}
    '''
    assert len(batched_anchors) == len(batched_gt_bboxes) == len(batched_gt_labels)
    #print(f"batched_anchors = {batched_anchors[0].shape}")
    batch_size = len(batched_anchors)
    n_assigners = len(assigners)
    batched_labels, batched_label_weights = [], []
    batched_bbox_reg, batched_bbox_reg_weights = [], []
    batched_dir_labels, batched_dir_labels_weights = [], []
    for i in range(batch_size):
        anchors = batched_anchors[i]
        gt_bboxes, gt_labels = batched_gt_bboxes[i], batched_gt_labels[i]
        # what we want to get next ?
        # 1. identify positive anchors and negative anchors  -> cls
        # 2. identify the regresstion values  -> reg
        # 3. indentify the direction  -> dir_cls
        multi_labels, multi_label_weights = [], []
        multi_bbox_reg, multi_bbox_reg_weights = [], []
        multi_dir_labels, multi_dir_labels_weights = [], []
        d1, d2, d3, d4, d5 = anchors.size()
        #print(f"d1, d2, d3, d4, d5 = {d1}, {d2}, {d3}, {d4}, {d5}")
        # Ground truth example
        #print("GT bboxes sample:")
        #print(gt_bboxes[:5])

        # Anchors sample
        #print("Anchor sample:")
        #print(batched_anchors[0].reshape(-1, 7)[:5])
        for j in range(n_assigners): # multi anchors
            assigner = assigners[j]
            pos_iou_thr, neg_iou_thr, min_iou_thr = \
            assigner['pos_iou_thr'], assigner['neg_iou_thr'], assigner['min_iou_thr']
            cur_anchors = anchors[:, :, j, :, :].reshape(-1, 7)
            overlaps = iou2d_nearest(gt_bboxes, cur_anchors) 
            max_overlaps, max_overlaps_idx = torch.max(overlaps, dim=0)
            #print(f"Max IoUs per anchor: {max_overlaps}")
            #print(f"Max IoU overall: {max_overlaps.max().item()}")
            gt_max_overlaps, _ = torch.max(overlaps, dim=1)

            assigned_gt_inds = -torch.ones_like(cur_anchors[:, 0], dtype=torch.long)
            # a. negative anchors
            assigned_gt_inds[max_overlaps < neg_iou_thr] = 0

            # b. positive anchors
            # rule 1
            #print(f"assigned_gt_inds = {assigned_gt_inds}")
            #print(f"Unique values in assigned_gt_inds: {torch.unique(assigned_gt_inds)}")
            pos_mask = max_overlaps >= pos_iou_thr
            #print(f"Number of positives by mask: {pos_mask.sum().item()}")
            #print(f"max_overlaps[pos_mask]: {max_overlaps[pos_mask]}")
            #print(f"indices being assigned: {max_overlaps_idx[pos_mask] + 1}")
            assigned_gt_inds[max_overlaps >= pos_iou_thr] = max_overlaps_idx[max_overlaps >= pos_iou_thr] + 1
            #print(f"gt_labels = {gt_labels}")
            #print(f"assigned_gt_inds_after = {assigned_gt_inds}")
            # rule 2
            # support one bbox to multi anchors, only if the anchors are with the highest iou.
            # rule2 may modify the labels generated by rule 1
            #print("Assigned inds BEFORE Rule 2:", assigned_gt_inds.unique())
            for i in range(len(gt_bboxes)):
                if gt_max_overlaps[i] >= min_iou_thr:
                    assigned_gt_inds[overlaps[i] == gt_max_overlaps[i]] = i + 1
            #print("Assigned inds AFTER Rule 2:", assigned_gt_inds.unique())

            pos_flag = assigned_gt_inds > 0
            neg_flag = assigned_gt_inds == 0
            # 1. anchor labels
            #print(f"gt_labels = {gt_labels}")
            #print("=== Ground Truth Labels (gt_labels) ===")
            #for i in range(nclasses):
            #    count = (gt_labels == i).sum().item()
            #    print(f"Class {i}: {count} samples")
            assigned_gt_labels = torch.zeros_like(cur_anchors[:, 0], dtype=torch.long) + nclasses # -1 is not optimal, for some bboxes are with labels -1
            assigned_gt_labels[pos_flag] = gt_labels[assigned_gt_inds[pos_flag] - 1].long()
            #print("Assigned labels (pos only):", assigned_gt_labels[pos_flag])
            #print(f"assigned_gt_labels = {assigned_gt_labels}")
            #print("=== Assigned Labels (assigned_gt_labels) ===")
            #valid_mask = (assigned_gt_labels >= 0) & (assigned_gt_labels < nclasses)

            #for i in range(nclasses):
            #    count = (assigned_gt_labels[valid_mask] == i).sum().item()
            #    print(f"Class {i}: {count} samples")
            assigned_gt_labels_weights = torch.zeros_like(cur_anchors[:, 0])
            assigned_gt_labels_weights[pos_flag] = 1
            assigned_gt_labels_weights[neg_flag] = 1

            # 2. anchor regression
            assigned_gt_reg_weights = torch.zeros_like(cur_anchors[:, 0])
            assigned_gt_reg_weights[pos_flag] = 1

            assigned_gt_reg = torch.zeros_like(cur_anchors)
            positive_anchors = cur_anchors[pos_flag]
            corr_gt_bboxes = gt_bboxes[assigned_gt_inds[pos_flag] - 1]
            assigned_gt_reg[pos_flag] = bboxes2deltas(corr_gt_bboxes, positive_anchors)

            # 3. anchor direction
            assigned_gt_dir_weights = torch.zeros_like(cur_anchors[:, 0])
            assigned_gt_dir_weights[pos_flag] = 1

            assigned_gt_dir = torch.zeros_like(cur_anchors[:, 0], dtype=torch.long)
            dir_cls_targets = limit_period(corr_gt_bboxes[:, 6].cpu(), 0, 2 * np.pi).to(corr_gt_bboxes)
            dir_cls_targets = torch.floor(dir_cls_targets / np.pi).long()
            assigned_gt_dir[pos_flag] = torch.clamp(dir_cls_targets, min=0, max=1)

            multi_labels.append(assigned_gt_labels.reshape(d1, d2, 1, d4))
            multi_label_weights.append(assigned_gt_labels_weights.reshape(d1, d2, 1, d4))
            multi_bbox_reg.append(assigned_gt_reg.reshape(d1, d2, 1, d4, -1))
            multi_bbox_reg_weights.append(assigned_gt_reg_weights.reshape(d1, d2, 1, d4))
            multi_dir_labels.append(assigned_gt_dir.reshape(d1, d2, 1, d4))
            multi_dir_labels_weights.append(assigned_gt_dir_weights.reshape(d1, d2, 1, d4))
        
        multi_labels = torch.cat(multi_labels, dim=-2).reshape(-1)
        multi_label_weights = torch.cat(multi_label_weights, dim=-2).reshape(-1)
        multi_bbox_reg = torch.cat(multi_bbox_reg, dim=-3).reshape(-1, d5)
        multi_bbox_reg_weights = torch.cat(multi_bbox_reg_weights, dim=-2).reshape(-1)
        multi_dir_labels = torch.cat(multi_dir_labels, dim=-2).reshape(-1)
        multi_dir_labels_weights = torch.cat(multi_dir_labels_weights, dim=-2).reshape(-1)

        #print("Unique values in multi_labels before stacking:", multi_labels.unique())
        batched_labels.append(multi_labels)
        batched_label_weights.append(multi_label_weights)
        batched_bbox_reg.append(multi_bbox_reg)
        batched_bbox_reg_weights.append(multi_bbox_reg_weights)
        batched_dir_labels.append(multi_dir_labels) 
        batched_dir_labels_weights.append(multi_dir_labels_weights)
        #print(f"batched_labels = {batched_labels[0].shape}")
    
    rt_dict = dict(
        batched_labels=torch.stack(batched_labels, 0), # (bs, y_l * x_l * 3 * 2)
        batched_label_weights=torch.stack(batched_label_weights, 0), # (bs, y_l * x_l * 3 * 2)
        batched_bbox_reg=torch.stack(batched_bbox_reg, 0), # (bs, y_l * x_l * 3 * 2, 7)
        batched_bbox_reg_weights=torch.stack(batched_bbox_reg_weights, 0), # (bs, y_l * x_l * 3 * 2)
        batched_dir_labels=torch.stack(batched_dir_labels, 0), # (bs, y_l * x_l * 3 * 2)
        batched_dir_labels_weights=torch.stack(batched_dir_labels_weights, 0) # (bs, y_l * x_l * 3 * 2)
    )
    unique_values = torch.unique(torch.stack(batched_labels, 0))
    #print(f"Unique values in batched_labels: {unique_values}")
    return rt_dict
            