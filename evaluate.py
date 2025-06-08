import argparse
import numpy as np
import os
import torch
import pdb
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from vod.frame import KittiLocations, FrameDataLoader, FrameTransformMatrix, homogeneous_transformation

from pointpillars.utils import setup_seed, keep_bbox_from_image_range, \
    keep_bbox_from_lidar_range, write_pickle, write_label, \
    iou2d, iou3d_camera, iou_bev
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillars

def interpolate_precision_recall(precisions, recalls, num_points=100):
    # Ensure recalls are sorted ascending for interpolation
    sorted_indices = np.argsort(recalls)
    recalls_sorted = np.array(recalls)[sorted_indices]
    precisions_sorted = np.array(precisions)[sorted_indices]

    # Create uniform recall points for interpolation
    recall_interp = np.linspace(0, 1, num_points)

    # Interpolate precision values at these recall points
    precision_interp = np.interp(recall_interp, recalls_sorted, precisions_sorted, left=0, right=0)

    return recall_interp, precision_interp

def max_f1_from_precision_recall(precisions, recalls):
    recall_interp, precision_interp = interpolate_precision_recall(precisions, recalls)

    # Avoid division by zero
    denom = precision_interp + recall_interp
    denom[denom == 0] = 1e-6

    f1_scores = 2 * (precision_interp * recall_interp) / denom

    max_f1 = np.max(f1_scores)
    best_idx = np.argmax(f1_scores)
    best_recall = recall_interp[best_idx]
    best_precision = precision_interp[best_idx]

    return max_f1

def compute_weighted_class_ap(AP_3_easy, AP_3_medium, AP_3_hard,
                              counts_easy, counts_medium, counts_hard):
    """
    Compute weighted AP for each class given APs and counts per difficulty.

    Args:
        AP_3_easy, AP_3_medium, AP_3_hard: dict[class → AP]
        counts_easy, counts_medium, counts_hard: dict[class → GT count]

    Returns:
        dict[class → weighted AP]
    """
    weighted_ap_per_class = {}
    all_classes = set(AP_3_easy) | set(AP_3_medium) | set(AP_3_hard)

    for cls in all_classes:
        ap_easy = AP_3_easy.get(cls, 0.0)
        ap_medium = AP_3_medium.get(cls, 0.0)
        ap_hard = AP_3_hard.get(cls, 0.0)

        n_easy = counts_easy.get(cls, 0)
        n_medium = counts_medium.get(cls, 0)
        n_hard = counts_hard.get(cls, 0)

        total = n_easy + n_medium + n_hard
        if total == 0:
            weighted_ap = 0.0
        else:
            weighted_ap = (ap_easy * n_easy + ap_medium * n_medium + ap_hard * n_hard) / total

        weighted_ap_per_class[cls] = weighted_ap

    return weighted_ap_per_class

def compute_ap_trapezoidal(precisions, recalls):
    """
    Compute Average Precision (AP) using the trapezoidal rule.

    Args:
        precisions (list or np.array): precision values at various thresholds
        recalls (list or np.array): corresponding recall values at the same thresholds

    Returns:
        ap (float): average precision as area under PR curve
    """
    # Ensure input is numpy array
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # Sort by recall ascending
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    # Optional: ensure the curve starts at (0,1) and ends at (1,0)
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([precisions[0]], precisions, [0.0]))

    # Smooth precision to be non-increasing (optional but COCO-style)
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])

    # Trapezoidal integration
    ap = 0.0
    for i in range(1, len(recalls)):
        delta_r = recalls[i] - recalls[i - 1]
        ap += delta_r * precisions[i]

    return ap

def get_score_thresholds(tp_scores, total_num_valid_gt, num_sample_pts=41):
    score_thresholds = []
    tp_scores = sorted(tp_scores)[::-1]
    cur_recall, pts_ind = 0, 0
    for i, score in enumerate(tp_scores):
        lrecall = (i + 1) / total_num_valid_gt
        rrecall = (i + 2) / total_num_valid_gt

        if i == len(tp_scores) - 1:
            score_thresholds.append(score)
            break

        if (lrecall + rrecall) / 2 < cur_recall:
            continue

        score_thresholds.append(score)
        pts_ind += 1
        cur_recall = pts_ind / (num_sample_pts - 1)
    return score_thresholds

def radar2camera(parts,transforms):

    #class_name = parts[0]
    w, l, h = map(float, parts[3:6])  # dims in radar frame: w, l, h
    x_r, y_r, z_r = map(float, parts[:3])  # position in radar frame
    yaw_r = float(parts[6])  # yaw in radar frame

    # Convert center from radar to camera coords
    center_radar = np.array([[x_r, y_r, z_r, 1]])
    center_camera = homogeneous_transformation(center_radar, transforms.t_camera_radar)[:, :3].flatten()

    # Adjust center y to bottom-centered (KITTI format)
    y_bottom_centered = center_camera[1] + (h / 2)  # Add half height to go from center to bottom

    # Transform yaw vector from radar to camera frame
    R_radar_to_cam = transforms.t_camera_radar[:3, :3]
    yaw_vec_radar = np.array([np.cos(yaw_r), 0, np.sin(yaw_r)])
    yaw_vec_camera = R_radar_to_cam @ yaw_vec_radar
    yaw_camera = np.arctan2(yaw_vec_camera[2], yaw_vec_camera[0])  # Notice: KITTI uses x-z plane for yaw

    # KITTI dims are h,w,l
    dims_camera = [h, w, l]

    # Rebuild line with updated dims, location and yaw at correct indices:
    new_parts = parts[:]

    new_parts[3] = f"{dims_camera[0]:.2f}"  # h
    new_parts[4] = f"{dims_camera[1]:.2f}"  # w
    new_parts[5] = f"{dims_camera[2]:.2f}"  # l

    new_parts[0] = f"{center_camera[0]:.2f}"
    new_parts[1] = f"{y_bottom_centered:.2f}"
    new_parts[2] = f"{center_camera[2]:.2f}"

    new_parts[6] = f"{yaw_camera:.2f}"
    return new_parts


def do_eval(det_results, gt_results, CLASSES, saved_path):
    '''
    det_results: list,
    gt_results: dict(id -> det_results)
    CLASSES: dict
    '''
    # Store score distributions
    score_distributions = defaultdict(list)





    assert len(det_results) == len(gt_results)
    f = open(os.path.join(saved_path, 'eval_results.txt'), 'w')

    max_f1_scores_per_class_2d_easy = {}
    max_f1_scores_per_class_2d_medium = {}
    max_f1_scores_per_class_2d_hard = {}
    max_f1_scores_per_class_3d_easy = {}
    max_f1_scores_per_class_3d_medium = {}
    max_f1_scores_per_class_3d_hard = {}

    score_for_max_f1_2d_easy = {}
    score_for_max_f1_2d_medium = {}
    score_for_max_f1_2d_hard = {}
    score_for_max_f1_3d_easy = {}
    score_for_max_f1_3d_medium = {}
    score_for_max_f1_3d_hard = {}

    AP_2_easy = {}
    AP_2_medium = {}
    AP_2_hard = {}
    AP_3_easy = {}
    AP_3_medium = {}
    AP_3_hard = {}

    counts_easy = {}
    counts_medium = {}
    counts_hard = {}


    # 1. calculate iou
    ious = {
        'bbox_2d': [],
        'bbox_bev': [],
        'bbox_3d': []
    }
    ids = list(sorted(gt_results.keys()))
    for id in ids:

        gt_result = gt_results[id]['annos']
        det_result = det_results[id]
        #print(f"Evaluating frame {id} with {len(gt_result['bbox'])} GT boxes and {len(det_result['bbox'])} DET boxes")
        #print("gt_result = ", gt_result)
        #print("det_result = ", det_result)
        
        
        




        # Collect detection scores per class
        for cls_name, score in zip(det_result['name'], det_result['score']):
            score_distributions[cls_name].append(score)


        # 1.1, 2d bboxes iou
        gt_bboxes2d = gt_result['bbox'].astype(np.float32)
        det_bboxes2d = det_result['bbox'].astype(np.float32)
        #print(id)
        #print(f"GT = {gt_bboxes2d}")
        #print(f"DET = {det_bboxes2d}")
        if gt_bboxes2d.shape[0] == 0 or det_bboxes2d.shape[0] == 0:
            ious['bbox_2d'].append(np.zeros((gt_bboxes2d.shape[0], det_bboxes2d.shape[0])))
        else:
            iou2d_v = iou2d(torch.from_numpy(gt_bboxes2d).cuda(), torch.from_numpy(det_bboxes2d).cuda())
            ious['bbox_2d'].append(iou2d_v.cpu().numpy())
        #print(f"IOU = {iou2d_v}")
        #iou2d_v = iou2d(torch.from_numpy(gt_bboxes2d).cuda(), torch.from_numpy(det_bboxes2d).cuda())
        #ious['bbox_2d'].append(iou2d_v.cpu().numpy())

        # 1.2, bev iou
        gt_location = gt_result['location'].astype(np.float32)
        h= gt_result['dimensions'][:, 0].astype(np.float32)  # height
        gt_location[:,1] = gt_location[:,1]+(h/2)
        gt_dimensions = gt_result['dimensions'].astype(np.float32)
        #gt_dimensions = gt_dimensions[:, [1, 2, 0]]  # [h, w, l] → [w, l, h]
        gt_rotation_y = gt_result['rotation_y'].astype(np.float32)
        det_location = det_result['location'].astype(np.float32)
        det_dimensions = det_result['dimensions'].astype(np.float32)
        det_rotation_y = det_result['rotation_y'].astype(np.float32)


        if gt_location.shape[0] == 0 or det_location.shape[0] == 0:
            ious['bbox_bev'].append(np.zeros((gt_location.shape[0], det_location.shape[0])))
        else:
            gt_bev = np.concatenate([gt_location[:, [1, 0]], gt_dimensions[:, [1, 0]], gt_rotation_y[:, None]], axis=-1)
            det_bev = np.concatenate([det_location[:, [1, 0]], det_dimensions[:, [1, 0]], det_rotation_y[:, None]], axis=-1)
            iou_bev_v = iou_bev(torch.from_numpy(gt_bev).cuda(), torch.from_numpy(det_bev).cuda())
            ious['bbox_bev'].append(iou_bev_v.cpu().numpy())

        #gt_bev = np.concatenate([gt_location[:, [0, 2]], gt_dimensions[:, [0, 2]], gt_rotation_y[:, None]], axis=-1)
        #det_bev = np.concatenate([det_location[:, [0, 2]], det_dimensions[:, [0, 2]], det_rotation_y[:, None]], axis=-1)
        #iou_bev_v = iou_bev(torch.from_numpy(gt_bev).cuda(), torch.from_numpy(det_bev).cuda())
        #ious['bbox_bev'].append(iou_bev_v.cpu().numpy())
       
        # 1.3, 3dbboxes iou
        if gt_location.shape[0] == 0 or det_location.shape[0] == 0:
            ious['bbox_3d'].append(np.zeros((gt_location.shape[0], det_location.shape[0])))
        else:
            gt_bboxes3d = np.concatenate([gt_location, gt_dimensions, gt_rotation_y[:, None]], axis=-1)
            det_bboxes3d = np.concatenate([det_location, det_dimensions, det_rotation_y[:, None]], axis=-1)
            #print(f"gt_bboxes3d = {gt_bboxes3d}")
            #print(f"det_bboxes3d = {det_bboxes3d}")
            #print(f"[Frame {id}] 3D GT boxes:")
            #for i, box in enumerate(gt_bboxes3d):
                #print(f"  GT-{i}: {box}")

            #print(f"[Frame {id}] 3D DET boxes:")
            #for i, box in enumerate(det_bboxes3d):
            #    print(f"  DET-{i}: {box}, Score={det_result['score'][i]}")

            #print(f"[Frame {id}] IOU 3D Matrix:\n{iou3d_v.cpu().numpy()}")

            iou3d_v = iou3d_camera(torch.from_numpy(gt_bboxes3d).cuda(), torch.from_numpy(det_bboxes3d).cuda())
            ious['bbox_3d'].append(iou3d_v.cpu().numpy())

    MIN_IOUS = {
        'Pedestrian': [0.25, 0.25, 0.25],
        'bicycle': [0.25, 0.25, 0.25],
        'rider': [0.25, 0.25, 0.25],
        'Cyclist': [0.25, 0.25, 0.25],
        'moped_scooter': [0.25, 0.25, 0.25],
        'motor': [0.25, 0.25, 0.25],
        'ride_other': [0.25, 0.25, 0.25],
        'bicycle_rack': [0.25, 0.25, 0.25],
        'Car': [0.5, 0.5, 0.5]
    }
    MIN_HEIGHT = [40, 25, 25]
    #MIN_HEIGHT = [20, 10, 10]

    overall_results = {}
    for e_ind, eval_type in enumerate(['bbox_2d', 'bbox_bev', 'bbox_3d']):
        eval_ious = ious[eval_type]
        eval_ap_results, eval_aos_results = {}, {}
        for cls in CLASSES:
            print(f'=========={cls.upper()}==========') 
            eval_ap_results[cls] = []
            eval_aos_results[cls] = []
            CLS_MIN_IOU = MIN_IOUS[cls][e_ind]
            for difficulty in [0, 1, 2]:
                print(f'=========={difficulty}==========')
                # 1. bbox property
                total_gt_ignores, total_det_ignores, total_dc_bboxes, total_scores = [], [], [], []
                total_gt_alpha, total_det_alpha = [], []
                for id in ids:
                    gt_result = gt_results[id]['annos']
                    det_result = det_results[id]

                    # 1.1 gt bbox property
                    cur_gt_names = gt_result['name']
                    cur_difficulty = gt_result['difficulty']
                    gt_ignores, dc_bboxes = [], []

                    ignore_difficulty = 0
                    ignore_similar_class = 0
                    ignore_irrelevant = 0
                    total_valid_gt = 0

                    for j, cur_gt_name in enumerate(cur_gt_names):
                        ignore = cur_difficulty[j] < 0 or cur_difficulty[j] > difficulty
                        if cur_gt_name == cls:
                            valid_class = 1
                        elif cls == 'Pedestrian' and cur_gt_name == 'Person_sitting':
                            valid_class = 0
                        elif cls == 'Car' and cur_gt_name == 'Van':
                            valid_class = 0
                        else:
                            valid_class = -1
                       
                        if valid_class == 1 and not ignore:
                            gt_ignores.append(0)
                            total_valid_gt += 1
                        elif valid_class == 0 or (valid_class == 1 and ignore):
                            gt_ignores.append(1)
                            if valid_class == 0:
                                ignore_similar_class += 1
                            else:
                                ignore_difficulty += 1
                        else:
                            gt_ignores.append(-1)
                            ignore_irrelevant += 1

                        if cur_gt_name == 'DontCare':
                            dc_bboxes.append(gt_result['bbox'][j])
                    total_gt_ignores.append(gt_ignores)
                    total_dc_bboxes.append(np.array(dc_bboxes))
                    total_gt_alpha.append(gt_result['alpha'])

                    # 1.2 det bbox property
                    cur_det_names = det_result['name']
                    if det_result['bbox'].ndim == 1:
                        det_result['bbox'] = det_result['bbox'].reshape(0, 4)
                    cur_det_heights = det_result['bbox'][:, 3] - det_result['bbox'][:, 1]
                    det_ignores = []
                    min_height_ignored_count = 0
                    for j, cur_det_name in enumerate(cur_det_names):
                        if cur_det_heights[j] < MIN_HEIGHT[difficulty]:
                            det_ignores.append(1)
                            min_height_ignored_count += 1
                        elif cur_det_name == cls:
                            det_ignores.append(0)
                        else:
                            det_ignores.append(-1)
                    total_det_ignores.append(det_ignores)
                    total_scores.append(det_result['score'])
                    total_det_alpha.append(det_result['alpha'])

                # 2. calculate scores thresholds for PR curve
                tp_scores = []
                for i, id in enumerate(ids):
                    cur_eval_ious = eval_ious[i]
                    gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                    scores = total_scores[i]

                    nn, mm = cur_eval_ious.shape
                    assigned = np.zeros((mm, ), dtype=np.bool_)
                    for j in range(nn):
                        if gt_ignores[j] == -1:
                            continue
                        match_id, match_score = -1, -1
                        for k in range(mm):
                            if not assigned[k] and det_ignores[k] >= 0 and cur_eval_ious[j, k] > CLS_MIN_IOU and scores[k] > match_score:
                                match_id = k
                                match_score = scores[k]
                        if match_id != -1:
                            assigned[match_id] = True
                            if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                                tp_scores.append(match_score)
                total_num_valid_gt = np.sum([np.sum(np.array(gt_ignores) == 0) for gt_ignores in total_gt_ignores])
                total_num_detections = np.sum([np.sum(np.array(detections) == 0) for detections in total_det_ignores])






                                # DEBUG HOOK START
                #print(f"\n--- DEBUGGING CLASS {cls} @ difficulty={difficulty} ---")
                #print(f"Total GTs (valid): {total_num_valid_gt}")
                #print(f"Total detections (valid): {total_num_detections}")
                #print(f"TP scores collected: {len(tp_scores)}")

                if len(tp_scores) == 0:
                    #print("No true positives matched. Checking why...")
                    for i, id in enumerate(ids):
                        cur_eval_ious = eval_ious[i]
                        gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                        scores = total_scores[i]
                        nn, mm = cur_eval_ious.shape

                        #for j in range(nn):
                        #    for k in range(mm):
                        #        if gt_ignores[j] == 0 and det_ignores[k] == 0:
                        #            print(f"[Frame {id}] GT-{j} and DET-{k}: IOU={cur_eval_ious[j,k]:.3f}, Score={scores[k]:.3f}")

                        #if nn == 0:
                         #   print(f"[Frame {id}] No GT boxes.")
                        #if mm == 0:
                        #    print(f"[Frame {id}] No detections.")
                # DEBUG HOOK END








                score_thresholds = get_score_thresholds(tp_scores, total_num_valid_gt)    
                #print(f"score_thresholds = {score_thresholds}")
                #print(f"total_num_valid_gt = {total_num_valid_gt}")
                #print(f"total_num_detections = {total_num_detections}")
                # 3. draw PR curve and calculate mAP
                tps, fns, fps, total_aos = [], [], [], []

                for score_threshold in score_thresholds:
                    tp, fn, fp = 0, 0, 0
                    aos = 0
                    for i, id in enumerate(ids):
                        cur_eval_ious = eval_ious[i]
                        gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                        gt_alpha, det_alpha = total_gt_alpha[i], total_det_alpha[i]
                        scores = total_scores[i]

                        nn, mm = cur_eval_ious.shape
                        assigned = np.zeros((mm, ), dtype=np.bool_)
                        for j in range(nn):
                            if gt_ignores[j] == -1:
                                continue
                            match_id, match_iou = -1, -1
                            for k in range(mm):
                                if not assigned[k] and det_ignores[k] >= 0 and scores[k] >= score_threshold and cur_eval_ious[j, k] > CLS_MIN_IOU:
    
                                    if det_ignores[k] == 0 and cur_eval_ious[j, k] > match_iou:
                                        match_iou = cur_eval_ious[j, k]
                                        match_id = k
                                    elif det_ignores[k] == 1 and match_iou == -1:
                                        match_id = k

                            if match_id != -1:
                                assigned[match_id] = True
                                if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                                    tp += 1
                                    if eval_type == 'bbox_2d':
                                        aos += (1 + np.cos(gt_alpha[j] - det_alpha[match_id])) / 2
                            else:
                                if gt_ignores[j] == 0:
                                    fn += 1
                            
                        for k in range(mm):
                            if det_ignores[k] == 0 and scores[k] >= score_threshold and not assigned[k]:
                                fp += 1
                        
                        # In case 2d bbox evaluation, we should consider dontcare bboxes
                        if eval_type == 'bbox_2d':
                            dc_bboxes = total_dc_bboxes[i]
                            det_bboxes = det_results[id]['bbox']
                            if len(dc_bboxes) > 0:
                                ious_dc_det = iou2d(torch.from_numpy(det_bboxes), torch.from_numpy(dc_bboxes), metric=1).numpy().T
                                for j in range(len(dc_bboxes)):
                                    for k in range(len(det_bboxes)):
                                        if det_ignores[k] == 0 and scores[k] >= score_threshold and not assigned[k]:
                                            if ious_dc_det[j, k] > CLS_MIN_IOU:
                                                fp -= 1
                                                assigned[k] = True
                            
                    tps.append(tp)
                    fns.append(fn)
                    fps.append(fp)
                    if eval_type == 'bbox_2d':
                        total_aos.append(aos)

                tps, fns, fps = np.array(tps), np.array(fns), np.array(fps)

                print(f"tps = {tps}")
                print(f"fns = {fns}")
                print(f"fps = {fps}")
                #print(f"Ignored GTs due to difficulty: {ignore_difficulty}")
                #print(f"Ignored GTs due to similar class: {ignore_similar_class}")
                #print(f"Ignored GTs due to irrelevant class: {ignore_irrelevant}")
                print(f"score_thresholds = {score_thresholds}")
                recalls = tps / (tps + fns)
                print(f"recalls = {recalls}")
                precisions = tps / (tps + fps)
                
                #f1_scores = (2*precisions*recalls) / (precisions + recalls)
               

                #print(f'max f1_scores = {np.max(f1_scores)}')
                if eval_type == 'bbox_3d':  # Store only for bbox_3d
                    if len(score_thresholds) > 0:
                        max_f1=max_f1_from_precision_recall(precisions, recalls)
                        if difficulty == 0:
                            #best_idx = np.argmax(f1_scores)
                            max_f1_scores_per_class_3d_easy[cls] = max_f1
                            #score_for_max_f1_3d_easy[cls] = score_thresholds[best_idx]
                            AP_3_easy[cls]=compute_ap_trapezoidal(precisions, recalls)
                            counts_easy[cls]= tp+fn
                        elif difficulty == 1:   
                            #best_idx = np.argmax(f1_scores)
                            max_f1_scores_per_class_3d_medium[cls] = max_f1
                            #score_for_max_f1_3d_medium[cls] = score_thresholds[best_idx]
                            AP_3_medium[cls]=compute_ap_trapezoidal(precisions, recalls)
                            counts_medium[cls]= tp+fn
                        elif difficulty == 2:
                            #best_idx = np.argmax(f1_scores)
                            max_f1_scores_per_class_3d_hard[cls] = max_f1
                            #score_for_max_f1_3d_hard[cls] = score_thresholds[best_idx]
                            AP_3_hard[cls]=compute_ap_trapezoidal(precisions, recalls)
                            counts_hard[cls]= tp+fn
                    else:
                        if difficulty == 0:
                            max_f1_scores_per_class_3d_easy[cls] = 0.0
                            score_for_max_f1_3d_easy[cls] = 0.0
                            AP_3_easy[cls]=0.0
                            counts_easy[cls]=0
                        elif difficulty == 1:
                            max_f1_scores_per_class_3d_medium[cls] = 0.0
                            score_for_max_f1_3d_medium[cls] = 0.0
                            AP_3_medium[cls]=0.0
                            counts_medium[cls]=0
                        elif difficulty == 2:
                            max_f1_scores_per_class_3d_hard[cls] = 0.0
                            score_for_max_f1_3d_hard[cls] = 0.0
                            AP_3_hard[cls]=0.0
                            counts_hard[cls]=0

                if eval_type == 'bbox_2d':  # Store only for bbox_2d
                    if len(score_thresholds)> 0:
                        max_f1=max_f1_from_precision_recall(precisions, recalls)
                        if difficulty == 0:
                            #best_idx = np.argmax(f1_scores)
                            max_f1_scores_per_class_2d_easy[cls] = max_f1
                            #score_for_max_f1_2d_easy[cls] = score_thresholds[best_idx]
                            AP_2_easy[cls]=compute_ap_trapezoidal(precisions, recalls)
                        elif difficulty == 1:   
                            #best_idx = np.argmax(f1_scores)
                            max_f1_scores_per_class_2d_medium[cls] = max_f1
                           # score_for_max_f1_2d_medium[cls] = score_thresholds[best_idx]
                            AP_2_medium[cls]=compute_ap_trapezoidal(precisions, recalls)
                        elif difficulty == 2:
                            #best_idx = np.argmax(f1_scores)
                            max_f1_scores_per_class_2d_hard[cls] = max_f1
                            #score_for_max_f1_2d_hard[cls] = score_thresholds[best_idx]
                            AP_2_hard[cls]=compute_ap_trapezoidal(precisions, recalls)
                    else:
                        if difficulty == 0:
                            max_f1_scores_per_class_2d_easy[cls] = 0.0
                            score_for_max_f1_2d_easy[cls]=  0.0
                        elif difficulty == 1:
                            max_f1_scores_per_class_2d_medium[cls] = 0.0
                            score_for_max_f1_2d_medium[cls] = 0.0
                        elif difficulty == 2:
                            max_f1_scores_per_class_2d_hard[cls] = 0.0
                            score_for_max_f1_2d_hard[cls] = 0.0

                #print(f"f1_scores = {f1_scores}")
                for i in range(len(score_thresholds)):
                    precisions[i] = np.max(precisions[i:])
                
                
                sums_AP = 0
                amount_samples=0
                length = len(score_thresholds)
                space=length//11
                if space%11 > 5:
                    space = space + 1
                if space == 0:
                    for i in range(0, len(score_thresholds), 1):
                        sums_AP += precisions[i]
                        amount_samples+=1
                    if amount_samples == 0: 
                        mAP = sums_AP / 11 * 100
                    else:
                        mAP = sums_AP / amount_samples * 100
                else:
                    for i in range(0, len(score_thresholds), space):
                        sums_AP += precisions[i]
                        amount_samples+=1
                    if amount_samples == 0: 
                        mAP = sums_AP / 11 * 100
                    else:
                        mAP = sums_AP / amount_samples * 100
                #mAP = sums_AP / 11 * 100
                eval_ap_results[cls].append(mAP)

                if eval_type == 'bbox_2d':
                    total_aos = np.array(total_aos)
                    similarity = total_aos / (tps + fps)
                    for i in range(len(score_thresholds)):
                        similarity[i] = np.max(similarity[i:])
                    sums_similarity = 0
                    for i in range(0, len(score_thresholds), 4):
                        sums_similarity += similarity[i]
                    mSimilarity = sums_similarity / 11 * 100
                    eval_aos_results[cls].append(mSimilarity)

        print(f'=========={eval_type.upper()}==========')
        print(f'=========={eval_type.upper()}==========', file=f)
        for k, v in eval_ap_results.items():
            print(f'{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
            print(f'{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)
        if eval_type == 'bbox_2d':
            print(f'==========AOS==========')
            print(f'==========AOS==========', file=f)
            for k, v in eval_aos_results.items():
                print(f'{k} AOS@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
                print(f'{k} AOS@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)
        
        overall_results[eval_type] = np.mean(list(eval_ap_results.values()), 0)
        if eval_type == 'bbox_2d':
            overall_results['AOS'] = np.mean(list(eval_aos_results.values()), 0)
    
    print(f'\n==========Overall==========')
    print(f'\n==========Overall==========', file=f)
    for k, v in overall_results.items():
        print(f'{k} AP: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
        print(f'{k} AP: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)

    

    
    weighted_f1=compute_weighted_class_ap(
    max_f1_scores_per_class_3d_easy, max_f1_scores_per_class_3d_medium, max_f1_scores_per_class_3d_hard,
    counts_easy, counts_medium, counts_hard
    )
    weighted_aps = compute_weighted_class_ap(
    AP_3_easy, AP_3_medium, AP_3_hard,
    counts_easy, counts_medium, counts_hard
    )
    print("\nWeighted F1 Scores for each class (bbox_3d):")
    for cls, f1 in weighted_f1.items():
        print(f"{cls}: {f1*100:.4f}")
    print("\nWeighted APs for each class (bbox_3d):")
    for cls, ap in weighted_aps.items():
        print(f"{cls}: {ap*100:.4f}")

    
    print("Saving score distribution histograms per class...")

    dist_dir = os.path.join(saved_path, "score_distributions")
    os.makedirs(dist_dir, exist_ok=True)

    for cls, scores in score_distributions.items():
        if len(scores) == 0:
            continue
        plt.figure()
        plt.hist(scores, bins=20, range=(0, 1), alpha=0.75, color='blue', edgecolor='black')
        plt.title(f'Score Distribution for {cls}')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(dist_dir, f'{cls}_score_distribution.png'))
        plt.close()

    f.close()
    

def main(args):
    
    val_dataset = Kitti(data_root=args.data_root,
                        split='val')
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)
    CLASSES = Kitti.CLASSES
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}

    if not args.no_cuda:
        model = PointPillars(nclasses=args.nclasses).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=args.nclasses)
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))
    
    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)
    saved_submit_path = os.path.join(saved_path, 'submit')
    os.makedirs(saved_submit_path, exist_ok=True)

    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        format_results = {}
        print('Predicting and Formatting the results.')
        for i, data_dict in enumerate(tqdm(val_dataloader)):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            #print("batched_gt_bboxes  = ", batched_gt_bboxes)
            batched_labels = data_dict['batched_labels']
            batched_difficulty = data_dict['batched_difficulty']
            batch_results = model(batched_pts=batched_pts, 
                                  mode='val',
                                  batched_gt_bboxes=batched_gt_bboxes, 
                                  batched_gt_labels=batched_labels)
            # pdb.set_trace()
            for j, result in enumerate(batch_results):
                format_result = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }
                
                calib_info = data_dict['batched_calib_info'][j]
                tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
                r0_rect = calib_info['R0_rect'].astype(np.float32)
                P2 = calib_info['P2'].astype(np.float32)
                image_shape = data_dict['batched_img_info'][j]['image_shape']
                idx = data_dict['batched_img_info'][j]['image_idx']
                #print(f"Before filtering: {len(result['lidar_bboxes'])}")
                #print(f"Frame {idx}: {len(result['lidar_bboxes'])} detections")


                result_filter = keep_bbox_from_image_range(result, tr_velo_to_cam, r0_rect, P2, image_shape,idx)
                #result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)

                #print(f"After image range filtering: {len(result_filter['lidar_bboxes'])}")

                lidar_bboxes = result_filter['lidar_bboxes']
                labels, scores = result_filter['labels'], result_filter['scores']
                bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']
                for lidar_bbox, label, score, bbox2d, camera_bbox in \
                    zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
                    #print(f"Radar Box Before Transform: {lidar_bbox}")
                    #print(f"Camera Box After Transform: {camera_bbox}")
                    format_result['name'].append(LABEL2CLASSES[label])
                    format_result['truncated'].append(0.0)
                    format_result['occluded'].append(0)
                    alpha = camera_bbox[6] - np.arctan2(camera_bbox[0], camera_bbox[2])
                    format_result['alpha'].append(alpha)
                    format_result['bbox'].append(bbox2d)
                    format_result['dimensions'].append(camera_bbox[3:6])
                    format_result['location'].append(camera_bbox[:3])
                    format_result['rotation_y'].append(camera_bbox[6])
                    format_result['score'].append(score)
                
                write_label(format_result, os.path.join(saved_submit_path, f'{idx:06d}.txt'))

                format_results[idx] = {k:np.array(v) for k, v in format_result.items()}
        
        write_pickle(format_results, os.path.join(saved_path, 'results.pkl'))
    
    print('Evaluating.. Please wait several seconds.')
    do_eval(format_results, val_dataset.data_infos, CLASSES, saved_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/home/student2/Documents/Datasets/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC/radar', 
                        help='your data root for kitti')
    parser.add_argument('--ckpt', default='pillar_logs/checkpoints_less_classes/epoch_50.pth', help='your checkpoint for kitti')
    parser.add_argument('--saved_path', default='results', help='your saved path for predicted results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=9)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()
    
    #pillar_logs/checkpoints/epoch_20.pth

    main(args)
