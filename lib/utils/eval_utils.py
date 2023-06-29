import numpy as np
from .data_utils import bbox_denormalize, cxcywh_to_x1y1x2y2
def compute_IOU(bbox_true, bbox_pred):
    '''
    compute IOU
    [x1, y1, x2, y2]
    '''
    xmin = np.maximum(bbox_true[:, 0], bbox_pred[:, 0])
    xmax = np.minimum(bbox_true[:, 2], bbox_pred[:, 2])
    ymin = np.maximum(bbox_true[:, 1], bbox_pred[:, 1])
    ymax = np.minimum(bbox_true[:, 3], bbox_pred[:, 3])
    w_true = bbox_true[:, 2] - bbox_true[:, 0]
    h_true = bbox_true[:, 3] - bbox_true[:, 1]
    w_pred = bbox_pred[:, 2] - bbox_pred[:, 0]
    h_pred = bbox_pred[:, 3] - bbox_pred[:, 1]

    w_inter = np.maximum(0, xmax - xmin)
    h_inter = np.maximum(0, ymax - ymin)
    intersection = w_inter * h_inter
    union = (w_true * h_true + w_pred * h_pred) - intersection

    return intersection/union

def eval_jaad_pie(input_traj_np, target_traj, all_dec_traj):
    input_traj_np = np.expand_dims(input_traj_np[:,-1,:], axis=1)
    target_traj = input_traj_np + target_traj
    all_dec_traj = input_traj_np + all_dec_traj
    all_dec_traj = bbox_denormalize(all_dec_traj, W=1920, H=1080)
    target_traj = bbox_denormalize(target_traj, W=1920, H=1080)

    all_dec_traj_xyxy = cxcywh_to_x1y1x2y2(all_dec_traj)
    target_traj_xyxy = cxcywh_to_x1y1x2y2(target_traj)

    MSE_15 = np.square(target_traj_xyxy[:,0:45,:] - all_dec_traj_xyxy[:,0:45,:]).mean(axis=1).mean(axis=1).sum()
    MSE_05 = np.square(target_traj_xyxy[:,0:15,:] - all_dec_traj_xyxy[:,0:15,:]).mean(axis=1).mean(axis=1).sum()
    MSE_10 = np.square(target_traj_xyxy[:,0:30,:] - all_dec_traj_xyxy[:,0:30,:]).mean(axis=1).mean(axis=1).sum()

    FMSE_15 =np.square(target_traj_xyxy[:,44,:] - all_dec_traj_xyxy[:,44,:]).mean(axis=1).sum()
    FMSE_05 =np.square(target_traj_xyxy[:,14,:] - all_dec_traj_xyxy[:,14,:]).mean(axis=1).sum()
    FMSE_10 =np.square(target_traj_xyxy[:,29,:] - all_dec_traj_xyxy[:,29,:]).mean(axis=1).sum()


    CMSE_15 = np.square(target_traj[:,0:45,:2] - all_dec_traj[:,0:45,:2]).mean(axis=1).mean(axis=1).sum()
    CMSE_05 = np.square(target_traj[:,0:15,:2] - all_dec_traj[:,0:15,:2]).mean(axis=1).mean(axis=1).sum()
    CMSE_10 = np.square(target_traj[:,0:30,:2] - all_dec_traj[:,0:30,:2]).mean(axis=1).mean(axis=1).sum()

    CFMSE_15 = np.square(target_traj[:,44,:2] - all_dec_traj[:,44,:2]).mean(axis=1).sum()
    CFMSE_05 = np.square(target_traj[:,14,:2] - all_dec_traj[:,14,:2]).mean(axis=1).sum()
    CFMSE_10 = np.square(target_traj[:,29,:2] - all_dec_traj[:,29,:2]).mean(axis=1).sum()

    FIOU_15 = compute_IOU(target_traj_xyxy[:,44,:], all_dec_traj_xyxy[:,44,:]).sum()
    FIOU_05 = compute_IOU(target_traj_xyxy[:,14,:], all_dec_traj_xyxy[:,14,:]).sum()
    FIOU_10 = compute_IOU(target_traj_xyxy[:,29,:], all_dec_traj_xyxy[:,29,:]).sum()
    
    return MSE_15, MSE_05, MSE_10, FMSE_15, FMSE_05, FMSE_10,\
           CMSE_15, CMSE_05, CMSE_10, CFMSE_15, CFMSE_05, CFMSE_10,\
           FIOU_15,FIOU_05, FIOU_10


