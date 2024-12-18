import torch
import numpy as np
import cv2 as cv

#match all candidates to a groundtruth if and only if that groundtruth's center coordinates lie in its grid cell

#image dimensions: (channels, width, height)
#candidates shape: (grid_x, grid_y, 12, 4)
#groundtruth tensor: (img_batch, number of annotations, 2) <-- repeat for the number of levels (torch.stack([groundtruth] * # of levels))
def match(img_dim, candidates, groundtruth):
    batch_size = groundtruth.shape[0];
    grid_x = torch.from_numpy(np.linspace(0, img_dim[1], num=candidates[0], endpoint=False)); # <-- this is wrong we don't want endstep included (FIXED)
    grid_y = torch.from_numpy(np.linspace(0, img_dim[2], num=candidates[1], endpoint=False));

    x_only = groundtruth[...,0].unsqueeze(-1).repeat(1,1,len(grid_x));
    y_only = groundtruth[...,1].unsqueeze(-1).repeat(1,1,len(grid_y));

    x_bool = x_only >= grid_x;
    y_bool = y_only >= grid_y;

    x_indices = x_bool.sum(dim=-1) - 1; #groundtruth will definitely have some tensor that represent no objects
    y_indices = y_bool.sum(dim=-1) - 1; #you will need to account for that before you assign the positive values

    coordinate_indices = torch.zeros(*x_indices.shape, 2);
    coordinate_indices[..., 0] = x_indices;
    coordinate_indices[..., 1] = y_indices;

    bc_indices = torch.zeros(*coordinate_indices.shape[:2], coordinate_indices.shape[2] + 1 + 1); # add one for the batch index and 1 for the index of the ground truth
    batch_idx = torch.arange(batch_size).unsqueeze(dim=-1).repeat(1, groundtruth.shape[1]);
    gt_idx = torch.arange(groundtruth.shape[1]);

    bc_indices[..., 0] = gt_idx;
    bc_indices[..., 1] = batch_idx;
    bc_indices[..., 2:] = coordinate_indices;

    #good luck understanding this
    flatten_indices =  bc_indices.view(-1, bc_indices.shape[2]).long();

    #if flatten indices has a row tensors of the form [a, b, -1, -1] then we must get rid of it
    #because that means the groundtruth tensors is a no object and came in the form of [-1, -1, -1, -1]
    neg_mask = (flatten_indices[...,2:] == torch.Tensor([-1, -1])).any(dim=-1);
    flatten_indices = flatten_indices[~neg_mask];

    unique, return_inverse = torch.unique(flatten_indices[...,1:], dim=0, return_inverse=True);
    mask_range = torch.arange(unique.shape[0]).unsqueeze(dim=-1);
    stack_return_inverse = torch.stack([return_inverse] * unique.shape[0]);
    
    masks = stack_return_inverse == mask_range;
    loc_gt = torch.zeros(batch_size, *candidates);
    #I tried not to use a for loop I really tried!
    loc_gt_tensors = map(lambda args: torch.from_numpy(np.resize(groundtruth[args[0][0], flatten_indices[args[1]][...,0]], (candidates[2], candidates[3]))), zip(unique, masks));

    matched_loc_gt_stack = torch.stack(list(loc_gt_tensors)).float();
    loc_gt[unique[...,0], unique[...,1], unique[...,2]] = matched_loc_gt_stack;
    
    return flatten_indices, loc_gt.reshape(loc_gt.shape[0], -1, *loc_gt.shape[3:]);

def visualize_detect(levels, anchors, img, num=30):
    #assume only one image in the batch
    detections = torch.empty(size=(0, levels[0].shape[-1]));
    # print(f"Detections shape: {detections.shape}");
    img_dim = img.shape;
    for level, anchor in zip(levels, anchors):
        grid_w = img_dim[1] / (level.shape[1]);
        grid_h = img_dim[0] / (level.shape[2]);
        x_pred = level[..., 0] * grid_w + anchor[..., 0];
        y_pred = level[..., 1] * grid_h + anchor[..., 1];
        w_pred =  anchor[..., 2] * torch.exp(level[..., 2]);
        h_pred = anchor[..., 3] * torch.exp(level[..., 3]);
        loc_pred = torch.stack((x_pred, y_pred, w_pred, h_pred), dim=-1);
        # print(f"loc shape: {loc_pred.shape}");
        conf_pred = torch.softmax(level[..., 4:], dim=-1);
        # print(f"conf shape: {level[..., 4:].shape}");
        detection = torch.concat((loc_pred, conf_pred), dim=-1);
        # print(f"Detection shape: {detection.shape}");
        detections = torch.concat((detections, detection.reshape(-1, detection.shape[-1])));
    #sort by most confident values
    indices = torch.argsort(detections[...,4], descending=False);
    sorted_detections = detections[indices];

    #[x_ctr, y_ctr, width, height] --> [x1, y1, x2, y2]
    sorted_detections[..., :2] -= sorted_detections[...,2:4] / 2;
    sorted_detections[..., 2:4] += sorted_detections[...,:2];
    # print(int(sorted_detections[0][0]));
    # sorted_detections = sorted_detections[...,:4];
    for i in range(num):
        print(sorted_detections[i])
        img = cv.rectangle(img, (int(sorted_detections[i][0]), int(sorted_detections[i][1])),\
                            (int(sorted_detections[i][2]), int(sorted_detections[i][3])), (0,0,255), 2);
    
    cv.imwrite("test.jpg", img);  