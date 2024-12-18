import torch.nn as nn
import torch
import numpy as np
from decimal import Decimal


class PriorBoxes(nn.Module):
    # default_boxes --> m x 2 numpy array of (width, height) of each bounding box (m = # of default bounding boxes)
    # ratios --> aspect ratios
    # scale --> scale we apply to each bounding box in a feature map (lower levels have smaller scales and vice versa)
    #np.array([[93.79, 44.11], [33.63, 23.26], [214.02, 72.70], [61.02, 32.52], [141.40, 56.21], [334.45, 94.26]])
    def __init__(self, default_boxes, ratios=np.array([1,2,3,5,7,10]),scale_bounds=[0.75, 1.25]):
        super(PriorBoxes, self).__init__();
        self.default_boxes = default_boxes;
        self.ratios = np.empty(shape=(len(ratios), 2));
        for i, ratio in enumerate(ratios):
            self.ratios[i] = Decimal(float(ratio)).as_integer_ratio();
        self.aspect_ratio_matrix = self.ratios.repeat(len(self.default_boxes), axis=0);
        self.scale_bounds = scale_bounds;
        self.stack = np.stack(tuple([self.default_boxes]*len(self.ratios))).reshape(len(self.ratios) * len(self.default_boxes), -1);
    
    # feature_map_dim --> list of dimensions for feature maps
    #img_dim: (batch_size, channels, width, heigth)
    def forward(self, feature_map_dims, img_dim):
        if((len(self.default_boxes) * len(self.ratios)) != (feature_map_dims[0][3])):
            raise ValueError(f"Number of bounding boxes from feature map ({feature_map_dims[0][1]}) must be equal to number of anchor boxes({len(self.default_boxes)} * {len(self.ratios)})!");

        scales = np.linspace(self.scale_bounds[0], self.scale_bounds[1], num=len(feature_map_dims));
        #initialize a tensor for the anchor boxes at each level
        self.anchors = [np.zeros(shape=(*dim[1:3], len(self.default_boxes) * len(self.ratios), 4)) for dim in feature_map_dims ];
        #define the width and height of each anchor box
        #stack = np.stack(tuple([self.default_boxes]*len(self.ratios))).reshape(len(self.ratios) * len(self.default_boxes), -1);
        #unsqueeze and transpose
        #repeat twice for the width and height on the innermost axis
        #repeat by the number of default boxes on the outermost axis
        # aspect_ratio_matrix = np.expand_dims(self.ratios, 0).T.repeat(2, axis=-1).repeat(len(self.default_boxes), axis=0);
        for i in range(len(self.anchors)):
            grid_x = np.linspace(0, img_dim[2], num=self.anchors[i].shape[0], endpoint=False);
            grid_y = np.linspace(0, img_dim[3], num=self.anchors[i].shape[1], endpoint=False);
            stack = np.stack(np.meshgrid(grid_x, grid_y), axis=0);
            stack_T = np.expand_dims(np.transpose(stack, (2,1,0)), axis=2);
            stack_T = stack_T.repeat(12, axis=2);
            #add actual xy coordinates of center point for anchors
            self.anchors[i][..., :2] = stack_T
            #multiply by scales (smaller scales should correspond to feature maps with higher dimensions)
            self.anchors[i][..., 2:] = self.stack * self.aspect_ratio_matrix * scales[i];
        return [torch.stack([torch.from_numpy(anchor.astype(np.float32))] * img_dim[0]) for anchor in self.anchors];
