import numpy as np
from SimpleBoxes import SimpleBoxes
from utils import visualize_detect, train
from prior import PriorBoxes
import torch
import cv2 as cv

model = SimpleBoxes();
model.load_state_dict(torch.load("./state.pt", map_location=torch.device("cpu")));
model.eval();
default_boxes = np.array([[93.79, 44.11], [33.63, 23.26], [214.02, 72.70], [61.02, 32.52], [141.40, 56.21], [334.45, 94.26]]);
prior = PriorBoxes(default_boxes, np.array([1,2]));

# function for loading npz file and viewing detections from the model
def load_detections(npz_path, num):
    data = np.load(npz_path);
    img_raw = data["image"];

    img_tensor = torch.from_numpy(img_raw).permute(2,1,0).unsqueeze(dim=0);

    img_bgr = (img_raw - img_raw.min()) / (img_raw.max() - img_raw.min());
    img_bgr = (img_bgr * 255).astype(np.uint8);
    img_bgr = cv.cvtColor(img_bgr, cv.COLOR_RGB2BGR);

    level2 = model(img_tensor);
    anchors = prior([level2.shape], img_tensor.shape);

    visualize_detect([level2], anchors, img_bgr,num);

    
