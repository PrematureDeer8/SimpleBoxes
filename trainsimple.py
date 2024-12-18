import torch
from torch.optim import Adam
from SimpleBoxes import SimpleBoxes
from nms import nms
from simpleloss import SimpleBoxesLoss
from utils import match
from prior import PriorBoxes
import numpy as np
import pathlib
from torch.utils.data import DataLoader
from dataloader import SynthtextDataset

if(torch.cuda.is_available()):
    device = ("cuda:0");
# elif(torch.backends.mps.is_available()):
    # device = ("mps"); # <-- this is for apple machines (use the neural cores)
else:
    device = ("cpu");




DIR_PATH = "/data/torre324/.cache/kagglehub/datasets/wstevens11/training-sample-8472/versions/1/training_sample_8472/107"; 

def main():

    print(f"Using device: {device}");
    default_boxes = np.array([[93.79, 44.11], [33.63, 23.26], [214.02, 72.70], [61.02, 32.52], [141.40, 56.21], [334.45, 94.26]]); # k means boxes
    priors = PriorBoxes(default_boxes, ratios=np.array([1,2])); #initialize prior boxes
    model = SimpleBoxes().to(device);
    dataset = SynthtextDataset(DIR_PATH);
    if(pathlib.Path("./state.pt").exists()):
        model.load_state_dict(torch.load("./state.pt"));

    #increasing alpha will make the bounding box loss have a greater weight
    #neg_pos_ratio is for the hard negative mining (smaller number means less negative samples being applied to the loss function)
    loss_func = SimpleBoxesLoss(torch.Tensor([1, 1]).to(device), alpha=1.5, neg_pos_ratio=15); 

    #TODO -- karthik adds backpropagation method
    optimizer = Adam(model.parameters(), lr=.001)

    epochs = 1 # (PLACEHOLDER)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True);

    #change this line if you are going to do different size images

    for epoch in range(epochs):
        model.train(True);
        size = len(train_loader.dataset);
        total_loss = 0;
        for batch, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.to(device)
            level2 = model(x_train);

            anchors = priors([level2.shape], x_train.shape)[0];
            reshape_anchors = anchors.reshape(anchors.shape[0], -1, *anchors.shape[3:]);
            conf_gt = torch.zeros(*anchors.shape[:-1]);

            c_x = x_train.shape[2] / level2.shape[1];
            c_y = x_train.shape[3] / level2.shape[2];
            indices, loc_gt = match(x_train.shape[1:], anchors.shape[1:], y_train);

            conf_gt[indices[...,1], indices[...,2], indices[...,3]] = 1;
            offset_xy = loc_gt[...,:2] - reshape_anchors[...,:2];
            offset_xy[...,0] /= c_x;
            offset_xy[...,1] /= c_y;

            offset_wh = torch.log(loc_gt[...,2:] / reshape_anchors[...,2:]);
            offsets = torch.concat((offset_xy, offset_wh), dim=-1);
            ground_truth = torch.concat((offsets.reshape(offsets.shape[0], -1, offsets.shape[-1]) ,conf_gt.reshape(conf_gt.shape[0], -1).unsqueeze(dim=-1)), dim=-1);

            reshaped_level2 = level2.view(level2.shape[0], -1, level2.shape[-1]);
            
            loss, loc_loss, conf_loss = loss_func(reshaped_level2, ground_truth.to(device));          
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if(batch % 3*(16) == 0):
                    loss, current = loss.item(), (batch + 1) * len(x_train);
                    print(f"loss: {loss:>5f}   loc_loss: {loc_loss:>5f} conf_loss: {conf_loss:5>f} [{current:>5d}/{len(train_loader.dataset):>5}]")


        cur_loss = total_loss / len(train_loader);
        print(f"Epoch {epoch + 1}: {cur_loss}");

    torch.save(model.state_dict(), "./test.pt")


if(__name__ == "__main__"):
    main();