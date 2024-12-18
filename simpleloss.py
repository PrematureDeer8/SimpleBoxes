
import torch
import torch.nn as nn

class SimpleBoxesLoss(nn.Module):
    def __init__(self, weight, alpha=1.0, neg_pos_ratio=3.0):
        '''
        Initializes Loss function parameters

        alpha: Weight for localization loss
        neg_pos_ratio: Ratio of negative to positive examples to use (do not want to use all negative examples)
        '''
        super(SimpleBoxesLoss, self).__init__()
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio

        # Functions for confidence and localization losses, respectively
        self.conf_loss_fn = nn.CrossEntropyLoss(weight=weight, reduction='sum') # PyTorch applies Softmax in this function, no need to do it separately
        self.loc_loss_fn = nn.SmoothL1Loss(reduction='sum', beta=0.5);

    #predictions shape --> (batch_size, num of predictions, 6)
    #ground_truth shape --> (batch_size, num of predictions, 5)
    def forward(self, predictions, ground_truth):
        pos_mask = ground_truth[...,4].bool();
        positive_predictions = predictions[pos_mask];
        positive_gt = ground_truth[pos_mask];
        indices = ground_truth[...,4].long();
        loc_loss = self.loc_loss_fn(positive_predictions[...,:4], positive_gt[...,:4]);
        conf_loss = self.conf_loss_fn(predictions[...,4:].permute(0, 2, 1), indices);

        loss = (self.alpha * loc_loss) + conf_loss;
        return loss, (self.alpha * loc_loss), conf_loss;
        

        