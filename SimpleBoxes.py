import torch.nn as nn
import torch
from torchvision.models import vgg16
from models.conv_layers import Conv_Layers
from models.text_box_layers import Text_box_Layer

if(torch.cuda.is_available()):
    device = ("cuda:0");
# elif(torch.backends.mps.is_available()):
    # device = ("mps"); # <-- this is for apple machines (use the neural cores)
else:
    device = ("cpu");

#a one level alternative
class SimpleBoxes(nn.Module):

    def __init__(self):
        super(SimpleBoxes, self).__init__();
        out_channels = [512, 1024, 512, 256, 256, 256];
        # VGG backbone with IMAGENET1K weights
        backbone = vgg16().features[:23]; #TODO -- no autograd on these weights
        self.layers = nn.Sequential(
            *backbone,
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels[2], 72, kernel_size=(1,5), padding=(0,2))
        );
        self.sigmoid = nn.Sigmoid();
        self.tanh = nn.Tanh();
        self.ReLU = nn.ReLU();
    
    def forward(self, x):

        level2 = self.layers(x);
        level2 = level2.view(level2.shape[0],  *level2.shape[2:],-1, 6);
        level2[..., :2] = self.sigmoid(level2[..., :2]);
        level2[..., 2:4] = self.tanh(level2[..., 2:4]);
        level2[..., 4:] = self.ReLU(level2[..., 4:]);

        return level2;

        