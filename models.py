import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        # input shape: (b x 3 x 227 x 227)
        self.features = nn.Sequential(
            # layer1
            nn.Conv2d(3, 96, kernel_size=11, stride=4),                 # (b x 96 x 55 x 55)
            nn.ReLU(True),                                              # (b x 96 x 55 x 55)
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),   # (b x 96 x 55 x 55)
            nn.MaxPool2d(kernel_size=3, stride=2),                      # (b x 96 x 27 x 27)

            # layer2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),               # (b x 256 x 27 x 27), (2, 4, 5번 째 conv layer에서는 같은 GPU에 있는 커널맵 연산)
            nn.ReLU(True),                                              # (b x 256 x 27 x 27)
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),   # (b x 256 x 27 x 27)
            nn.MaxPool2d(kernel_size=3, stride=2),                      # (b x 256 x 13 x 13)

            # layer3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),              # (b x 384 x 13 x 13)
            nn.ReLU(True),                                              # (b x 384 x 13 x 13)

            # layer4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),    # (b x 384 x 13 x 13)
            nn.ReLU(True),                                              # (b x 384 x 13 x 13)

            # layer5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),    # (b x 256 x 13 x 13)
            nn.ReLU(True),                                              # (b x 256 x 13 x 13)
            nn.MaxPool2d(kernel_size=3, stride=2)
        )        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(True),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
 