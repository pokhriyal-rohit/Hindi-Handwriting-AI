import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from src.models.ocr.base import BaseOCRModel
from src.models.ocr.registry import register_ocr_model

class CNNEncoder(nn.Module):
    """
    Reusable VGG-style CNN feature extractor.
    Takes input of shape (B, C, H, W) and outputs (B, Channels, H', W').
    """
    def __init__(self, in_channels=1, out_channels=512):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d((2, 1), (2, 1))
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d((2, 1), (2, 1))
        
        self.conv7 = nn.Conv2d(512, out_channels, kernel_size=2, padding=0)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool4(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x)))))))
        x = F.relu(self.conv7(x))
        return x

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        output = self.linear(recurrent)
        return output

@register_ocr_model("crnn_baseline")
class CRNN(BaseOCRModel):
    """
    Baseline CRNN for Handwriting Recognition.
    """
    def __init__(self, vocab_size: int, config: Dict[str, Any]):
        super(CRNN, self).__init__(vocab_size, config)
        img_channels = config.get("img_channels", 1)
        hidden_size = config.get("hidden_size", 256)
        
        self.cnn = CNNEncoder(in_channels=img_channels, out_channels=512)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, vocab_size)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        conv = self.cnn(images)
        b, c, h, w = conv.size()
        assert h == 1, "The height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        output = self.rnn(conv)
        return output

    def get_output_length(self, input_width: torch.Tensor) -> torch.Tensor:
        """
        The CNN backbone reduces the width dimension by a factor of 4 roughly.
        - pool1: w // 2
        - pool2: w // 4
        - pool3: kernel (2,1), so width is unaffected
        - pool4: kernel (2,1), width unaffected
        - conv7: kernel 2 without padding, width - 1
        """
        return (input_width / 4).to(torch.long) - 1
