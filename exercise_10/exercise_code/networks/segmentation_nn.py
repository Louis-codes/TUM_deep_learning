"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x



class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        # Transfer learning: pretrained MobileNetV2 feature extractor as encoder
        mobilenet = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.encoder = mobilenet.features  # (N, 3, H, W) -> (N, 1280, H/32, W/32)

        # Decoder: 1x1 conv to shrink channels, then upsample + conv blocks
        # (plain upsampling + convolution avoids checkerboard artifacts)
        self.decoder = nn.Sequential(
            nn.Conv2d(1280, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvLayer(128, 128),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvLayer(128, 64),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvLayer(64, 64),

            nn.Conv2d(64, num_classes, kernel_size=1),
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        input_size = x.shape[2:]
        x = self.encoder(x)
        x = self.decoder(x)
        # final upsample back to the exact input resolution
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")