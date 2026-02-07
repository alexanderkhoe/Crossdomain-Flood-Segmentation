import torch.nn as nn

from models.prithvi_cafe.encoder import AdaptedPrithvi  
from models.prithvi_cafe.decoder import PT2Decoder 

class PrithviCafe(nn.Module):
    def __init__(self, in_channels=6, num_classes=2):
        super().__init__()
        self.encoder = AdaptedPrithvi(in_channels=in_channels)
        # PT2Decoder expects [160, 320, 640, 1280] but AdaptedPrithvi outputs these after FAT_Net fusion
        self.decoder = PT2Decoder(embed_dim=[160, 320, 640, 1280])
        self.head = nn.Conv2d(self.decoder.out_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        features = self.encoder(x)  # Returns 4 feature maps
        decoder_out = self.decoder(features)  # Returns high-res feature
        logits = self.head(decoder_out)
        return logits