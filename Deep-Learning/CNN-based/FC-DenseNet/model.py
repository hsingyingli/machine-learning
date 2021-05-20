import torch
import torch.nn as nn 
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels ,kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        out = self.norm(x)
        return out

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.norm       = nn.BatchNorm2d(in_channels)
        self.relu       = nn.ReLU(True)
        self.conv_block = ConvBlock(in_channels, 64)
        self.conv       = nn.Conv2d(64, out_channels, kernel_size = 3, padding = 1)
    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv_block(x)
        out = self.conv(x)
        return out
    
class DenseBlcok(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlcok, self).__init__()
        self.layers = nn.ModuleList([DenseLayer(in_channels + i * growth_rate, 
                                                 growth_rate) for i in range(n_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], 1)
        return x

class TransitionDown(nn.Module):
    def __init__(self, in_channels, original):
        super(TransitionDown, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size = 3, 
                                    stride = 2, padding = 1, groups = in_channels // 2)

        self.conv       = nn.Conv2d(in_channels, in_channels //2, kernel_size = 3, padding = 1)    
        self.pooling    = nn.MaxPool2d(2)                    
        self.norm       = nn.BatchNorm2d(in_channels // 2)
        self.original   = original
    def forward(self, x):
        if self.original:
            x = self.conv(x)
            x = self.pooling(x)
            x = F.dropout(x, p = 0.2, training = self.training)
            out = self.norm(x)
            return out
        else:
            x = self.depth_conv(x)
            out = self.norm(x)
            # x = F.dropout(x, p = 0.2, training = self.training)
            return out

class TransitionUp(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__() 
        self.convTrans = nn.ConvTranspose2d( 
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=3, stride=2, padding=0, bias=True) 
    def forward(self, x): 
        out = self.convTrans(x) 
        return out 

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(Bottleneck, self).__init__()
        self.neck = DenseBlcok(in_channels, growth_rate, n_layers)

    def forward(self, x):
        return self.neck(x)



class FCDenseNet(nn.Module):
    def __init__(self, blocks, growth_rate, in_channels, n_classes, original):
        super(FCDenseNet, self).__init__()
        self.first_conv_layer = ConvBlock(in_channels, n_classes, kernel_size =  3, stride = 2, padding = 1)
        self.denseBlockDown = nn.ModuleList([])
        self.transBlockDown = nn.ModuleList([])
        self.denseBlockUp   = nn.ModuleList([])
        self.transBlockUp   = nn.ModuleList([])

        cur_channels = n_classes
        skip_channels = []

        for i in range(len(blocks)-1):
            self.denseBlockDown.append(DenseBlcok(cur_channels, growth_rate, blocks[i]))
            cur_channels += growth_rate * blocks[i]
            skip_channels.append(cur_channels)
            self.transBlockDown.append(TransitionDown(cur_channels, original))
            cur_channels = cur_channels // 2

        self.bottleneck = Bottleneck(cur_channels, growth_rate, blocks[-1])
        cur_channels += growth_rate * blocks[-1]

        for i in range(len(blocks)-2, -1, -1):
            self.transBlockUp.append(TransitionUp(cur_channels, growth_rate * (blocks[i] +1) ))
            cur_channels =  growth_rate * (blocks[i] +1) + skip_channels[i]
            self.denseBlockUp.append(DenseBlcok(cur_channels, growth_rate, blocks[i]))
            cur_channels += growth_rate * (blocks[i])


        self.upsample   = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.final_conv = nn.Conv2d(cur_channels, n_classes, kernel_size = 1)
        self.softmax    = nn.Softmax(dim = 1)

    def forward(self, x):
        skip_connection = []
        out = self.first_conv_layer(x)
        for i in range(len(self.denseBlockDown)):
            out = self.denseBlockDown[i](out)
            skip_connection.append(out)
            out = self.transBlockDown[i](out)

        out = self.bottleneck(out)
       

        for i in range(len(self.denseBlockUp)):
            out = self.transBlockUp[i](out)
            skip = skip_connection.pop()
            out = self.center_crop(out, skip.shape[2], skip.shape[3])
            out = torch.cat([out, skip], 1)
            out = self.denseBlockUp[i](out)
            
        out = self.upsample(out)
        out = self.final_conv(out)
        
        return self.softmax(out)


    def center_crop(self, x, height, width):
        _, _, h, w = x.shape
        xy1 = (w - width)  // 2
        xy2 = (h - height) // 2

        return x[:, :, xy2:(xy2 + height), xy1:(xy1 + width)]


if __name__ == "__main__":
    
    test = FCDenseNet([4,5,6,7,8], 16, 3, 32)
    print(test)
    data = torch.rand(32,3, 64, 64)
    out = test(data)
    print("input  shape : ", data.shape)
    print("output shape : ", out.shape)