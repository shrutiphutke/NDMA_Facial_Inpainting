import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.kv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.kv_conv = DeformableConv2d(channels * 2, channels *2)#nn.Conv2d(channels * 2, channels *2, kernel_size=3, padding=1, groups=channels * 2, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x, q):
        b, c, h, w = x.shape
        k, v = self.kv_conv(self.kv(x)).chunk(2, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out, q

class Nested_MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(Nested_MDTA, self).__init__()
        self.pack_attention = MDTA(channels, num_heads)
        self.unpack_attention = MDTA(channels, num_heads)

    def forward(self,x, p):
        packed_context, query = self.pack_attention(x, p)
        unpacked_context, _ = self.unpack_attention(packed_context, query)
        return unpacked_context, packed_context

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x

class LunaTransformerEncoderLayer(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(LunaTransformerEncoderLayer, self).__init__()
        self.luna_attention = Nested_MDTA(channels, num_heads)
        self.feed_forward = GDFN(channels, expansion_factor)
        self.packed_context_layer_norm = nn.LayerNorm(channels)
        self.unpacked_context_layer_norm = nn.LayerNorm(channels)
        # self.unpacked_context_layer_norm = nn.LayerNorm(channels)
        self.feed_forward_layer_norm = nn.LayerNorm(channels)

    def forward(self, x, p):
        b, c, h, w = x.shape
        unpacked_context, packed_context = self.luna_attention(x,p)

        packed_context = self.packed_context_layer_norm((packed_context + p).reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)

        unpacked_context = self.unpacked_context_layer_norm((unpacked_context + x).reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)

        outputs = self.feed_forward(unpacked_context)

        outputs = self.feed_forward_layer_norm((outputs + unpacked_context).reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)

        return outputs, packed_context


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          stride=self.stride,
                                          )
        return x

class GatedConv2dWithActivation(torch.nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=False, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x

class GatedDeConv2dWithActivation(torch.nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=False,activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv2dWithActivation, self).__init__()
        self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)


class res_bottleneck(nn.Module):
    def __init__(self, in_ch, num_heads, expansion_factor = 2.21):
        super(res_bottleneck, self).__init__()


        self.res_block1 = LunaTransformerEncoderLayer(in_ch, num_heads, expansion_factor)
        self.res_block2 = LunaTransformerEncoderLayer(in_ch, num_heads, expansion_factor)
        self.res_block3 = LunaTransformerEncoderLayer(in_ch, num_heads, expansion_factor)
        self.res_block4 = LunaTransformerEncoderLayer(in_ch, num_heads, expansion_factor)
        self.res_block5 = LunaTransformerEncoderLayer(in_ch, num_heads, expansion_factor)
        self.res_last = nn.Conv2d(in_ch*2, in_ch, kernel_size=1, bias=False)

    def forward(self, x):

        res1, res1_p1 = self.res_block1(x,x)

        res2, res1_p2 = self.res_block2(res1, res1_p1)

        res3 , res1_p3= self.res_block3(res2+res1, res1_p2+res1_p1)

        res4, res1_p4 = self.res_block4(res3+res2+res1 , res1_p3+res1_p2+res1_p1)

        res5, res1_p5 = self.res_block5(res4+res3+res2+res1, res1_p4+res1_p3+res1_p2+res1_p1)

        res = torch.cat([res5, res1_p5], axis = 1)
        out = self.res_last(res)

        return out


class Luna_Net(nn.Module):
    def __init__(self, in_channels, out_channels, factor):
        super(Luna_Net, self).__init__()

        self.Conv1 = GatedConv2dWithActivation(in_channels, 64//factor,  kernel_size=3, stride = 1, padding=1)

        self.Conv2 = GatedConv2dWithActivation(64//factor,  128//factor,   kernel_size=3, stride = 2, padding=1)
        self.Conv3 = GatedConv2dWithActivation(128//factor, 256//factor,   kernel_size=3, stride = 2, padding=1)
        self.Conv4 = GatedConv2dWithActivation(256//factor, 512//factor,  kernel_size=3, stride = 2, padding=1)
        self.Conv5 = GatedConv2dWithActivation(512//factor, 1024//factor,  kernel_size=3, stride = 2, padding=1)

        self.dil_conv1 = GatedConv2dWithActivation(1024//factor, 1024//factor, kernel_size=3,  stride = 1, padding=1)
        self.dil_conv2 = GatedConv2dWithActivation(1024//factor, 1024//factor,  kernel_size=3, stride = 1, padding=1)
        self.dil_conv3 = GatedConv2dWithActivation(1024//factor, 1024//factor,  kernel_size=3, stride = 1, padding=1)
        self.dil_conv4 = GatedConv2dWithActivation(1024//factor, 1024//factor,  kernel_size=3, stride = 1, padding=1)

        self.Up5 = GatedDeConv2dWithActivation(2, 1024//factor, 512//factor, kernel_size=3, padding = 1)
        self.Up_conv5 = GatedConv2dWithActivation(1024//factor, 512//factor, kernel_size=3, stride =1 , padding=1)

        self.Up4 = GatedDeConv2dWithActivation(2, 512//factor, 256//factor, kernel_size=3, padding = 1)
        self.Up_conv4 = GatedConv2dWithActivation(512//factor, 256//factor, kernel_size=3, stride = 1, padding=1)
        
        self.Up3 = GatedDeConv2dWithActivation(2, 256//factor, 128//factor, kernel_size=3, padding = 1)
        self.Up_conv3 = GatedConv2dWithActivation(256//factor, 128//factor, kernel_size=3, stride = 1, padding=1)
        
        self.Up2 = GatedDeConv2dWithActivation(2, 128//factor, 64//factor, kernel_size=3, padding = 1)
        self.Up_conv2 = GatedConv2dWithActivation(64//factor, 64//factor, kernel_size=3, stride = 1, padding=1)

        self.Conv_1x1 = nn.Conv2d(64//factor, out_channels, 1)


        self.p2_Conv1 = GatedConv2dWithActivation(in_channels, 64//factor,  kernel_size=3, stride = 1, padding=1)

        self.p2_Conv2 = GatedConv2dWithActivation(64//factor,  128//factor,   kernel_size=3, stride = 2, padding=1)
        self.p2_Conv3 = GatedConv2dWithActivation(128//factor, 256//factor,   kernel_size=3, stride = 2, padding=1)
        self.p2_Conv4 = GatedConv2dWithActivation(256//factor, 512//factor,  kernel_size=3, stride = 2, padding=1)
        self.p2_Conv5 = GatedConv2dWithActivation(512//factor, 1024//factor,  kernel_size=3, stride = 2, padding=1)

        self.p2_dil_conv1 = GatedConv2dWithActivation(1024//factor, 1024//factor,  kernel_size=3, stride = 1, padding=1)
        self.p2_dil_conv2 = GatedConv2dWithActivation(1024//factor, 1024//factor,  kernel_size=3, stride = 1, padding=1)
        self.p2_dil_conv3 = GatedConv2dWithActivation(1024//factor, 1024//factor,  kernel_size=3, stride = 1, padding=1)
        self.p2_dil_conv4 = GatedConv2dWithActivation(1024//factor, 1024//factor,  kernel_size=3, stride = 1, padding=1)

        self.p2_Up5 = GatedDeConv2dWithActivation(2, 1024//factor, 512//factor, kernel_size=3, padding = 1)
        self.gmlp_attn1 = LunaTransformerEncoderLayer(512//factor, 8, 2.33)
        self.p2_Up_conv5 = GatedConv2dWithActivation(1024//factor, 512//factor,  kernel_size=3, stride = 1, padding=1)

        self.p2_Up4 = GatedDeConv2dWithActivation(2, 512//factor, 256//factor, kernel_size=3, padding = 1)
        self.gmlp_attn2 = LunaTransformerEncoderLayer(256//factor, 8, 2.33)
        self.p2_Up_conv4 = GatedConv2dWithActivation(512//factor, 256//factor,  kernel_size=3, stride = 1, padding=1)
        
        self.p2_Up3 = GatedDeConv2dWithActivation(2, 256//factor, 128//factor, kernel_size=3, padding = 1)
        self.gmlp_attn3 = LunaTransformerEncoderLayer(128//factor, 8, 2.33)
        self.p2_Up_conv3 = GatedConv2dWithActivation(256//factor, 128//factor,  kernel_size=3, stride = 1, padding=1)
        
        self.p2_Up2 = GatedDeConv2dWithActivation(2, 128//factor, 64//factor, kernel_size=3, padding = 1)
        self.gmlp_attn4 = LunaTransformerEncoderLayer(64//factor, 8, 2.33)
        self.p2_Up_conv2 = GatedConv2dWithActivation(128//factor, 64//factor,  kernel_size=3, stride = 1, padding=1)

        self.p2_Conv_1x1 = nn.Conv2d(64//factor, out_channels, 1)


        

    def forward(self, in1, in2):
        

        # encoding path
        x = torch.cat((in1,in2),dim=1)
        x1 = self.Conv1(x)

       
        x2 = self.Conv2(x1)
        
        
        x3 = self.Conv3(x2)

        
        x4 = self.Conv4(x3)

        
        x5 = self.Conv5(x4)

        dil1 = self.dil_conv1(x5)
        dil2 = self.dil_conv2(dil1)
        dil3 = self.dil_conv3(dil2)
        dil4 = self.dil_conv4(dil3)

        # decoding + concat path
        d5 = self.Up5(dil4)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        second_masked_img = in1 * (1 - in2) + d1 * in2
        p2_x = torch.cat((second_masked_img,in2),dim=1)


        p2_x1 = self.p2_Conv1(p2_x)
        p2_x2 = self.p2_Conv2(p2_x1)
        p2_x3 = self.p2_Conv3(p2_x2)
        p2_x4 = self.p2_Conv4(p2_x3)
        p2_x5 = self.p2_Conv5(p2_x4)

        p2_dil1 = self.p2_dil_conv1(p2_x5)
        p2_dil2 = self.p2_dil_conv2(p2_dil1)
        p2_dil3 = self.p2_dil_conv3(p2_dil2)
        p2_dil4 = self.p2_dil_conv4(p2_dil3)

        # decoding + concat path
        p2_d5 = self.p2_Up5(p2_dil4)
        o2_x4_skip, p2_x4_skip = self.gmlp_attn1(p2_d5, p2_x4)
        p2_d5 = torch.cat((o2_x4_skip, p2_x4_skip),dim=1)
        
        p2_d5 = self.p2_Up_conv5(p2_d5)
        
        p2_d4 = self.p2_Up4(p2_d5)
        o2_x3_skip, p2_x3_skip = self.gmlp_attn2(p2_d4, p2_x3)
        p2_d4 = torch.cat((o2_x3_skip, p2_x3_skip),dim=1)
        p2_d4 = self.p2_Up_conv4(p2_d4)

        p2_d3 = self.p2_Up3(p2_d4)
        o2_x2_skip, p2_x2_skip = self.gmlp_attn3(p2_d3, p2_x2)
        p2_d3 = torch.cat((o2_x2_skip, p2_x2_skip),dim=1)
        p2_d3 = self.p2_Up_conv3(p2_d3)

        p2_d2 = self.p2_Up2(p2_d3)
        o2_x1_skip, p2_x1_skip = self.gmlp_attn4(p2_d2, p2_x1)
        p2_d2 = torch.cat((o2_x1_skip, p2_x1_skip),dim=1)
        p2_d2 = self.p2_Up_conv2(p2_d2)

        p2_d1 = self.p2_Conv_1x1(p2_d2)


        return d1, p2_d1