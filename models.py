import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from common import FCABs
from einops import rearrange
from natten import NeighborhoodNonlocalAttention2D

# Note: residual/upsample blocks are adapted from EDVR: https://github.com/xinntao/BasicSR
def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        identity = x
        out = self.conv2((self.relu(self.conv1(x))))
        #out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class Upnet(nn.Module):
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(Upnet, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        scale_factor = 4
        self.upscale = []
        for _ in range(scale_factor // 2):
            self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                 nn.PixelShuffle(2)])
        self.upscale = nn.Sequential(*self.upscale)
        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        x = self.upscale(x)
        x = self.output(x)
        return x

class nonlocal_attention(nn.Module):
    def __init__(self, config, is_training=True):
        super(nonlocal_attention, self).__init__()
        self.in_channels = config['in_channels']
        self.inter_channels = self.in_channels // 2
        self.is_training=is_training
        width=config['width']
        height=config['height']
                
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Conv2d(in_channels=self.inter_channels*7, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.W_z.weight, 0)
        nn.init.constant_(self.W_z.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=self.in_channels, eps=1e-6, affine=True)
        
        x1=np.linspace(0,width-1,width)
        y1=np.linspace(0,height-1,height)
        x2=np.linspace(0,width-1,width)
        y2=np.linspace(0,height-1,height)
        X1,Y1,Y2,X2=np.meshgrid(x1,y1,y2,x2)
        D=(X1-X2)**2+(Y1-Y2)**2
        D=torch.from_numpy(D)
        D=rearrange(D, 'a b c d -> (a b) (c d)')
        if self.is_training:
            D=D.float()
        else:
            D=D.half()
        #self.D = D
        #self.D=torch.nn.Parameter(D,requires_grad=False)
        #self.std=torch.nn.Parameter(4*torch.ones(1).float())
        #if self.is_training==False:
            #self.std.data=self.std.half()
        self.W_z1 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.W_z1.weight, 0)
        nn.init.constant_(self.W_z1.bias, 0)
        self.mb = torch.nn.Parameter(torch.randn(self.inter_channels, 256))

        self.att = NeighborhoodNonlocalAttention2D(dim = self.inter_channels, num_heads=1,kernel_size=3)
    
    def forward(self, x):
        b, t, c, h, w = x.size()
        q = x[:, 3, :, :, :]
        
        #weight=torch.exp(-0.5*(self.D/(self.std*self.std)))
        #weight=weight.unsqueeze(0).repeat(b,1,1)

        reshaped_x = x.view(b*t , c, h, w).contiguous()
        h_ = self.norm(reshaped_x)
        q_=self.norm(q)

        g_x = self.g(h_).view(b, t, self.inter_channels, h,w).contiguous()
        theta_x = self.theta(h_).view(b, t, self.inter_channels,  h,w).contiguous()
        # phi_x = self.phi(q_).view(b,self.inter_channels, -1)
        # phi_x_for_quant=phi_x.permute(0,2,1)
        # phi_x= phi_x.permute(0,2,1).contiguous()
        phi_x = torch.unsqueeze(self.phi(q_),dim=1)
        phi_x  =phi_x.permute(0,1,3,4,2).contiguous()
        phi_x_for_quant = phi_x.permute(0,1,4,2,3).view(b,self.inter_channels, -1).permute(0,2,1)

        corr_l = []
        #for i in range(3,4):
        for i in range(t):
            theta = theta_x[:, i, :, :, :]
            g = g_x[:, i, :, :, :]

            # g = g.view(b, self.inter_channels, -1).permute(0,2,1).contiguous()
            # theta = theta.view(b, self.inter_channels, -1).contiguous()
            g = torch.unsqueeze(g.permute(0,2,3,1).contiguous(),dim=1)
            theta = torch.unsqueeze(theta.permute(0,2,3,1).contiguous(),dim=1)
            
            if self.is_training:
                #f = torch.matmul(phi_x, theta)
                f = self.att(phi_x, theta, g)
            else: 
                f = self.att(phi_x, theta, g)
                # del f
                # torch.cuda.empty_cache()  # 释放显存
                # #print(torch.cuda.memory_summary())
                # f = torch.matmul(phi_x.half(), theta.half())
                # #print(torch.cuda.memory_summary())
            # torch.cuda.empty_cache()
            # #print(torch.cuda.memory_summary())                
            #f = F.softmax(f, dim=-1)

            #print(torch.cuda.memory_summary())
            #f_div_C = F.softmax(f, dim=-1)*weight
            # if self.is_training:
            #     y = torch.matmul(f, g).float()
            # else:
            #     y = torch.matmul(f, g.half()).float()
            # y=y.permute(0,2,1).view(b,self.inter_channels,h,w)
            y = f.view(b,self.inter_channels,h,w)
            corr_l.append(y)
            #del f, y, g, theta
            #time.sleep(0.03)
            #torch.cuda.empty_cache()
            #torch.cuda.empty_cache()

        corr_prob = torch.cat(corr_l, dim=1).view(b, -1, h, w)
        W_y = self.W_z(corr_prob)
        
        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
        f1 = torch.matmul(phi_x_for_quant, mbg)
        f_div_C1 = F.softmax(f1 * (int(self.inter_channels) ** (-0.5)), dim=-1)
        y1 = torch.matmul(f_div_C1, mbg.permute(0, 2, 1))
        qloss=torch.mean(torch.abs(phi_x_for_quant-y1))
        y1 = y1.permute(0, 2, 1).view(b, self.inter_channels, h, w).contiguous()
        W_y1 = self.W_z1(y1)
        
        z = W_y + q+W_y1
        # z = W_y + q
        # qloss = 0

        return z, qloss


class mana(nn.Module):
    def __init__(self, config,is_training):
        super(mana, self).__init__()
        self.in_channels = config['in_channels']
        self.conv_first = nn.Conv2d(1, self.in_channels, 3, 1, 1)
        #self.conv_fus = nn.Conv2d(self.in_channels*7, self.in_channels, 1)
        self.conv_input = nn.Conv2d(1, 64, 3, 1, 1)
        self.dropout = nn.Dropout(p=0.3)
        #self.FCAblock1 = FCAB(config['in_channels'], size_psc=config['patch_size'])
        #self.FCAblock2 = FCAB(config['in_channels'], size_psc=config['patch_size'])
        #self.FCAblock3 = FCAB(config['in_channels'], size_psc=config['patch_size'])
        #self.FCAblock4 = FCAB(config['in_channels'], size_psc=config['patch_size'])

        #self.encoder = make_layer(ResidualBlockNoBN, config['encoder_nblocks'], num_feat=self.in_channels)
        #self.decoder = make_layer(ResidualBlockNoBN, config['decoder_nblocks'], num_feat=self.in_channels) 
        self.encoder = make_layer(FCABs, 2, channel=self.in_channels)
        self.decoder = make_layer(FCABs, 5, channel=self.in_channels)

        self.nonloc_spatial = nonlocal_attention(config, is_training)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.sig = nn.Sigmoid()

        # upsample
        self.upconv1 = nn.Conv2d(self.in_channels, 64 * 9, 3, 1, 1)
        #self.upconv1 = nn.Conv2d(self.in_channels, self.in_channels * 9, 3, 1, 1)
        self.upconv2 = nn.Conv2d(self.in_channels, 64 * 9, 3, 1, 1) #这里也要改
        self.pixel_shuffle = nn.PixelShuffle(3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1)
        self.conv_last1 = nn.Conv2d(64, 1, 3, 1, 1)
        self.conv_last2 = nn.Conv2d(64, 1, 3, 1, 1)


    def forward(self, x):
        b, t, c, h, w = x.size()
        lx = F.interpolate(x[:, 3, :, :, :], scale_factor=3, mode='bilinear', align_corners=False)#这里要改
        
        encode = self.encoder((self.lrelu(self.conv_first(x.view(-1, c, h, w)))))
        #encode = self.encoder(self.FCAblock1(self.lrelu(self.conv_first(x.view(-1, c, h, w)))))
        #encode = self.FCAblock2(encode)
        mem=encode.view(b,t,self.in_channels,h,w).contiguous()
        # mem = mem.view(b,-1,h,w)
        # res = self.conv_fus(mem)
        # qloss = 0

        res,qloss = self.nonloc_spatial(mem)
        #res = self.FCAblock3(res)
        out = self.decoder(res)
        #out = self.FCAblock4(out)
        #out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        #out = self.dropout(self.lrelu(self.pixel_shuffle(self.upconv2(out))))
        #out = self.dropout(self.lrelu(self.conv_hr(out)))
        
        out1 = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out2 = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out1 = self.lrelu(self.conv_hr1(out1))
        out2 = self.lrelu(self.conv_hr2(out2))

        feature_input = self.conv_input(lx)
        out1 = out1 + feature_input
        out2 = out2 + feature_input

        out1 = self.conv_last1(out1)
        out1 = lx + out1

        out2 = self.conv_last2(out2)
        out2 = self.sig(out2)

        oup = torch.cat((out1, out2), 1)
        
        '''
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        feature_input = self.conv_input(lx)
        out_add = out + feature_input
        outs = self.conv_last(out_add)
        #mean_map = torch.add(lx,out[:,0:1,:,:])
        std_map = self.sig(outs[:,1:2,:,:])
        oup = torch.cat((outs[:,0:1,:,:],std_map), 1)
        '''
        
        # out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        # out = self.lrelu(self.conv_hr(out))
        # out = self.conv_last(out)
        # oup = lx + out
        
        return oup,qloss