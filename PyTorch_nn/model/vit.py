import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224,patch_size=16,in_c=3,embed_dim=768,norm_layer=None):
        # img size 图像大小   patch size 每个patch的大小 
        super().__init__()
        img_size = (img_size,img_size) #将输入的图像大小变为二维元组
        patch_size = (patch_size,patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.gird_size = (img_size[0]//patch_size[0],img_size[1]//patch_size[1]) #patch的网格大小  224//16=14   (14,14)
        self.num_patches = self.gird_size[0]*self.gird_size[1] # 14*14=196 patch的总数

        self.proj = nn.Conv2d(in_c,embed_dim,kernel_size=patch_size,stride=patch_size) # B,3,224,224->B,768,14,14
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity() # 若有layer norm则使用  若无则保持不变

    def forward(self,x):
        B,C,H,W = x.shape  # 获取我门输入张量的形状
        assert H==self.img_size[0] and W==self.img_size[1],\
        f"输入图像大小{H}*{W}与模型期望大小{self.img_size[0]}*{self.img_size[1]}不匹配"
        #B,3,224,224->B,768,14,14 -> B，768，196 ->B ,196,768
        x = self.proj(x).flatten(2).transpose(1,2)
        x = self.norm(x) # 若有归一化层 则使用
        return x
