import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

class Downsample(nn.Module):
    """下采样块"""
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        return self.reduction(x)

class Stem(nn.Module):
    """初始特征提取 - 修改为接受单通道输入"""
    
    def __init__(self, in_chans=1, out_chans=64):  # 改为单通道
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, 2, 1, bias=False),  # 输入通道改为1
            nn.BatchNorm2d(32, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(32, out_chans, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_chans, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        return self.stem(x)

class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x

class MambaVisionMixer(nn.Module):
    """完整的Mamba混合器"""
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        
        # 初始化参数
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        y = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=None, 
                             delta_bias=self.dt_proj.bias.float(), delta_softplus=True, return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

def window_partition(x, window_size):
     """窗口分割"""
     B, C, H, W = x.shape
     x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
     windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
     return windows

def window_reverse(windows, window_size, H, W):
    """窗口重组"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x

class Block(nn.Module):
    """基础块"""
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class MambaVisionLayer(nn.Module):
    """Mamba视觉层 - 确保下采样正常工作"""
    
    def __init__(self, dim, depth, window_size, downsample=True,  # 确保有这个参数
                 mlp_ratio=4., drop=0., drop_path=0., layer_scale=None):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(dim=dim, mlp_ratio=mlp_ratio, drop=drop, 
                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                  layer_scale=layer_scale)
            for i in range(depth)
        ])

        self.downsample = None
        if downsample:
            self.downsample = Downsample(dim=dim)  # 正确设置下采样
        self.window_size = window_size

    def forward(self, x):
        B, C, H, W = x.shape

        # 窗口化处理
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W
        
        x = window_partition(x, self.window_size)

        for blk in self.blocks:
            x = blk(x)

        x = window_reverse(x, self.window_size, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()

        if self.downsample is not None:
            x = self.downsample(x)  # 应用下采样
            
        return x

class TrafficMambaVision(nn.Module):
    """修正的交通MambaVision模型 - 正确的下采样流程"""
    
    def __init__(self, in_chans=1, num_classes=1, depths=[3, 3, 2, 2],
                 dims=[64, 128, 256, 512, 1024], window_sizes=[8, 8, 14, 7],
                 mlp_ratio=4, drop_rate=0., drop_path_rate=0.2, layer_scale=None):
        super().__init__()
        
        self.dims = dims
        self.num_classes = num_classes
        
        # Stem: 2次下采样 (64x64 -> 16x16)
        self.stem = Stem(in_chans=in_chans, out_chans=dims[0])
        
        # Stage 1: 1次下采样 (16x16 -> 8x8)
        self.stage1 = nn.Sequential(
            *[ConvBlock(dim=dims[0], drop_path=drop_path_rate) for _ in range(depths[0])],
            Downsample(dim=dims[0])  # 64 -> 128
        )
        
        # Stage 2: 1次下采样 (8x8 -> 4x4)
        self.stage2 = nn.Sequential(
            *[ConvBlock(dim=dims[1], drop_path=drop_path_rate) for _ in range(depths[1])],
            Downsample(dim=dims[1])  # 128 -> 256
        )
        
        # Stage 3: 1次下采样 (4x4 -> 2x2)
        self.stage3 = MambaVisionLayer(
            dim=dims[2], depth=depths[2], window_size=window_sizes[2],
            downsample=True, mlp_ratio=mlp_ratio, drop=drop_rate,  # 需要下采样
            drop_path=drop_path_rate, layer_scale=layer_scale
        )  # 256 -> 512
        
        # Stage 4: 1次下采样 (2x2 -> 1x1)
        self.stage4 = MambaVisionLayer(
            dim=dims[3], depth=depths[3], window_size=window_sizes[3],
            downsample=True, mlp_ratio=mlp_ratio, drop=drop_rate,  # 需要下采样！
            drop_path=drop_path_rate, layer_scale=layer_scale
        )  # 512 -> 1024
        
        # Head
        self.norm = nn.BatchNorm2d(dims[4])  # 1024
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[4], num_classes)  # 1024 -> num_classes
    
    def forward_features(self, x):
        #print(f"Input to features: {x.shape}")
    
        x = self.stem(x)
        #print(f"After stem: {x.shape}")
    
        x = self.stage1(x)
        #print(f"After stage1: {x.shape}")
    
        x = self.stage2(x)
        #print(f"After stage2: {x.shape}")
    
        x = self.stage3(x)
        #print(f"After stage3: {x.shape}")
    
        x = self.stage4(x)
        #print(f"After stage4: {x.shape}")
    
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #print(f"Final features: {x.shape}")
    
        return x
    
    def forward(self, x):
        #print(f"MambaVision input: {x.shape}")
    
        # 保存原始形状
        original_shape = x.shape
    
        # 处理4维输入: (B, C, H, W)
        if x.dim() == 4:
            B, C, H, W = x.shape
            #print(f"4D input - B: {B}, C: {C}, H: {H}, W: {W}")
        
            # 直接处理
            x = self.forward_features(x)
            x = self.head(x)
        
            # 确保输出是2维: (B, features)
            if x.dim() == 3:
                # 如果是3维 (B, 1, features)，压缩中间维度
                if x.shape[1] == 1:
                    x = x.squeeze(1)  # (B, features)
                else:
                    # 其他情况，取平均值
                    x = x.mean(dim=1)  # (B, features)
    
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")

        #print(f"MambaVision output: {x.shape}")
        return x
