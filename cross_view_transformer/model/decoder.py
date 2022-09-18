import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]

def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width) #生成0，1之间的等间距点，个数为width
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w ，生成网格
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w ，padding
    indices = indices[None]                                                 # 1 3 h w

    return indices

class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int = 1,
        bev_height: int = 200,
        bev_width: int = 200,
        h_meters: int = 100,
        w_meters: int = 100,
        offset: int = 0,
        decoder_blocks: list = [128, 128, 64],
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        # decoder_blocks [128,128,64],bev_height==bev_width==200
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))
        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
        # V_inv = invmat(torch.FloatTensor(V))                                    # 3 3
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w) @为矩阵乘法，相当于numpy中matmul
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)                    # 3 h w 在内存中定义一个常量，供使用
        self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))    # d h w dim=128

    def get_prior(self):
        return self.learned_features #此处的learned_features应该为query中的c

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        _, _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim，此处作用是将multi-heads的heads合并到b维度，作为“个数”
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras，attention = softmax(Q @ K^t)V
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k) #q与k的转置相乘
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # # Optional skip connection
        # if skip is not None:
        #     z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z

class CrossViewAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        skip: bool = True,
    ):
        super().__init__()



        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
        self,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        output_query: torch.FloatTensor,
        E_inv: torch.FloatTensor,
    ):
        """
        x: (b, n, d, h, w) encoder的输出,key&value
        output_query: (b, d, H, W)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
        b, n, _, _= E_inv.shape

        # pixel = self.image_plane                                                # b n 3 h w
        # _, _, _, h, w = pixel.shape

        #此处的c应该是外参中的t，代表相机的位置（x,y,z,1）
        c = E_inv[..., -1:]                                                     # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1


        world = bev.grid[:2]                                                    # 2 H W
        w_embed = self.bev_embed(world[None])                                   # 1 d H W
        bev_embed = w_embed - c_embed                                           # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)      # b n d H W



        # Expand + refine the BEV embedding
        output_query = output_query
        query = query_pos + output_query[:, None]                    # b n d H W
        key = x             # b n d h w
        val = x            # b n d h w

        return self.cross_attend(query, key, val, skip=query if self.skip else None)

class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor,):
        super().__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up

        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, dim, blocks, residual=True, factor=2,):
        super().__init__()

        layers = list()
        channels = dim
        for out_channels in blocks:
            layer = DecoderBlock(channels, out_channels, dim, residual, factor)
            layers.append(layer)

            channels = out_channels
        cross_attens = list()
        cva = CrossViewAttention(dim, True, 4 , 32, True)
        cross_attens.append(cva)
        self.cross_attens = nn.ModuleList(cross_attens)
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels
        self.bev_embedding = BEVEmbedding(dim)
    def forward(self, x, E_inv):
        b, _, _, _, _ = x.shape
        output_query = self.bev_embedding.get_prior()                         # d H W
        output_query = repeat(output_query, '... -> b ...', b=b)              # b d H W
        for cross_atten in zip(self.cross_attens):
            x = cross_atten[0](x, self.bev_embedding, output_query, E_inv)

        # cvt decoder
        y = x

        for layer in self.layers:
            y = layer(y, x)

        return y
