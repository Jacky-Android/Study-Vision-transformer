# transformer研习

## 所有attention放vit_attention文件夹下，及插及用。

### 已经实现

[vsion transformer attention](https://github.com/Jacky-Android/Study-Vision-transformer/blob/main/vit_attention/Vit.py)

# Q,K,V

Transformer里面的Q，K，V是指查询（Query），键（Key）和值（Value）三个矩阵，它们都是通过对输入进行线性变换得到的。它们的作用是实现一种注意力机制（Attention），用于计算输入的每个元素（token）之间的相关性，并根据相关性对输入进行加权和，得到一个新的输出。

具体来说，查询矩阵Q用于询问键矩阵K中的哪个token与查询最相似，通过点积计算得到一个相似度序列。然后对相似度序列进行归一化处理，得到一个概率分布，表示每个token被注意的程度。最后，用这个概率分布对值矩阵V进行加权和，得到一个新的token，表示查询的结果。

这种注意力机制可以让模型学习到输入的每个元素之间的依赖关系，从而提高模型的表达能力和性能。Transformer模型中使用了两种不同的注意力机制，分别是自注意力（Self-Attention）和编码器-解码器注意力（Encoder-Decoder Attention）。

自注意力是指Q，K，V都来自于同一个输入，用于计算输入的每个元素与自身的相关性，从而捕捉输入的内部结构。编码器-解码器注意力是指Q来自于解码器的输出，K，V来自于编码器的输出，用于计算解码器的输出与编码器的输出的相关性，从而捕捉输入和输出之间的对应关系。

![Untitled](transformer%E7%A0%94%E4%B9%A0%20402fabb8d72a44e8a65f5977ac393f14/Untitled.png)

## QKV Attention

- 首先，将输入的特征向量（例如词向量）分别乘以三个不同的可学习的权重矩阵，得到查询矩阵（query matrix）Q，键矩阵（key matrix）K和值矩阵（value matrix）V。这相当于对输入的特征向量进行了三种不同的线性变换，得到了三种不同的表示。
- 然后，计算Q和K的点积，得到注意力的对数值（logit），并乘以一个缩放因子（scale），用于调节注意力的权重。缩放因子通常是K的维度的平方根的倒数，用于避免点积值过大或过小，影响梯度的稳定性。
- 接着，对注意力的对数值进行softmax激活，得到注意力的权重（weight），表示每个输入元素被注意的程度。注意力的权重是一个概率分布，它的和为1。
- 最后，用注意力的权重对V进行加权和，得到注意力的输出（output），表示每个输入元素的新的表示。注意力的输出是一个加权平均的结果，它的维度和V相同。

QKV attention 计算公式可以用数学公式表示为：

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V

$$

其中，dk是K的维度，QKT表示Q和K的转置的点积，softmax函数将注意力的对数值归一化，使得它们的和为1。

# Say something

实现一些经典的vit attention ，也算是笔记。😁😁😁

# Vision transformer

![Untitled](transformer%E7%A0%94%E4%B9%A0%20402fabb8d72a44e8a65f5977ac393f14/Untitled%201.png)

1) patch embedding：例如输入图片大小为224x224，将图片分为固定大小的patch，patch大小为16x16，则每张图像会生成224x224/16x16=196个patch，即输入序列长度为**196**，每个patch维度16x16x3=**768**，线性投射层的维度为768xN (N=768)，因此输入通过线性投射层之后的维度依然为196x768，即一共有196个token，每个token的维度是768。这里还需要加上一个特殊字符cls，因此最终的维度是**197x768**。到目前为止，已经通过patch embedding将一个视觉问题转化为了一个seq2seq问题

(2) positional encoding（standard learnable 1D position embeddings）：ViT同样需要加入位置编码，位置编码可以理解为一张表，表一共有N行，N的大小和输入序列长度相同，每一行代表一个向量，向量的维度和输入序列embedding的维度相同（768）。注意位置编码的操作是sum，而不是concat。加入位置编码信息之后，维度依然是**197x768**

(3) LN/multi-head attention/LN：LN输出维度依然是197x768。多头自注意力时，先将输入映射到q，k，v，如果只有一个头，qkv的维度都是197x768，如果有12个头（768/12=64），则qkv的维度是197x64，一共有12组qkv，最后再将12组qkv的输出拼接起来，输出维度是197x768，然后在过一层LN，维度依然是**197x768**

(4) MLP：将维度放大再缩小回去，197x768放大为197x3072，再缩小变为**197x768**

## vision transformer的attention实现

```python
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

# Swin Tranformer

## 框架

![Untitled](transformer%E7%A0%94%E4%B9%A0%20402fabb8d72a44e8a65f5977ac393f14/Untitled%202.png)

- Swin Transformer（上图为 Swin-T，T 为 Tiny）首先通过补丁分割模块（如[ViT）](https://sh-tsang.medium.com/review-vision-transformer-vit-406568603de0)将输入 RGB 图像分割为不重叠的补丁。
- 每个补丁都被视为一个“令牌”，其特征被设置为原始像素 RGB 值的串联。使用**4×4 的 patch 大小，因此每个 patch 的特征维度为 4×4×3=48**。线性嵌入层应用于该原始值特征，将**其投影到任意维度*C***。
- 为了产生**分层表示**，随着网络变得更深，通过补丁合并层来减少标记的数量。第一个补丁合并层**连接每组 2×2 相邻补丁的特征，并在4 *C*维连接特征**上应用线性层。这**将令牌数量减少了 2×2 = 4 的倍数**（2 次分辨率下采样）
- 输出**尺寸**设置为**2 *C**，*分辨率保持为***H* /8× *W* /8**。补丁合并和特征转换的第一个块被表示为**“阶段 2”**。
- 在每个 MSA 模块和每个 MLP 之前应用 LayerNorm (LN) 层，并在**每个[模块](https://sh-tsang.medium.com/review-layer-normalization-ln-6c2ae88bae47)**之后应用**残差连接。**

## ****Shifted Window Based Self-Attention****

### ****Window Based Self-Attention (W-MSA)****

![Untitled](transformer%E7%A0%94%E4%B9%A0%20402fabb8d72a44e8a65f5977ac393f14/Untitled%203.png)

假设每个窗口包含***M* × *M 个*patch**，全局 MSA 模块和基于***h* × *w*个patch图像**的窗口的计算复杂度为：

![Untitled](transformer%E7%A0%94%E4%B9%A0%20402fabb8d72a44e8a65f5977ac393f14/Untitled%204.png)

其中前者与补丁号*hw*成二次方，后者**在*M*固定（默认设置为 7）**时呈线性。

### ****Window Based Self-Attention (W-MSA)****

- 基于窗口的自注意力模块**缺乏跨窗口的连接**，这限制了它的建模能力。
- 提出了一种移位窗口分区方法，该方法**在连续 Swin Transformer 块中的两个分区配置之间交替**。

![Untitled](transformer%E7%A0%94%E4%B9%A0%20402fabb8d72a44e8a65f5977ac393f14/Untitled%205.png)

• 其中*zl* -1 是前一层的输出特征。

 在计算相似性时，每个头都包含**相对位置偏差*B。***

![Untitled](transformer%E7%A0%94%E4%B9%A0%20402fabb8d72a44e8a65f5977ac393f14/Untitled%206.png)

## 细节

### window_partition

我复现的代码如下

- 首先，将输入的图像张量x（B, C,H, W）按照窗口大小window_size划分为（B, C,H // window_size, window_size, W // window_size, window_size）的形状，其中B是批量大小，H是图像高度，W是图像宽度，C是通道数。
- 然后，将x在第二维和第四维上进行交换，得到（B, C, W // window_size, H // window_size, window_size, window_size）的形状，这样可以保证每个窗口内部的元素是连续存储的。
- 最后，将x展平为（num_windows * B, window_size, window_size, C）的形状，其中***num_windows = (H // window_size) * (W // window_size)***是每张图像划分出的窗口数量。

输入feature map为 [1,128,32,32] ，输出形状为 [16,64,128] 的tokens

```python
def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
     # [batch_size*num_windows, Mh*Mw, total_embed_dim]
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm

```

### windows attention

参照官方代码：

这段代码是定义一个基于窗口的多头自注意力模块的类，它可以实现shifted和non-shifted两种窗口划分方式，并且加入了相对位置偏置。它的主要功能和参数如下：

- **`__init__`**方法是初始化模块的参数，包括输入通道数dim，窗口大小window_size，注意力头数num_heads，以及一些可选的参数，如qkv_bias，qk_scale，attn_drop和proj_drop。这些参数分别控制了是否给查询、键、值添加可学习的偏置，查询和键的缩放因子，注意力权重的dropout比例，以及输出的dropout比例。此外，这个方法还定义了一个相对位置偏置表relative_position_bias_table，它是一个可学习的张量，用来存储每个窗口内部每对位置之间的相对位置偏置。它的形状是(2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)，表示每个头有(2 * window_size[0] - 1) * (2 * window_size[1] - 1)种不同的相对位置关系。这个方法还注册了一个缓冲区relative_position_index，它是一个固定的张量，用来记录每个窗口内部每对位置之间的相对位置索引。它的形状是window_size[0] * window_size[1], window_size[0] * window_size[1]），表示每个窗口有window_size[0] * window_size[1]个位置，每个位置与其他位置之间有一个相对位置索引，范围是0到(2 * window_size[0] - 1) * (2 * window_size[1] - 1) - 1。
- **`forward`**方法是执行模块的前向计算，它接受两个参数：x和mask。x是输入特征，它的形状是(num_windows*B, N, C)，其中num_windows是每张图像划分出的窗口数量，B是批量大小，N是每个窗口内部的位置数量（等于window_size[0] * window_size[1]），C是通道数。mask是一个可选的参数，它是一个掩码张量，用来屏蔽一些不需要计算注意力的位置。它的形状是(num_windows, N, N)，其中num_windows和N与x相同。如果没有提供mask，则默认为None。这个方法的主要步骤如下：
    - 首先，调用self.qkv(x)得到查询、键、值三个张量，并将它们重塑为(B_, N, 3, self.num_heads, C // self.num_heads)的形状，并在第一维和第三维上进行交换，得到(3, B_, self.num_heads, N, C // self.num_heads)的形状。然后将这个张量分解为q, k, v三个张量，分别表示查询、键、值。
    - 然后，将q乘以self.scale得到缩放后的查询，并与k进行转置矩阵乘法得到注意力得分张量attn。它的形状是(B_, self.num_heads, N, N)，表示每个头每个窗口内部每对位置之间的注意力得分。
    - 接着，根据self.relative_position_index从self.relative_position_bias_table中取出相应的相对位置偏置，并将其重塑为(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)的形状，并在第一维和第三维上进行交换，得到(-1, self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1])的形状。然后将其在第零维上增加一个维度，并与attn相加得到加入了相对位置偏置的注意力得分张量attn。
    - 然后，如果提供了mask，则将attn重塑为(B_ // nW, nW, self.num_heads, N, N)的形状，并在第一维和第二维上增加一个维度，与mask相加得到屏蔽了一些位置的注意力得分张量attn。然后将其重塑为(-1, self.num_heads, N, N)的形状。如果没有提供mask，则直接将attn通过self.softmax函数得到注意力权重张量attn，并对其进行self.attn_drop操作。
    - 最后，将attn与v进行矩阵乘法得到输出特征x，并将其在第一维和第二维上进行交换，并重塑为(B_, N, C)的形状。然后通过self.proj(x)得到最终的输出特征x，并对其进行self.proj_drop操作。返回x作为模块的输出。

![Untitled](transformer%E7%A0%94%E4%B9%A0%20402fabb8d72a44e8a65f5977ac393f14/Untitled%207.png)

![Untitled](transformer%E7%A0%94%E4%B9%A0%20402fabb8d72a44e8a65f5977ac393f14/Untitled%208.png)

```python
class WindowAttention(nn.Module):
	def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

    super().__init__()
    self.dim = dim
    self.window_size = window_size  # Wh, Ww
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = qk_scale or head_dim ** -0.5

    # define a parameter table of relative position bias
    self.relative_position_bias_table = nn.Parameter(
        torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

    # get pair-wise relative position index for each token inside the window
    coords_h = torch.arange(self.window_size[0])
    coords_w = torch.arange(self.window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += self.window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    self.register_buffer("relative_position_index", relative_position_index)

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)

    trunc_normal_(self.relative_position_bias_table, std=.02)
    self.softmax = nn.Softmax(dim=-1)

def forward(self, x, mask=None):
    """
    Args:
        x: input features with shape of (num_windows*B, N, C)
        mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
    """
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
```

### window_reverse

将一个个window还原成一个feature map

- 首先，根据H_sp和W_sp计算出每张图像划分出的窗口数量，即H * W / H_sp / W_sp，并用它除以B’得到批量大小B。
- 然后，将输入张量img_splits_hw重塑为(B, H // H_sp, W // W_sp, H_sp, W_sp, C)的形状，其中每个维度分别表示批量大小，窗口行数，窗口列数，窗口高度，窗口宽度和通道数。
- 然后，将img在第二维和第四维上进行交换，得到(B, W // W_sp, H // H_sp, H_sp, W_sp, C)的形状，并将其展平为(B, C, H, W)的形状。这样就实现了将每个窗口内部的元素按照原始图像的顺序排列，并合并成一个完整的图像。

```python
def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, img.shape[-1],H,W)
    return img
```

> [图解Swin Transformer - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/367111046)
>