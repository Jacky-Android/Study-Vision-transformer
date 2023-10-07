# transformer研习

# Q,K,V

Transformer里面的Q，K，V是指查询（Query），键（Key）和值（Value）三个矩阵，它们都是通过对输入进行线性变换得到的。它们的作用是实现一种注意力机制（Attention），用于计算输入的每个元素（token）之间的相关性，并根据相关性对输入进行加权和，得到一个新的输出。

具体来说，查询矩阵Q用于询问键矩阵K中的哪个token与查询最相似，通过点积计算得到一个相似度序列。然后对相似度序列进行归一化处理，得到一个概率分布，表示每个token被注意的程度。最后，用这个概率分布对值矩阵V进行加权和，得到一个新的token，表示查询的结果。

这种注意力机制可以让模型学习到输入的每个元素之间的依赖关系，从而提高模型的表达能力和性能。Transformer模型中使用了两种不同的注意力机制，分别是自注意力（Self-Attention）和编码器-解码器注意力（Encoder-Decoder Attention）。

自注意力是指Q，K，V都来自于同一个输入，用于计算输入的每个元素与自身的相关性，从而捕捉输入的内部结构。编码器-解码器注意力是指Q来自于解码器的输出，K，V来自于编码器的输出，用于计算解码器的输出与编码器的输出的相关性，从而捕捉输入和输出之间的对应关系。

![Untitled](https://github.com/Jacky-Android/Study-Vision-transformer/assets/55181594/ed7507ab-98b1-447b-9055-9a90ae2fdd86)


## QKV Attention

- 首先，将输入的特征向量（例如词向量）分别乘以三个不同的可学习的权重矩阵，得到查询矩阵（query matrix）Q，键矩阵（key matrix）K和值矩阵（value matrix）V。这相当于对输入的特征向量进行了三种不同的线性变换，得到了三种不同的表示。
- 然后，计算Q和K的点积，得到注意力的对数值（logit），并乘以一个缩放因子（scale），用于调节注意力的权重。缩放因子通常是K的维度的平方根的倒数，用于避免点积值过大或过小，影响梯度的稳定性。
- 接着，对注意力的对数值进行softmax激活，得到注意力的权重（weight），表示每个输入元素被注意的程度。注意力的权重是一个概率分布，它的和为1。
- 最后，用注意力的权重对V进行加权和，得到注意力的输出（output），表示每个输入元素的新的表示。注意力的输出是一个加权平均的结果，它的维度和V相同。

QKV attention 计算公式可以用数学公式表示为：

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

其中，dk是K的维度，QKT表示Q和K的转置的点积，softmax函数将注意力的对数值归一化，使得它们的和为1。

# Say something

实现一些经典的vit attention ，也算是笔记。😁😁😁

# Vision transformer
paper:[An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/abs/2010.11929)

![Untitled 1](https://github.com/Jacky-Android/Study-Vision-transformer/assets/55181594/93183f78-7a1c-46e9-a4d9-05577ad3fe22)


(1) patch embedding：例如输入图片大小为224x224，将图片分为固定大小的patch，patch大小为16x16，则每张图像会生成224x224/16x16=196个patch，即输入序列长度为**196**，每个patch维度16x16x3=**768**，线性投射层的维度为768xN (N=768)，因此输入通过线性投射层之后的维度依然为196x768，即一共有196个token，每个token的维度是768。这里还需要加上一个特殊字符cls，因此最终的维度是**197x768**。到目前为止，已经通过patch embedding将一个视觉问题转化为了一个seq2seq问题

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

[图解Swin Transformer - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/367111046)
