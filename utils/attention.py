class Attention(nn.Module):
    def __init__(self, word_size:int=512, embed_dim:int=64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dim_K = torch.tensor(embed_dim)
        self.query = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.key  = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.value = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)

    def self_attention(self, Q: Tensor, K: Tensor, V: Tensor,
                       mask:Optional[BoolTensor]=None) -> Tensor:
        K_T = torch.transpose(K, 0, 1)
        score = torch.matmul(Q, K_T)                # Matmul
        score /= torch.sqrt(self.dim_K)             # Scale
        if mask is not None:
            # 如果存在mask，则将得分矩阵中mask值为0的位置，用负无穷（-inf）来填充。这样在softmax操作后，这些位置对应的注意力权重就会变为0，相当于将这些位置的信息忽略掉。
            score = torch.masked_fill(score, mask==0, -torch.inf)
        score = torch.softmax(score, dim=-1)        # SoftMax
        Z = torch.matmul(score, V)    # Matmul
        # Z = F.scaled_dot_product_attention(Q, K, V),优化过的代码，可以尝试
        return Z
class  MultiQueryAttention(Attention):
    r"""
    https://arxiv.org/pdf/1911.02150.pdf
    """
    def __init__(self, word_size: int = 512, embed_dim: int = 64, n_query:int=8) -> None:
        super().__init__(word_size, embed_dim)
        self.n_query = n_query
        #这个需要加在最后调整维度
        self.proj = nn.Linear(in_features=embed_dim * n_query,
                              out_features=embed_dim, bias=False)
        delattr(self, 'query')
        #这个需要写一下
        self.querys = nn.ModuleList([
            nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
            for _ in range(n_query)
        ])
        self.key = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.value = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)

    def forward(self, x: Tensor, mask:Optional[BoolTensor]=None) -> Tensor:
        K = self.key(x)
        V = self.value(x)
        Z_s = torch.cat([
            self.self_attention(query(x), K, V, mask) for query in self.querys
        ], dim=1)
        Z = self.proj(Z_s)
        return Z
