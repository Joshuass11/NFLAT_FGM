# --coding: utf-8-
"""
@Time : 2022/3/18 17:56
@Author : 吴双
@File : infomer.py
@Software: PyCharm
"""
import math
import random

import torch
from torch import nn
import torch.nn.functional as F

from modules.rel_pos_embedding import Four_Pos_Fusion_Embedding
from modules.utils import get_embedding, char_lex_len_to_mask

#four_pos_fusion: 四个位置融合的方式，默认为 'ff'。
class InterFormerEncoder(nn.Module):#vocab=vocabs['lattice']
    def __init__(self, num_layers, hidden_size, n_head, max_seq_len, feedforward_dim, softmax_axis=-1, four_pos_fusion= 'ff',
                 fc_dropout=0.3, attn_dropout=0, q_proj=False, k_proj=True, v_proj=True, r_proj=True, scale=False, vocab=None):
        super().__init__()
        assert num_layers > 0
        self.num_layers = num_layers
#通过 nn.Parameter(pe, requires_grad=False) 将位置编码张量转换为模型的参数
        # max_seq_len和hidden_size是一路从main函数穿进来的，也就是模型的维度，=d_model = n_heads * head_dims
        self.max_seq_len = max_seq_len
        #返回的矩阵形状为 (num_embeddings, embedding_dim)，其中 num_embeddings 为行数，表示不同位置的编码；embedding_dim 为列数
        pe = get_embedding(512, hidden_size)
        self.pe = nn.Parameter(pe, requires_grad=False)
        self.pe_ss = self.pe
        self.pe_se = self.pe
        self.pe_es = self.pe
        self.pe_ee = self.pe

#self.four_pos_fusion_embedding = Four_Pos_Fusion_Embedding(...): 初始化四个位置融合的嵌入。
        self.four_pos_fusion_embedding = \
            Four_Pos_Fusion_Embedding(self.pe, four_pos_fusion, self.pe_ss, self.pe_se, self.pe_es, self.pe_ee,
                                      self.max_seq_len, hidden_size)

#setattr 方法将编码器的每个层添加为类的属性，属性名为 layer_0, layer_1, ...，并初始化为 InterFormerLayer 类的实例。
        for i in range(self.num_layers):
            setattr(self, 'layer_{}'.format(i), InterFormerLayer(hidden_size, n_head, feedforward_dim, softmax_axis=softmax_axis,
                                                             fc_dropout=fc_dropout, attn_dropout=attn_dropout, q_proj=q_proj,
                                                             k_proj=k_proj, v_proj=v_proj, r_proj=r_proj, scale=scale, vocab=vocab))

#这里传入进来的chars都是向量，但是chars_id是原本的chars
    def forward(self,matrix_label, chars,words, pos_s, pos_e, lex_s, lex_e, seq_len, lex_num, char_ids=None, word_ids=None ):
        rel_pos_embedding = self.four_pos_fusion_embedding(seq_len, lex_num, pos_s, pos_e, lex_s, lex_e)
       #rel_pos_embeding的形状是2,68,46,128
        output = chars
        for i in range(self.num_layers):
            now_layer = getattr(self,'layer_{}'.format(i))
            output,loss_new = now_layer(matrix_label,output, words, seq_len, lex_num, rel_pos_embedding, char_ids, word_ids)

        return output,loss_new


class InterFormerLayer(nn.Module):
    def __init__(self, hidden_size, n_head, feedforward_dim, softmax_axis=-1, fc_dropout=0.3, attn_dropout=0,
                 q_proj=True, k_proj=True, v_proj=True, r_proj=True, scale=False, vocab=None):
        super().__init__()

        self.attn = InterAttention(hidden_size, n_head, attn_dropout=attn_dropout, softmax_axis=softmax_axis,
                 q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, r_proj=r_proj, scale=scale, vocab=vocab)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(nn.Linear(hidden_size, feedforward_dim),
                                 nn.ReLU(),
                                 nn.Dropout(fc_dropout),
                                 nn.Linear(feedforward_dim, hidden_size),
                                 nn.Dropout(fc_dropout))

    def forward(self,matrix_label,chars, words, seq_len, lex_num, rel_pos_embedding, char_ids, word_ids):
       #residual也就是原本的chars向量（2,68,128）
        residual = chars

        output,loss_new= self.attn(matrix_label,chars, words, seq_len, lex_num, rel_pos_embedding, char_ids, word_ids)

        output += residual
        output = self.norm1(output)

        residual = output
        output = self.ffn(output)

        output += residual
        output = self.norm2(output)

        return output,loss_new


class InterAttention(nn.Module):
    def __init__(self, hidden_size, n_head, attn_dropout=0, softmax_axis=-1,
                 q_proj=True, k_proj=True, v_proj=True, r_proj=True, scale=False, vocab=None):
        super().__init__()
        self.hidden_size = hidden_size
        #这里可以改成n_query
        self.n_head = n_head
        #注意这个后面用到,因为hidden_size其实是d_model，所以per_head_size应该是head_dims
        self.per_head_size = hidden_size
        self.vocab = vocab

        self.softmax_axis = softmax_axis
        self.scale = scale

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.r_proj = r_proj
        self.proj = nn.Linear(hidden_size * n_head,
                              hidden_size, bias=False)
#用于计算注意力分数
        self.word_id_proj = nn.Linear(hidden_size, 1)
        #对输入张量的最后一个维度，即 (2, 68, 46, 18) 中的 18 进行线性变换。
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.querys = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(n_head)
        ])
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        if self.r_proj:
            self.w_r = nn.Linear(hidden_size, hidden_size)
#这行代码创建了一个可学习的参数矩阵 u，它是一个形状为 (n_head, per_head_size) 的张量，用于计算注意力分数。
        self.u = nn.Parameter(torch.Tensor(1, self.per_head_size))
        self.v = nn.Parameter(torch.Tensor(1, self.per_head_size))

        self.dropout = nn.Dropout(attn_dropout)

        self.times = 0
    #这时候穿进去的chars和words和lex_s等都是两个一组的，但是chars和words被投射成三维（最后一维是hidden_,别的是2维
    def _forward(self,chars_query, words_key, words_value,rel_pos_embedding, batch, max_char_len, seq_len, lex_num,max_word_len):
        chars_query = chars_query.view(batch, max_char_len, 1, self.per_head_size)
        chars_query = chars_query.transpose(1, 2)
        query_and_u_for_c = chars_query + self.u.unsqueeze(0).unsqueeze(-2)
        A_C = torch.matmul(query_and_u_for_c, words_key)

        rel_pos_embedding = rel_pos_embedding.view(batch, max_char_len, max_word_len, 1, self.per_head_size)
        rel_pos_embedding_for_b = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        query_for_b_and_v_for_d = chars_query.view(batch, 1, max_char_len, 1, self.per_head_size) \
                                  + self.v.view(1, 1, 1, 1, self.per_head_size)
        B_D = torch.matmul(query_for_b_and_v_for_d, rel_pos_embedding_for_b).squeeze(-2)
        attn_score_raw = A_C + B_D
        if self.scale:
            attn_score_raw = attn_score_raw / math.sqrt(self.per_head_size)
        # get attention mask，不改变维度
        mask = char_lex_len_to_mask(seq_len, lex_num).unsqueeze(1)
        att = attn_score_raw.masked_fill(~mask, -1e15)
#着重更改
        att_score = F.softmax(att, dim=self.softmax_axis)
        #用这个attn和正确分类进行对比，正确分类结果为（2,68,46）
        att_score = self.dropout(att_score)
        att_score = att_score.masked_fill(~mask, 0)
        if self.training:
            self.times += 1
#（2,68,46,128）
        value_weighted_sum = torch.matmul(att_score, words_value)
        chars_result = value_weighted_sum.transpose(1, 2).contiguous(). \
            reshape(batch, max_char_len, -1)
#记得把att_score传给下面,这个也可以按照维度做一下缩放。
        return chars_result,A_C#,att_score

    def forward(self, matrix_label, chars, words, seq_len, lex_num, rel_pos_embedding, char_ids, word_ids):
        criterion = nn.BCELoss()
        batch, max_char_len, _ = chars.size()
        _, max_word_len, _ = words.size()
        # 这里传入的chars都是dataset的正常字段向量化
        chars_query = chars
        words_key = words
        words_value = words

        if self.q_proj:#这里应该是false
            chars_query = self.query_proj(chars_query)
        if self.k_proj:
            words_key = self.key_proj(words_key)
        if self.v_proj:
            words_value = self.value_proj(words_value)
        if self.r_proj:#这里应该是false
            rel_pos_embedding = self.w_r(rel_pos_embedding)
        # 以下的chars_query都是经过投射的chars向量2,68,128,然后最后两个维度被分解
        words_key = words_key.view(batch, max_word_len, 1, self.per_head_size)
        words_value = words_value.view(batch, max_word_len, 1, self.per_head_size)
        words_key = words_key.transpose(1, 2).transpose(-1, -2)
        words_value = words_value.transpose(1, 2)
        AC_list = []
        chars_result = torch.empty(0)
        for query in self.querys:
            chars_output, att_score_output = self._forward(query(chars), words_key, words_value,
                                                           query(rel_pos_embedding), batch, max_char_len, seq_len,
                                                           lex_num, max_word_len)
            chars_result = torch.cat((chars_result, chars_output), dim=2)
            AC_list.append(att_score_output)
        A_C = torch.cat(AC_list, dim=1)
        mean_A_C = torch.mean(A_C, dim=1)
        mean_A_C = F.sigmoid(mean_A_C)

        chars_result = self.proj(chars_result)
        loss_new = criterion(mean_A_C.view(-1, 1), matrix_label.float().view(-1, 1))
        if self.times % 1000 == 0:
            # print(att[1,1,:])
            # print(att.size())
            # print(att[1])
            # print(x_chars[1])
            # print(x_words[1])
            # print(mask[1][0])
            # print(att_score[1])

            # hot map
            def attention_visualization(head_i):
                import matplotlib.pyplot as plt
                import seaborn as sns
                # from sklearn.preprocessing import normalize
                index_ = 0
                seq_length = seq_len[index_].item()
                lex_length = lex_num[index_].item()
                print("第%d个头注意力可视化" % head_i)
                data = att_score.cpu().detach().numpy()[index_][head_i][:seq_length, :lex_length].T
                plt.figure(figsize=(seq_length // 4 + 2, lex_length // 4 + 2))
                plt.rcParams['font.sans-serif'] = ['PingFang SC']  # 设置字体为黑体
                if not char_ids is None and not word_ids is None:
                    row_chars = char_ids[index_][:seq_length].cpu().numpy().tolist()
                    row_words = word_ids[index_][:lex_length].cpu().numpy().tolist()
                    row_chars = [self.vocab.to_word(x) for x in row_chars]
                    row_words = [self.vocab.to_word(x) for x in row_words]
                    sns.heatmap(data,
                                xticklabels=row_chars,
                                yticklabels=row_words,
                                cbar_kws={"orientation": "horizontal"})
                    plt.show()
#这个result就是得到的z
        return chars_result, loss_new