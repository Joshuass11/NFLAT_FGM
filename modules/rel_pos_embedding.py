import torch
from torch import nn
import torch.nn.functional as F

from modules.utils import char_lex_len_to_mask
from modules.transformer import MultiheadPosAttn
#这里的pe_ss,pe_se,pe_es,pe_ee都是一样的pe，也就是pe = get_embedding(512, hidden_size)这个矩阵
class Four_Pos_Fusion_Embedding(nn.Module):
    def __init__(self,pe,four_pos_fusion,pe_ss,pe_se,pe_es,pe_ee,max_seq_len,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len=max_seq_len
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.pe = pe
        self.four_pos_fusion = four_pos_fusion
        if self.four_pos_fusion == 'ff':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size),
                                                    nn.ReLU(inplace=True))
        if self.four_pos_fusion == 'ff_linear':
            self.pos_fusion_forward = nn.Linear(self.hidden_size*4,self.hidden_size)
        #降维
        elif self.four_pos_fusion == 'ff_two':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*2,self.hidden_size),
                                                    nn.ReLU(inplace=True))
        #这里的hidden_size都是只对最后一个维度。
        elif self.four_pos_fusion == 'attn':
            self.w_r = nn.Linear(self.hidden_size,self.hidden_size)
            self.multihead = MultiheadPosAttn(self.hidden_size*4, 16, dropout=0.1, scale=False)
            #这里n_head的位置本来应该是传入的，嫌麻烦没写
            #self.pos_attn_score= MultiheadPosAttn(self.hidden_size*4, 8, dropout=0.1, scale=False)
            self.pos_attn_score = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size * 4),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size * 4, 4),
                                                nn.Softmax(dim=-1))

            # print('暂时不支持以attn融合pos信息')
        elif self.four_pos_fusion == 'gate':
            self.w_r = nn.Linear(self.hidden_size,self.hidden_size)
            self.pos_gate_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*2),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*2,4*self.hidden_size))

            # print('暂时不支持以gate融合pos信息')
            # exit(1208)
    def forward(self, seq_len, lex_num, pos_s, pos_e, lex_s=None, lex_e=None):
        if lex_s is None or lex_s is None:
            lex_s = pos_s
            lex_e = pos_e


        mask = char_lex_len_to_mask(seq_len, lex_num)

        #这里的seq_len已经是之前的seq_len+lex_num了
        #下面应该都变成3维,例如（2,68,46）
        pos_ss = pos_s.unsqueeze(-1)-lex_s.unsqueeze(-2)
        pos_se = pos_s.unsqueeze(-1)-lex_e.unsqueeze(-2)
        pos_es = pos_e.unsqueeze(-1)-lex_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1)-lex_e.unsqueeze(-2)
        #以下两个没有用到不用管
        pos_ss_reverse = lex_s.unsqueeze(-1)-pos_s.unsqueeze(-2)
        pos_ee_reverse = lex_e.unsqueeze(-1)-pos_e.unsqueeze(-2)
#将输入的位置索引张量与预先定义的位置编码矩阵进行了索引和重塑操作,表示将 pos_ss 张量中对应掩码中为 False 的位置的值替换为 510。
        pos_ss = pos_ss.masked_fill(~mask, 510)
        pos_ee = pos_ee.masked_fill(~mask, 510)
        #将 pos_ss 和 pos_ee 张量的第三维的所有元素的第一个位置（索引为0）设置为0
        pos_ss[:, :, 0] = 0
        pos_ee[:, :, 0] = 0
#batch_size是这里吗，一般都是2，
        batch, max_char_len = pos_s.size()
        _, max_word_len = lex_s.size()

        # if self.mode['debug']:
        #     print('pos_s:{}'.format(pos_s))
        #     print('pos_e:{}'.format(pos_e))
        #     print('pos_ss:{}'.format(pos_ss))
        #     print('pos_se:{}'.format(pos_se))
        #     print('pos_es:{}'.format(pos_es))
        #     print('pos_ee:{}'.format(pos_ee))
        # B prepare relative position encoding
        max_seq_len = pos_s.size(1)
        # rel_distance = self.seq_len_to_rel_distance(max_seq_len)

        # rel_distance_flat = rel_distance.view(-1)
        # rel_pos_embedding_flat = self.pe[rel_distance_flat+self.max_seq_len]
        # rel_pos_embedding = rel_pos_embedding_flat.view(size=[max_seq_len,max_seq_len,self.hidden_size])
        #.view(-1) 的作用是将 pos_ss 展平为一维张量，形状为 (2 * 68 * 46) = 6264,把pe_ss的第一个维度变成6264
        pe_ss_1 = self.pe_ss[(pos_ss).view(-1)]
        pe_ss = self.pe_ss[(pos_ss).view(-1)].view(size=[batch, max_char_len, max_word_len,-1])
        pe_se = self.pe_se[(pos_se).view(-1)].view(size=[batch, max_char_len, max_word_len, -1])
        pe_es = self.pe_es[(pos_es).view(-1)].view(size=[batch, max_char_len, max_word_len, -1])
        #转换成了四维张量，其中最后一个维度的大小由 -1 表示，表示根据其他维度的大小来自动计算，以保持总的元素数量不变
        pe_ee = self.pe_ee[(pos_ee).view(-1)].view(size=[batch, max_char_len, max_word_len, -1])
        pe_ss_reverse = self.pe_ee[(pos_ss_reverse).view(-1)].view(
            size=[batch, max_char_len, max_word_len, -1])
        pe_ee_reverse = self.pe_ee[(pos_ee_reverse).view(-1)].view(
            size=[batch, max_char_len, max_word_len, -1])

        # print('pe_ss:{}'.format(pe_ss.size()))

        if self.four_pos_fusion == 'ff':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee],dim=-1)
            rel_pos_embedding = self.pos_fusion_forward(pe_4)
        if self.four_pos_fusion == 'ff_linear':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee],dim=-1)
            rel_pos_embedding = self.pos_fusion_forward(pe_4)
        if self.four_pos_fusion == 'ff_two':
            pe_2 = torch.cat([pe_ss,pe_ee],dim=-1)
            rel_pos_embedding = self.pos_fusion_forward(pe_2)
        elif self.four_pos_fusion == 'attn':#-1是最后一个维度
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee],dim=-1)
            attn_score_0 = self.multihead(pe_4)#不会改变pe_4的维度
            attn_score = self.pos_attn_score(attn_score_0)#最后一个维度变成4
            pe_4_unflat = self.w_r(pe_4.view(batch,max_seq_len,max_word_len,4,self.hidden_size))
            pe_4_fusion = (attn_score.unsqueeze(-1) * pe_4_unflat).sum(dim=-2)
            rel_pos_embedding = pe_4_fusion#最后维度(batch_size, max_char_len, max_word_len, hidden_size)
            # if self.mode['debug']:
            #     print('pe_4照理说应该是 Batch * SeqLen * SeqLen * HiddenSize')
            #     print(pe_4_fusion.size())

        elif self.four_pos_fusion == 'gate':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            gate_score = self.pos_gate_score(pe_4).view(batch,max_seq_len,max_word_len,4,self.hidden_size)
            gate_score = F.softmax(gate_score,dim=-2)
            pe_4_unflat = self.w_r(pe_4.view(batch, max_seq_len, max_word_len, 4, self.hidden_size))
            pe_4_fusion = (gate_score * pe_4_unflat).sum(dim=-2)
            rel_pos_embedding = pe_4_fusion


        return rel_pos_embedding

