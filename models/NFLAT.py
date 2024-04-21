#以下两个引入都是类
from modules.infomer import InterFormerEncoder
from modules.transformer import TransformerEncoder

from torch import nn
import torch
import torch.nn.functional as F
#创建并初始化一个条件随机场（Conditional Random Field，CRF）层
from modules.utils import get_crf_zero_init
from modules.attack import FGM

#模型参数,空白就是填入padding，最长序列长度就是max_seq_len
#init是初始化，定义基本的模型和结构
class NFLAT(nn.Module):
    def __init__(self, tag_vocab, char_embed, word_embed, num_layers, hidden_size, n_head,head_dims, feedforward_dim, dropout,
                 max_seq_len, after_norm=True, attn_type='adatrans',  bi_embed=None, softmax_axis=-2,
                 char_dropout=0.5, word_dropout=0.5, fc_dropout=0.3, pos_embed=None, attn_dropout=0, four_pos_fusion='ff',
                 q_proj=True, k_proj=True, v_proj=True, r_proj=True, scale=False,
                 vocab=None, before=True, is_less_head=1):
        """

        :param tag_vocab: fastNLP Vocabulary
        :param embed: fastNLP TokenEmbedding
        :param num_layers: number of self-attention layers
        :param d_model: input size
        :param n_head: number of head
        #！！！
        :param feedforward_dim: the dimension of ffn
        :param dropout: dropout in self-attention
        :param after_norm: normalization place
        #！！！
        :param attn_type: adatrans, naive
        :param rel_pos_embed: position embedding的类型，支持sin, fix, None. relative时可为None
        :param bi_embed: Used in Chinese scenerio
        :param fc_dropout: dropout rate before the fc layer
        """
        super().__init__()
        self.vocab = vocab
#以下两个一样都是lattice的embedding
        self.char_embed = char_embed
        self.word_embed = word_embed#以下的size是50，是设计好的
        char_embed_size = self.char_embed.embed_size
        word_embed_size = self.word_embed.embed_size
        self.bi_embed = None
        if bi_embed is not None:
            self.bi_embed = bi_embed
            #这里的lattice的数量加上bigrams的数量
            char_embed_size += self.bi_embed.embed_size

        self.char_fc = nn.Linear(char_embed_size, hidden_size)
        self.word_fc = nn.Linear(word_embed_size, hidden_size)
        self.char_dropout = nn.Dropout(char_dropout)
        self.word_dropout = nn.Dropout(word_dropout)

        self.n_head = n_head
        self.hidden_size = hidden_size
        self.before = before
#这里is_less_head，模型自带的，看一下传进来多少(1)
        self.chars_transformer = TransformerEncoder(num_layers, hidden_size, n_head//is_less_head, feedforward_dim, dropout,
                                                    after_norm=after_norm, attn_type=attn_type,
                                                    scale=scale, attn_dropout=attn_dropout,
                                                    pos_embed=pos_embed)

        self.informer = InterFormerEncoder(num_layers, hidden_size, n_head, max_seq_len, feedforward_dim, softmax_axis=softmax_axis,
                                       four_pos_fusion=four_pos_fusion, fc_dropout=dropout, attn_dropout=attn_dropout,
                                       q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, r_proj=r_proj, scale=scale, vocab=vocab)


        self.fc_dropout = nn.Dropout(fc_dropout)
        self.out_fc = nn.Linear(hidden_size, len(tag_vocab))
        self.crf = get_crf_zero_init(len(tag_vocab))
        self.times = 0
        self.fgm = FGM(self)
#以下的forward函数包括了interformer和transformer
    def _forward(self, chars, target, words, pos_s, pos_e, lex_s, lex_e, seq_len, lex_num, bigrams=None):
        #以下三个变量都是新定义
        char_ids = chars
        word_ids = words
        target_ids = target

        vocab = self.vocab

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        target = target.to(device)
        target_new = torch.where(target == 2, torch.tensor(0, device=device), target)

        target_new = torch.where(target_new != 0, torch.tensor(1, device=device), target_new)
        def get_bigrams(input_string):
            # 使用列表推导式生成 bigrams_new
            return [input_string[i:i + 2] for i in range(len(input_string) - 1)]
        def get_trigrams(input_string):
            # 使用列表推导式生成 trigrams
            return [input_string[i:i + 3] for i in range(len(input_string) - 2)]
        bi_tri_gram_set = set()
        current_sentence = ''
        for char_list, target_label_list in zip(char_ids, target_new):
            for char, target_label in zip(char_list, target_label_list):
                char = char.item()
                target_label = target_label.item()
                if target_label == 1:
                    char = vocab.to_word(char)  # 假设 to_word 方法用于根据索引获取词语
                    current_sentence += char
                elif current_sentence:
                    bigrams_new = get_bigrams(current_sentence)
                    trigrams = get_trigrams(current_sentence)
                    bi_tri_gram_set.update(bigrams_new)
                    bi_tri_gram_set.update(trigrams)
        # 根据vocab_words找出所有bigram_set的索引，放进words_index_set集合里
        words_index_set = {vocab.to_index(bigram) for bigram in bi_tri_gram_set if bigram in vocab}
        words_tensor = torch.tensor(
            [[int(word_index.item() in words_index_set) for word_index in row] for row in word_ids])
        #!!这里修改
        def process_word(word):
            char_indices = [vocab.to_index(char) for char in word]
            return torch.tensor(char_indices)

        matrix_list = []
        char_tensor_list = []
        for batch_idx in range(len(words)):
            # 如果对应位置是0，则将chars全部标为0
            char_tensor = torch.zeros_like(chars[batch_idx])

            for word_idx, is_entity in enumerate(words_tensor[batch_idx]):
                if is_entity.item() == 1:
                    word = vocab.to_word(words[batch_idx][word_idx].item())
                    char_indices = process_word(word)

                    # 找到chars张量中对应的字符索引并标记为1
                    for i in range(len(chars[batch_idx])):
                        char_indices = char_indices.to(device)
                        chars[batch_idx] = chars[batch_idx].to(device)
                        if (char_indices == chars[batch_idx][i:i + len(char_indices)]).all():
                            char_tensor[i:i + len(char_indices)] = 1
                            char_tensor[:i] = 0
                            char_tensor[i + len(char_indices):] = 0
                            break
                char_tensor_list.append(char_tensor)
                char_tensor = torch.zeros_like(chars[batch_idx])
            matrix = torch.stack(char_tensor_list, dim=1)
            char_tensor_list = []
            #matrix = torch.transpose(matrix, 0, 1)
            matrix_list.append(matrix)
        matrix_label = torch.stack(matrix_list, dim=0)
        #根据字符序列张量中非零元素的位置生成一个掩码张量 chars_mask，用于标记哪些字符是有效的，哪些是填充的
        chars_mask = chars.ne(0)
        #char_embed是lattice的静态向量投射,投射完最后一个维度50
        chars = self.char_embed(chars)
        if self.bi_embed is not None:
            #lattice的静态向量投射
            bigrams = self.bi_embed(bigrams)
            #拼起来(这里没有words)拼完后变成2,68,100
            chars = torch.cat([chars, bigrams], dim=-1)
            #dropout意思是有多少比率的数据是每次回传不更新的
        chars = self.char_dropout(chars)
        # fc改变维度，最后一个维度与hidden_size一样
        chars = self.char_fc(chars)
        # word_embed是lattice的静态向量投射,投射完最后一个维度50
        words = self.word_embed(words)
        words = self.word_dropout(words)
        words = self.word_fc(words)

        target_new = self.char_embed(target_new)
        target_new = self.char_dropout(target_new)
        target_new = self.word_fc(target_new)

        if self.before:
            chars.masked_fill_(~(chars_mask.unsqueeze(-1)), 0)
            #把词表向量经过transformer运算
            chars = self.chars_transformer(chars, chars_mask)
##这时候穿进去的chars和words和lex_s等都是两个一组的，但是chars和words被投射成三维（最后一维是hidden_,别的是2维
        chars,loss_new = self.informer(matrix_label,chars, words, pos_s, pos_e, lex_s, lex_e, seq_len, lex_num, char_ids, word_ids)
        # 将字符级别嵌入填充空白，然后传递给字符 Transformer 模块 chars_transformer 进行特征提取。
        if not self.before:
            chars.masked_fill_(~(chars_mask.unsqueeze(-1)), 0)
            chars = self.chars_transformer(chars, chars_mask)

        self.fc_dropout(chars)
        chars = self.out_fc(chars)
#这一行对输出进行 softmax 操作得到预测结果的 logit
        logits = F.log_softmax(chars, dim=-1)
#如果不处于训练状态，则使用 CRF 模块进行维特比解码并返回预测结果。如果处于训练状态，则计算 CRF 损失并返回损失值
        if not self.training:
            paths, _ = self.crf.viterbi_decode(logits, chars_mask)
            return {'pred': paths}
        else:
            loss = self.crf(logits, target, chars_mask)+loss_new
            return loss
#定义了一个模型的前向传播函数 _forward，以及一个包装了 _forward 的 forward 方法
    def forward(self, chars, target, words, pos_s, pos_e, lex_s, lex_e, seq_len, lex_num, bigrams=None):
        loss = self._forward(chars, target, words, pos_s, pos_e, lex_s, lex_e, seq_len, lex_num, bigrams)
        loss_mean = torch.mean(loss)
        #loss_mean.backward()
        #loss_mean.backward(retain_gragh=True)
        # 在此也可以使用pgd方式进行对抗学习扰动
        self.fgm.attack(epsilon=1.0)  # 在这个地方可以调整扰动的数值概率
        #logits = self.module(**inputs)
        attack_loss = self._forward(chars, target, words, pos_s, pos_e, lex_s, lex_e, seq_len, lex_num, bigrams)
        #attack_loss.backward()
        avg_tensor = loss+attack_loss*0.1
        self.fgm.restore()
        #self._on_backward_record(loss, **kwargs)
        return  {'loss': avg_tensor}
#与forward方法类似，但不接受目标标签作为输入。（预测无监督）
    def predict(self, chars, target, words, pos_s, pos_e, lex_s, lex_e, seq_len, lex_num, bigrams=None):
        return self._forward(chars, target, words, pos_s, pos_e, lex_s, lex_e, seq_len, lex_num, bigrams=bigrams)