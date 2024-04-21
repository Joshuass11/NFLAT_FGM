# --coding: utf-8-
import os
from functools import partial

from fastNLP import cache_results, Vocabulary, DataSet
from fastNLP.embeddings import StaticEmbedding
from fastNLP.io import ConllLoader

from utils.paths import yangjie_rich_pretrain_unigram_path, yangjie_rich_pretrain_bigram_path, data_filename
from utils.tools import Trie

#是否对文本中的单词进行索引化
# 索引化是将文本分割成单词或子词的过程，并为每个单词或子词分配一个唯一的标识符（索引
#因为在mian里传入resume的变量是str，所以这里是dataset_name
def load_data(dataset_name, index_token=False, char_min_freq=1, bigram_min_freq=1, only_train_min_freq=True,
              char_dropout=0.01, bigram_dropout=0.01, dropout=0, label_type='ALL', refresh_data=False):
        type = label_type if label_type!='ALL' else ''
        cache_name = ('cache/NER_dataset_{}{}'.format(dataset_name, type))
#data_filename是path的字典，dataset_name是main用parser传入的--dataset名,unigram_embedding_path, bigram_embedding_path？
        return load_ner(data_filename[dataset_name], yangjie_rich_pretrain_unigram_path, yangjie_rich_pretrain_bigram_path,
                 index_token, char_min_freq, bigram_min_freq, only_train_min_freq, char_dropout, bigram_dropout, dropout,
                 label_type, _cache_fp=cache_name, _refresh=refresh_data)
#以下的data_path应该是path定义的字典，字符和二元字符的预训练嵌入路径
@cache_results(_cache_fp='cache/datasets', _refresh=False)
def load_ner(data_path, unigram_embedding_path, bigram_embedding_path, index_token, char_min_freq,
                   bigram_min_freq, only_train_min_freq, char_dropout, bigram_dropout, dropout, label_type='ALL'):
    #参数 ['chars', 'target'] 指定了要加载的数据字段，即字符序列和目标标签。
    loader = ConllLoader(['chars', 'target'])
#两个拼起来，以下的data_path应该是path定义的字典
    train_path = os.path.join(data_path['path'], data_path['train'])
    dev_path = os.path.join(data_path['path'], data_path['dev'])
    test_path = os.path.join(data_path['path'], data_path['test'])
#相当于得到了完整路径
    paths = {'train': train_path, 'dev': dev_path, 'test': test_path}

    datasets = {}
#执行datasets[k] = bundle.datasets['train']后，实际上相当于将'dev'和'test'两个键也存入了datasets字典中
    for k, v in paths.items():#这里loader传入了char和target
        bundle = loader.load(v)
        datasets[k] = bundle.datasets['train']
#打印出各个数据集的长度！！！检查一下
    for k, v in datasets.items():
        print('{}:{}'.format(k, len(v)))
#新定义
    vocabs = {}
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()

    for k, v in datasets.items():
        # ignore the word segmentation tag
        # v.apply_field(lambda x: ['<start>'] + [w[0] for w in x], 'chars', 'chars')
        # add s label
        # v.apply_field(lambda x: ['O'] + x, 'target', 'target')
#将不符合条件的标记替换为 'O'。除了weibo外都不用替换本来就是O
        if label_type == 'NE':
            v.apply_field(lambda x: [w if len(w) > 1 and w.split('.')[1] == 'NAM' else 'O' for w in x],
                          'target', 'target')
        if label_type == 'NM':
            v.apply_field(lambda x: [w if len(w) > 1 and w.split('.')[1] == 'NOM' else 'O' for w in x],
                          'target', 'target')

        # datasets将所有digit转为0,这里对chars进行操作，但它是唯一用法,这里的v是datasets里的值，应该就是chars
        v.apply_field(lambda chars:[''.join(['0' if c.isdigit() else c for c in char]) for char in chars],
            field_name='chars', new_field_name='chars')
#datasets得到bigrams
        v.apply_field(get_bigrams, 'chars', 'bigrams')
#from_dataset 方法的作用是根据给定的数据集构建词汇表。通常在自然语言处理任务中，词汇表是指数据集中所有不重复词汇的集合，并且为每个词汇分配一个唯一的索引。，no_create_entry_dataset=[datasets['dev'], datasets['test']]: 这个参数指定了不需要为验证集和测试集创建新的词汇表条目。通常，词汇表只从训练集构建，而验证集和测试集则共享相同的词汇表。
    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']])
  #field_name='target'指定了要从数据集中获取标签的字段名，即数据集中包含标签的字段。对应loader的chars和target，就是标签
    label_vocab.from_dataset(datasets['train'], field_name='target')
    #label_vocab.idx2word: 这部分是标签词汇表对象的属性，用于获取一个从索引到词汇的映射，即将词汇表中的索引映射到对应的词汇。
    print('label_vocab:{}\n{}'.format(len(label_vocab), label_vocab.idx2word))
#用于添加一个新的字段来存储字符序列的长度
    for k, v in datasets.items():
        v.add_seq_len('chars', new_field_name='seq_len')
#往单词表里加入刚刚得到的值
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
#会将训练集中的所有字符 bigram 收集起来，构建一个词汇表 bigram_vocab，以便后续使用。
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    if index_token:#检查是否需要进行索引化，指定要索引化的字段是字符。new_field_name='chars': 指定索引化后的新字段名为 chars。
        char_vocab.index_dataset(*list(datasets.values()), field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(datasets.values()), field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(*list(datasets.values()), field_name='target', new_field_name='target')
#以上的If之后的东西不会影响vocabs里面的内容，datasets字典中的值不会发生变化，它们仍然是原始的数据集对象，只是在这些对象中添加了新的索引化字段。
    vocabs['bigram'] = bigram_vocab

    embeddings = {}
#StaticEmbedding是fastNLP方法，以下两个path都是导入
    if unigram_embedding_path is not None:
        unigram_embedding = StaticEmbedding(char_vocab, model_dir_or_name=unigram_embedding_path,
                                            word_dropout=char_dropout, only_norm_found_vector=True,
                                            min_freq=char_min_freq, only_train_min_freq=only_train_min_freq, dropout=dropout)
        embeddings['char'] = unigram_embedding
    if bigram_embedding_path is not None:#only_norm_found_vector=True：指定仅使用词汇表中找到的字符的向量，并且对这些向量进行归一化
        bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
                                           word_dropout=bigram_dropout, only_norm_found_vector=True,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq, dropout=dropout)
        embeddings['bigram'] = bigram_embedding






#embedding有char和bigrams,vocabs还有标签，它的标签、char、bigrams都有索引，
# datasets有char,bigrams,target还有chars字符序列的长度sq_len
    return datasets, vocabs, embeddings

#返回由相邻字符组成的 bigram 列表,最后加一个分隔，保证bigrams列表和chars长度结构一样
def get_bigrams(chars):
    return [char1 + char2
            for char1, char2 in zip(chars, chars[1:] + ['<eos>'])]


@cache_results(_cache_fp='cache/load_yangjie_rich_pretrain_word_list', _refresh=False)
#embedding_path就是腾讯什么的
def load_yangjie_rich_pretrain_word_list(embedding_path, drop_characters=True):
    f = open(embedding_path, 'r', encoding='utf-8')
    lines = f.readlines()
    w_list = []
    for line in lines:
        splited = line.strip().split(' ')
        w = splited[0]
        w_list.append(w)

    if drop_characters:
        w_list = list(filter(lambda x: len(x) != 1, w_list))
#最后得到一个词典
    return w_list


@cache_results(_cache_fp='need_to_defined_fp', _refresh=True)
def equip_chinese_ner_with_lexicon(datasets, vocabs, embeddings, w_list, word_embedding_path=None,
                                   only_lexicon_in_train=False, word_char_mix_embedding_path=None,
                                   lattice_min_freq=1, only_train_min_freq=True, dropout=0):
    if only_lexicon_in_train:
        print('已支持只加载在trian中出现过的词汇')
#通过 Trie 树获取该字符串在词典中的词语信息
    def get_skip_path(chars, w_trie):#这里就是一个加上[[-1, -1, '<non_word>']]然后获得result，只得到w_list里有的信息
        #将字符序列chars中的字符连接起来，形成一个字符串sentence
        sentence = ''.join(chars)#result 是一个列表，包含了跳跃路径（skip path）中的词语，而 bme_num 是跳跃路径中的词语数量
        result, bme_num = w_trie.get_lexicon(sentence)
        # print(result)
        if len(result) == 0:
            return [[-1, -1, '<non_word>']]
        return [[-1, -1, '<non_word>']] + result

    # def get_bme_num(chars, w_trie):
    #     sentence = ''.join(chars)
    #     result, bme_num = w_trie.get_lexicon(sentence)
    #     # print(result)
    #     return bme_num

    a = DataSet()
    # a.apply
    #w_trie得到wlist中的东西，也就是tx词典等里面的单词
    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)
#是否仅在训练集中构建词典
    if only_lexicon_in_train:
        lexicon_in_train = set()
        for s in datasets['train']['chars']:#这里的chars没有被分开，相当于get_lexicon的输入sentence
            #获取datasets中存在于 Trie 树中的词汇信息。然后将词汇信息中的词汇部分取出，并添加到 lexicon_in_train 集合中。
            lexicon_in_s = w_trie.get_lexicon(s)
            for s, e, lexicon in lexicon_in_s:#只有result参加迭代
                #把数据集中词典中有的word放进lexicon_in_train
                lexicon_in_train.add(''.join(lexicon))
#使用的 Trie 树仅包含训练集中的词汇信息，避免了将测试集或开发集中的词汇信息泄露到训练过程中
        print('lexicon in train:{}'.format(len(lexicon_in_train)))
        #打印出前十个
        print('i.e.: {}'.format(list(lexicon_in_train)[:10]))
        #初始化,这里的树里只有w_list，即词典中有，且原数据集中的词语
        w_trie = Trie()
        for w in lexicon_in_train:
            w_trie.insert(w)

    import copy
    #
    def get_max(x):
        max_num = [0, 0, 0]
        for item in x:
            max_num = [a if a > b else b for a, b in zip(max_num, item)]
        return max_num

    # max_num = [0, 0, 0]，这里的k应该是train,dev,test
    for k, v in datasets.items():#处理完后得到二维，子列表三个元素的结果
        v.apply_field(partial(get_skip_path, w_trie=w_trie), 'chars', 'lexicons')
        # v.apply_field(partial(get_bme_num,w_trie=w_trie),'chars','bme_num')
        # for num in v.apply_field(get_max, field_name='bme_num'):
        #     max_num = [a if a > b else b for a, b in zip(max_num, num)]
        v.apply_field(copy.copy, 'chars', 'raw_chars')
        #datasets增加新字段，之前加入的是seq_len
        v.add_seq_len('lexicons', 'lex_num')
        v.apply_field(lambda x: list(map(lambda y: y[0], x)), 'lexicons', 'lex_s')
        v.apply_field(lambda x: list(map(lambda y: y[1], x)), 'lexicons', 'lex_e')

    # print('max', max_num)
    #
    # def get_bme_feat(x, max_num=max_num):
    #     max_s = max(max_num)
    #     return [[1 if s == i else 0 for i in range(max_s+1)] for s in x]
    #
    # for k,v in datasets.items():
    #     v.apply_field(get_bme_feat,field_name='bme_num')
#chars 是一个字符序列的列表，lexicons 是一个包含词汇信息的列表，其中每个词汇信息由三个元素构成（起始位置、结束位置和词汇本身
    #map(lambda x: x[2], lexicons) 将 lexicons 列表中每个词汇信息的第三个元素提取出来，形成一个新的列表
    #chars + list(map(lambda x: x[2], lexicons)) 将两个列表连接起来，得到一个包含字符序列和词汇的词汇序列的列表。
    def concat(ins):
        chars = ins['chars']
        lexicons = ins['lexicons']
        result = chars + list(map(lambda x: x[2], lexicons))
        # print('lexicons:{}'.format(lexicons))
        # print('lex_only:{}'.format(list(filter(lambda x:x[2],lexicons))))
        # print('result:{}'.format(result))
        return result

    def get_pos_s(ins):
        # lex_s = ins['lex_s']
        seq_len = ins['seq_len']
        pos_s = list(range(seq_len))# + lex_s

        return pos_s

    def get_pos_e(ins):
        # lex_e = ins['lex_e']
        seq_len = ins['seq_len']
        pos_e = list(range(seq_len))# + lex_e

        return pos_e

    # def norm_bme(x, max_num):
    #     for i in range(len(x)):
    #         x[i] = [(2 * b - a) / a for a, b in zip(max_num, x[i])]
    #     return x
    # def get_word_label(ins):
    #     string = "".join(ins['entities'])
    #     entities = ins['entities']
    #     label = []
    #     for word in ins['raw_words']:
    #         if word in entities:
    #             label.append([0, 1])
    #         elif word not in entities and sum([1 if c in string else 0 for c in word]):
    #             label.append([1, 0])
    #         else:
    #             label.append([0, 0])
    #     return [[0, 0] for i in range(ins['seq_len'])] + label

    for k, v in datasets.items():
        #把lexicon每个元素第2个元素拿出来，就是单词本身,这里的rawwords是不是真的词语，是类似bigrams的
        v.apply_field(lambda x: [m[2] for m in x], field_name='lexicons', new_field_name='raw_words')
        #lattice就是char和lexicon结合的结果
        v.apply(concat, new_field_name='lattice')
        # v.set_input('lattice')
        #下面两个是一样的
        v.apply(get_pos_s, new_field_name='pos_s')
        v.apply(get_pos_e, new_field_name='pos_e')
        # v.apply_field(partial(norm_bme, max_num=max_num),  field_name='bme_num', new_field_name='bme_num')
        # v.set_input('pos_s', 'pos_e')

        # v.apply(get_word_label, new_field_name='word_label')

    word_vocab = Vocabulary()
    # word_vocab.add_word_lst(w_list)
    word_vocab.from_dataset(datasets['train'], field_name='raw_words',
                               no_create_entry_dataset=[datasets['dev'], datasets['test']])
    vocabs['word'] = word_vocab

    lattice_vocab = Vocabulary()
    lattice_vocab.from_dataset(datasets['train'], field_name='lattice',
                               no_create_entry_dataset=[datasets['dev'], datasets['test']])
    vocabs['lattice'] = lattice_vocab

    # if word_embedding_path is not None:
    #     word_embedding = StaticEmbedding(word_vocab, word_embedding_path,
    #                                         only_norm_found_vector=True, word_dropout=0, dropout=dropout)
    #     embeddings['word'] = word_embedding
    #如果 model_dir_or_name包含的词汇表中的某些词汇的嵌入向量在 char_vocab中找不到对应的字符索引，而且
   # only_norm_found_vector参数被设置为True，那么StaticEmbedding类会忽略这些未找到的嵌入向量，只使用在char_vocab
    #中找到的字符的向量，并对这些向量进行归一化。
    if word_char_mix_embedding_path is not None:
        lattice_embedding = StaticEmbedding(lattice_vocab, word_char_mix_embedding_path, word_dropout=0.01,
                                            only_norm_found_vector=True, dropout=dropout,
                                            min_freq=lattice_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['lattice'] = lattice_embedding
    # print(datasets['train'][77]['lattice'])
    vocabs['lattice'].index_dataset(*(datasets.values()),
                                    field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(*(datasets.values()),
                                   field_name='bigrams', new_field_name='bigrams')
    vocabs['label'].index_dataset(*(datasets.values()),
                                  field_name='target', new_field_name='target')
    vocabs['lattice'].index_dataset(*(datasets.values()),
                                    field_name='raw_words', new_field_name='words')

    return datasets, vocabs, embeddings