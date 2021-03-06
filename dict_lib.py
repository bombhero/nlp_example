# -- coding: utf-8 --
import os
import pandas as pd
import jieba


dict_path = 'c:/tmp/dict.csv'
line_limit = 0


def analyze_corpus_word(file_path, dict_df):
    line_count = 0
    fid = open(file_path, 'rb')
    while True:
        line = fid.readline()
        if not line:
            break
        if len(line.strip()) == 0:
            continue
        line_count += 1
        if (line_count >= line_limit) and (line_limit > 0):
            break
        word_list = jieba.lcut(line)
        for word in word_list:
            if len(word.strip()) == 0:
                continue
            if len(dict_df[dict_df['word'] == word]) == 0:
                dict_df = dict_df.append({'word': word, 'count': 1, 'hot': 0}, ignore_index=True)
                print('Add {}, dict size: {}, total: {}'.format(word, len(dict_df), line_count))
            else:
                dict_df.loc[dict_df['word'].isin([word]), 'count'] += 1
    fid.close()
    return dict_df


def analyze_corpus_letter(file_path, dict_df):
    line_count = 0
    fid = open(file_path, encoding='utf-8')
    while True:
        line = fid.readline()
        if not line:
            break
        line = line.strip()
        if len(line) == 0:
            continue
        line_count += 1
        if (line_count >= line_limit) and (line_limit > 0):
            break
        for idx in range(len(line)):
            if len(dict_df[dict_df['word'] == line[idx]]) == 0:
                dict_df = dict_df.append({'word': line[idx], 'count': 1, 'hot': 0}, ignore_index=True)
                print('Add %s, dict size: %d, total: %d' % (line[idx], len(dict_df), line_count))
            else:
                dict_df.loc[dict_df['word'].isin([line[idx]]), 'count'] += 1
    fid.close()
    return dict_df


def create_word_dict_cn(corpus_path, dict_file=None, split_by_word=False, max_word=10000):
    """
    Return dictionary dataframe.
    Column: word, count
    :param corpus_path:
    :param dict_file:
    :param split_by_word:
    :param max_word:
    :return:
    """
    if os.path.exists(dict_file):
        dict_df = pd.read_csv(dict_file, delimiter=',')
        dict_df = dict_df.append({'word': 'Nan', 'count': 0, 'hot': 0}, ignore_index=True)
        return dict_df
    dict_df = pd.DataFrame(columns=('word', 'count', 'hot'))
    if os.path.isdir(corpus_path):
        file_list = os.listdir(corpus_path)
        for file_name in file_list:
            file_path = corpus_path + '/' + file_name
            if split_by_word:
                df = analyze_corpus_word(file_path, dict_df)
            else:
                df = analyze_corpus_letter(file_path, dict_df)
            dict_df = pd.merge(dict_df, df, how='outer')
    else:
        if split_by_word:
            dict_df = analyze_corpus_word(corpus_path, dict_df)
        else:
            dict_df = analyze_corpus_letter(corpus_path, dict_df)
    dict_df = dict_df.sort_values(by='count', ascending=False)
    for idx in range(len(dict_df)):
        dict_df.iloc[idx]['hot'] = idx + 1
    if dict_file is not None:
        dict_df.to_csv(path_or_buf=dict_file, sep=',')
    return dict_df


class DictWordVec:
    def __init__(self, corpus, dict_file):
        self.dict_df = create_word_dict_cn(corpus, dict_file)
        self.dim_x = len(self.dict_df)

    def trans_word_to_vec(self, word):
        tmp_df = self.dict_df[self.dict_df['word'] == word]
        if len(tmp_df) == 0:
            word_hot = 0
        else:
            word_hot = tmp_df.iloc[0]['hot']
        return word_hot

    def trans_vec_to_word(self, word_vec):
        tmp_df = self.dict_df[self.dict_df['hot'] == word_vec]
        if len(tmp_df) == 0:
            word = ''
        else:
            word = tmp_df.iloc[0]['word']
        return word

    def get_dim_x(self):
        return self.dim_x


if __name__ == '__main__':
    tst_df = create_word_dict_cn(corpus_path='C:/bomb/book/ML/NLP/zenghuanzuang.txt', dict_file='c:/tmp/letter.csv')
    tst_df.to_csv(path_or_buf=dict_path, sep=',')
