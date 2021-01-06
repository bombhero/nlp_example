import os
import pandas as pd
import jieba


dict_path = 'c:/tmp/dict.csv'


def analyze_corpus(file_path, dict_df):
    fid = open(file_path, 'rb')
    while True:
        line = fid.readline()
        if not line:
            break
        if len(line.strip()) == 0:
            continue
        word_list = jieba.lcut(line)
        for word in word_list:
            if len(word.strip()) == 0:
                continue
            if len(dict_df[dict_df['word'] == word]) == 0:
                dict_df = dict_df.append({'word': word, 'count': 1}, ignore_index=True)
            else:
                dict_df.loc[dict_df['word'].isin([word]), 'count'] += 1
            print('dict size: {}'.format(len(dict_df)))
    fid.close()


def create_dict_cn(corpus_path, max_word=10000):
    """
    Return dictionary dataframe.
    Column: word, count
    :param corpus_path:
    :param max_word:
    :return:
    """
    dict_df = pd.DataFrame(columns=('word', 'count'))
    file_list = os.listdir(corpus_path)
    for file_name in file_list:
        file_path = corpus_path + '/' + file_name
        analyze_corpus(file_path, dict_df)
    return dict_df


if __name__ == '__main__':
    df = create_dict_cn(corpus_path='C:/bomb/proj/corpus')
    df.to_csv(path_or_buf=dict_path, sep=',')
