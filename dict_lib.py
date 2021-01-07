import os
import pandas as pd
import jieba


dict_path = 'c:/tmp/dict.csv'
line_limit = 0


def analyze_corpus(file_path, dict_df):
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
                dict_df = dict_df.append({'word': word, 'count': 1}, ignore_index=True)
                print('Add {}, dict size: {}, total: {}'.format(word, len(dict_df), line_count))
            else:
                dict_df.loc[dict_df['word'].isin([word]), 'count'] += 1
    fid.close()
    return dict_df


def create_dict_cn(corpus_path, max_word=10000):
    """
    Return dictionary dataframe.
    Column: word, count
    :param corpus_path:
    :param max_word:
    :return:
    """
    dict_df = pd.DataFrame(columns=('word', 'count'))
    if os.path.isdir(corpus_path):
        file_list = os.listdir(corpus_path)
        for file_name in file_list:
            file_path = corpus_path + '/' + file_name
            analyze_corpus(file_path, dict_df)
    else:
        dict_df = analyze_corpus(corpus_path, dict_df)
    return dict_df


if __name__ == '__main__':
    df = create_dict_cn(corpus_path='C:/bomb/book/ML/NLP/zenghuanzuang.txt')
    df.to_csv(path_or_buf=dict_path, sep=',')
