import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dict_lib import create_word_dict_cn

corpus_path = './corpus/zenghuanzuang.txt'
dict_path = './dict/letter_dict.csv'
example_path = './example/zenghuanzuang.csv'
line_limit = 0


class TrainDataset(Dataset):
    def __init__(self, corpus, dict_file, storage_path, seq_len=50, max_word=10000):
        self.corpus = corpus
        self.dict = create_word_dict_cn(corpus, dict_file)
        self.dim_x = len(self.dict)
        self.seq_len = seq_len
        self.slot = 5
        if os.path.exists(storage_path):
            self.data = np.loadtxt(storage_path, delimiter=',')
        else:
            self.data = self.create_data_x()
            np.savetxt(fname=example_path, X=self.data, delimiter=',', fmt='%d')
        print('Dataset created {}'.format(self.data.shape[0]))

    def __getitem__(self, index):
        """
        得到训练样本
        前seq_len个汉字为输入,后1个汉字为输出.
        一个样本为seq_len+1个汉字.
        :param index:
        :return:
        """
        index = index * self.slot
        seq_x = torch.LongTensor(self.data[index:(index+self.seq_len)])
        seq_y = torch.LongTensor(self.data[(index+self.seq_len):(index+self.seq_len+1)])
        return seq_x, seq_y

    def __len__(self):
        return int((self.data.shape[0] - self.seq_len) / self.slot)
        # return 1280

    def create_data_x(self):
        line_count = 0
        data_x = None
        fid = open(self.corpus, encoding='utf-8')
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) == 0:
                continue
            line_count += 1
            print('Analyze {}'.format(line_count))
            if (line_count >= line_limit) and (line_limit > 0):
                break
            for idx in range(len(line)):
                tmp_df = self.dict[self.dict['word'] == line[idx]]
                if len(tmp_df) == 0:
                    word_hot = 0
                else:
                    word_hot = tmp_df.iloc[0]['hot']
                word_vec = np.array([word_hot])
                if data_x is None:
                    data_x = word_vec
                else:
                    data_x = np.concatenate((data_x, word_vec), axis=0)
        fid.close()
        return data_x

    def get_dim_x(self):
        return self.dim_x
