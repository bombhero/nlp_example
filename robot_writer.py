"""
    Generate the document by LSTM model.
"""
import os
import time
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dict_lib import DictWordVec
from torch.utils.tensorboard import SummaryWriter
from dict_lib import create_word_dict_cn
from writer_data import TrainDataset

corpus_path = './corpus/zenghuanzuang.txt'
dict_path = './dict/letter_dict.csv'
example_path = './example/zenghuanzuang.csv'
model_path = './model/robot_writer.pkl'


class WriterNN(torch.nn.Module):
    def __init__(self, dim_x):
        super(WriterNN, self).__init__()
        hidden_dim = 1024
        self.emb = torch.nn.Embedding(num_embeddings=dim_x, embedding_dim=hidden_dim)
        self.rnn = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=3, batch_first=True)
        self.rnn1 = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=3, batch_first=True)
        self.rnn2 = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=3, batch_first=True)

    def forward(self, x, y, h, c):
        vec_x = self.emb(x)
        vec_y = self.emb(y)
        if h is None:
            output, (h_out, c_out) = self.rnn(vec_x, None)
        else:
            output, (h_out, c_out) = self.rnn(vec_x, (h, c))
        output, (h_out, c_out) = self.rnn1(output, (h_out, c_out))
        output, (h_out, c_out) = self.rnn2(output, (h_out, c_out))
        return output[:, -1:, :], vec_y, h_out, c_out

    def un_emb(self, vec):
        # return torch.argmax(self.emb.weight.matmul(vec.t()), dim=0)
        return torch.nn.functional.softmax(self.emb.weight.matmul(vec.t()), dim=0)


def select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def train():
    tmp_model_path = './model/robot_writer_tmp.pkl'
    device = select_device()
    train_data = TrainDataset(corpus=corpus_path, dict_file=dict_path, storage_path=example_path, seq_len=50)
    train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=4)
    writer_nn = WriterNN(train_data.get_dim_x()).to(device)
    print(writer_nn)

    optimizer = torch.optim.Adam(writer_nn.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()
    total_loss_list = []
    show_letter = ['-', '\\', '|', '/']
    for epoch in range(1000):
        start_ts = time.time()
        writer_nn.train()
        total_idx = int(len(train_data) / 128)
        total_loss = 0
        for idx, data in enumerate(train_loader):
            tensor_x, tensor_y = data
            h_out = None
            c_out = None
            train_x = torch.autograd.Variable(tensor_x).to(device)
            train_y = torch.autograd.Variable(tensor_y).to(device)
            optimizer.zero_grad()
            output, vec_y, h_out, c_out = writer_nn(train_x, train_y, h_out, c_out)
            h_out = h_out.data
            c_out = c_out.data
            loss = loss_func(output, vec_y)
            record_loss = loss.data.cpu()
            total_loss += record_loss
            loss.backward()
            optimizer.step()
            if total_idx < 20:
                print('{}:{}/{}: loss= {}'.format(epoch, idx, total_idx, record_loss))
            else:
                print("{}{}".format(idx, show_letter[idx % len(show_letter)]), end='\r')
        end_ts = time.time()
        print('{} Spent {}, loss={}'.format(epoch, (end_ts - start_ts), total_loss))
        if len(total_loss_list) == 0 or total_loss < total_loss_list[-1]:
            total_loss_list.append(total_loss)
            torch.save(writer_nn, model_path)
        else:
            break


def select_one_word(prob_seq):
    num = random.random()
    seq = []
    total = 0
    prob_seq = np.power(prob_seq, 1/0.8)
    prob_seq = prob_seq / np.sum(prob_seq)
    for idx in range(prob_seq.shape[0]):
        total += prob_seq[idx, 0]
        seq.append(total)
    for idx in range(len(seq)):
        if idx == 0 and num < seq[idx]:
            return 0
        if (num >= seq[idx-1]) and (num < seq[idx]):
            return idx
    return 0


def predict(first_sentence):
    device = select_device()
    writer_nn = torch.load(model_path)
    dict_word = DictWordVec(corpus_path, dict_path)

    h_out = None
    c_out = None
    result_sentence = first_sentence

    writer_nn.eval()
    # Input the first sentence
    for letter in first_sentence:
        if len(letter.strip()) == 0:
            continue
        word_vec = torch.LongTensor([[dict_word.trans_word_to_vec(letter)]])
        nouse_y = torch.LongTensor([[0]])
        output, vec_y, h_out, c_out = writer_nn(word_vec.to(device), nouse_y.to(device), h_out, c_out)
        h_out = h_out.data
        c_out = c_out.data

    for idx in range(100):
        letter_seq = writer_nn.un_emb(output[:, -1, :]).cpu().detach().numpy()
        input_word_id = select_one_word(letter_seq)
        letter = dict_word.trans_vec_to_word(input_word_id)
        result_sentence = result_sentence + letter
        print(result_sentence)
        word_vec = torch.LongTensor([[input_word_id]]).to(device)
        nouse_y = torch.LongTensor([[0]]).to(device)
        output, vec_y, h_out, c_out = writer_nn(word_vec, nouse_y, h_out, c_out)
        h_out = h_out.data
        c_out = c_out.data


if __name__ == '__main__':
    # train()
    predict('不知道')
    # rand_seq = [np.random.randint(0, 4500) for _ in range(100)]
    # seq = torch.tensor([rand_seq])
    # y = torch.tensor([[5]])
    # tensor_summary = SummaryWriter('./tensor_log')
    # writer_nn = WriterNN(4500)
    # tensor_summary.add_graph(writer_nn, [seq, y, None, None])
