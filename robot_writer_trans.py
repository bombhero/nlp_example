import math
import torch
import time
from torch.utils.data import DataLoader
from dict_lib import DictWordVec
from writer_data import TrainDataset
from cuda_utils import select_device

corpus_path = './corpus/zenghuanzuang.txt'
dict_path = './dict/letter_dict.csv'
example_path = './example/zenghuanzuang.csv'
model_path = './model/robot_writer_transform.pkl'


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class WriterNN(torch.nn.Module):
    def __init__(self, dim_x):
        super(WriterNN, self).__init__()
        hidden_dim = 512
        n_header = 8
        self.emb = torch.nn.Embedding(num_embeddings=dim_x, embedding_dim=hidden_dim)
        self.position = PositionalEncoding(hidden_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_header)
        self.trans_encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=3)
        self.decoder = torch.nn.Linear(hidden_dim, dim_x)
        self.output_layer = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.emb(x)
        x = self.position(x)
        x = self.trans_encoder(x)
        x = self.decoder(x)
        output = self.output_layer(x)
        return output[:, -1, :]


def train():
    device = select_device()
    train_data = TrainDataset(corpus=corpus_path, dict_file=dict_path, storage_path=example_path, seq_len=50)
    train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=4)
    writer_nn = WriterNN(train_data.get_dim_x()).to(device)
    print(writer_nn)

    optimizer = torch.optim.Adam(writer_nn.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    total_loss_list = []
    show_letter = ['-', '\\', '|', '/']
    for epoch in range(1000):
        start_ts = time.time()
        writer_nn.train()
        total_idx = int(len(train_data) / 128)
        total_loss = 0
        for idx, data in enumerate(train_loader):
            tensor_x, tensor_y = data
            train_x = torch.autograd.Variable(tensor_x).to(device)
            train_y = torch.autograd.Variable(tensor_y[:, 0]).to(device)
            optimizer.zero_grad()
            output = writer_nn(train_x)
            loss = loss_func(output, train_y)
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


def predict(first_sentence):
    device = select_device()
    write_nn = torch.load(model_path)
    dict_word = DictWordVec(corpus_path, dict_path)

    result_sentence = first_sentence

    write_nn.eval()



if __name__ == '__main__':
    train()
