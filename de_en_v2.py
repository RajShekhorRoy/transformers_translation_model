import copy
import torch
from torch import optim, nn
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import math
import time
import numpy as np

from llm_libs import build_vocab, data_process, train, evaluate, epoch_time
##https://pytorch.org/tutorials/beginner/translation_transformer.html
from seq_seq_model import Encoder, Attention, Decoder, Seq2Seq, count_parameters, init_weights

filename = '/Users/rajshekhorroy/Downloads/english-german-both.pkl'
clean_dataset = copy.deepcopy(np.load(open(filename, 'rb'), allow_pickle=True))
print('data loaded')

clean_train_dataset = copy.deepcopy(clean_dataset[0:9000])
clean_val_dataset = copy.deepcopy(clean_dataset[9000:10000])

print('split done')

de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

de_vocab = build_vocab(clean_train_dataset, de_tokenizer, _index=1)
en_vocab = build_vocab(clean_train_dataset, en_tokenizer, _index=0)

train_data = data_process(clean_train_dataset, de_vocab, en_vocab, de_tokenizer, en_tokenizer)
# train_data = data_process(train_filepaths)
val_data = data_process(clean_val_dataset, de_vocab, en_vocab, de_tokenizer, en_tokenizer)
# test_data = data_process(test_filepaths)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 1
PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']


def generate_batch(data_batch):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batch


train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)

INPUT_DIM = len(de_vocab)
OUTPUT_DIM = len(en_vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# ENC_HID_DIM = 512
# DEC_HID_DIM = 512
# ATTN_DIM = 64
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
ENC_HID_DIM = 256
DEC_HID_DIM = 256
ATTN_DIM = 64
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')

model.apply(init_weights)

optimizer = optim.Adam(model.parameters())

PAD_IDX = en_vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

N_EPOCHS = 100
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    torch.save(model.state_dict(), './models/ep_' + str(epoch))
test_loss = evaluate(model, valid_iter, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
