import copy
import torchtext
import torch
from torch import optim, nn
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
import io
import numpy as np

from seq_seq_model import Encoder, Attention, Decoder, Seq2Seq, count_parameters, init_weights


def build_vocab(_list, tokenizer, _index):
    counter = Counter()
    for string_ in _list:
        counter.update(tokenizer(str(string_[_index])))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


def find_seq_length(self, dataset):
    return max(len(seq.split()) for seq in dataset)


def find_vocab_size(self, tokenizer, dataset):
    tokenizer.fit_on_texts(dataset)


def data_process(_arr):
    data = []
    for [en, ger] in _arr:
        de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(str(ger))],
                                  dtype=torch.long)
        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(str(en))],
                                  dtype=torch.long)
        data.append((de_tensor_, en_tensor_))
    return data


filename = '/Users/rajshekhorroy/Downloads/english-german-both.pkl'
clean_dataset = copy.deepcopy(np.load(open(filename, 'rb'), allow_pickle=True))
print('data loaded')

for i in range(clean_dataset[:, 0].size):
    clean_dataset[i, 0] = "<bos> " + clean_dataset[i, 0] + " <eos>"
    clean_dataset[i, 1] = "<bos> " + clean_dataset[i, 1] + " <eos>"

clean_train_dataset = copy.deepcopy(clean_dataset[0:9000])
clean_val_dataset = copy.deepcopy(clean_dataset[9000:10000])

print('split done')

de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

de_vocab = build_vocab(clean_train_dataset, de_tokenizer, _index=1)
en_vocab = build_vocab(clean_train_dataset, en_tokenizer, _index=0)

# train_data = data_process(clean_train_dataset)
# train_data = data_process(train_filepaths)
val_data = data_process(clean_val_dataset)
# test_data = data_process(test_filepaths)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 1
PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def generate_batch(data_batch):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batch


valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)




INPUT_DIM = len(de_vocab)
OUTPUT_DIM = len(en_vocab)
##large
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ATTN_DIM = 64
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
###small
# ENC_EMB_DIM = 16
# DEC_EMB_DIM = 16
# ENC_HID_DIM = 32
# DEC_HID_DIM = 32
# ATTN_DIM = 4
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)


print(f'The model has {count_parameters(model):,} trainable parameters')


model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load("all_models/de_en/models/ep_17", weights_only=True))

# optimizer = optim.Adam(model.parameters())
import math
import time



def tensor_to_words(tensor, vocab):
    # tensor: [max_len, batch_size] -> output of the model after argmax
    sentences = []
   # Transpose to get batch-first
    words = [vocab.itos[idx.item()] for idx in tensor if vocab.itos[idx.item()] != '<pad>']
    sentences.append(' '.join(words))
    return sentences
def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):

    # model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            pred=output.argmax(axis=1)


            predicted_sentences = tensor_to_words(pred, en_vocab,)
            inp = tensor_to_words(src,de_vocab)
            target = tensor_to_words(trg, en_vocab)
            print(inp,target,predicted_sentences,loss)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

PAD_IDX = en_vocab.stoi['<pad>']


N_EPOCHS = 100
CLIP = 1

best_valid_loss = float('inf')
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)




valid_loss = evaluate(model, valid_iter, criterion)


