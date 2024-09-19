from torchtext.vocab import build_vocab_from_iterator

from typing import Iterable, List

import torch
from torch import optim, nn
from torchtext.data import load_sp_model, sentencepiece_numericalizer, simple_space_split
from torchtext.data.utils import get_tokenizer

import pandas as pd

from seq_seq_transformer_model import Seq2SeqTransformer
from collections import Counter
from torchtext.vocab import Vocab


def build_vocab(_list, tokenizer, _index):
    counter = Counter()
    if tokenizer != None:
        for string_ in _list:
            counter.update(tokenizer(str(string_)))
    else:
        for string_ in _list:
            counter.update((str(string_)).split())
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def evaluate(model):
    model.eval()
    losses = 0

    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    print("english , bangla ,prediction")
    for src, tgt in val_data:
        pred_word = translate(model,src)
        print(src,tgt,pred_word)


        #
        # tgt_input = tgt[:-1, :]
        #
        # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        #
        # logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        #
        # tgt_out = tgt[1:, :]
        # loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        # losses += loss.item()

    # return losses / len(list(val_dataloader))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys
def tensor_to_words(tensor, vocab):
    # tensor: [max_len, batch_size] -> output of the model after argmax
    sentences = []
   # Transpose to get batch-first
    words = [vocab.itos[idx.item()] for idx in tensor if vocab.itos[idx.item()] != '<pad>']
    sentences.append(' '.join(words))
    return sentences[0].replace('<bos>',"").replace('<eos>',"").strip()

# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = torch.tensor([en_vocab[token] for token in en_tokenizer(str(src_sentence))], dtype=torch.long).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return tensor_to_words(list(tgt_tokens.cpu().numpy()),bn_vocab)

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            print(txt_input)
            txt_input = transform(txt_input)
        return txt_input

    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##https://pytorch.org/tutorials/beginner/translation_transformer.html
from seq_seq_model import Encoder, Attention, Decoder, Seq2Seq, count_parameters, init_weights

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'bn'

# Place-holders
token_transform = {}
vocab_transform = {}
sp_model = load_sp_model("./spm_user.model")
# eng_data = pd.read_csv("./english_cleaned.csv")
# bn_data = pd.read_csv("./bangla_cleaned.csv")
train_data = []
val_data = []
train_data_size = 36000
counter_of_data = 0
train_pdf = pd.read_csv('/Users/rajshekhorroy/Downloads/english2Beng_cleaned.csv')
eng_data = train_pdf['english_caption']
bn_data = train_pdf['bengali_caption']
sp_model = load_sp_model("./spm_user.model")

en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

for i in range(len(train_pdf)):

    a_bn_data = bn_data.values[i]
    a_eng_data = eng_data.values[i]
    if counter_of_data < 36000:
        train_data.append([a_eng_data, a_bn_data])
        counter_of_data += 1
    else:
        val_data.append([a_eng_data, a_bn_data])
en_vocab = build_vocab(eng_data, en_tokenizer, _index=0)
bn_vocab = build_vocab(bn_data, None, _index=1)


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


def generate_batch(data_batch):
    en_batch, bn_batch = [], []
    for (en_item, bn_item) in data_batch:
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        bn_batch.append(torch.cat([torch.tensor([BOS_IDX]), bn_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    bn_batch = pad_sequence(bn_batch, padding_value=PAD_IDX)
    return en_batch, bn_batch

##model stuff
torch.manual_seed(0)

SRC_VOCAB_SIZE = len(en_vocab)
TGT_VOCAB_SIZE = len(bn_vocab)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


from torch.utils.data import DataLoader


def data_process_bn_en(_arr):
    data = []
    for [en, bn] in _arr:
        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(str(en))], dtype=torch.long)
        bn_tensor = torch.tensor([bn_vocab[token] for token in (str(bn).split())], dtype=torch.long)
        data.append((en_tensor_,bn_tensor ))
    return data


train_iter = data_process_bn_en(train_data)
val_iter = data_process_bn_en(val_data)

val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=generate_batch)


def append_file_information(_filename, _info):
    with open(_filename, "a+") as f:
        f.write(str(_info)+ "\n")

from timeit import default_timer as timer

NUM_EPOCHS = 100
transformer.load_state_dict(torch.load('./all_models/en_bn/09192024/ep_27', weights_only=True))
for epoch in range(1, NUM_EPOCHS + 1):
    start_time = timer()
    # train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    # torch.save(transformer.state_dict(), './all_models/en_bn/09192024/ep_' + str(epoch))
    evaluate(transformer)

