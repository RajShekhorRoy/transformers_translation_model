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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tensor_to_words(tensor, vocab):
    # tensor: [max_len, batch_size] -> output of the model after argmax
    sentences = []
    # Transpose to get batch-first
    words = [vocab.itos[idx.item()] for idx in tensor if vocab.itos[idx.item()] != '<pad>']
    sentences.append(' '.join(words))
    return sentences


def build_vocab(_list, tokenizer, _index):
    counter = Counter()
    for string_ in _list:
        counter.update(tokenizer(str(string_[_index])))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


def tensor_to_words(tensor, vocab):
    # tensor: [max_len, batch_size] -> output of the model after argmax
    sentences = []
    # Transpose to get batch-first
    words = [vocab.itos[idx.item()] for idx in tensor if vocab.itos[idx.item()] != '<pad>']
    sentences.append(' '.join(words))
    return sentences


def find_seq_length(dataset):
    return max(len(seq.split()) for seq in dataset)


def find_vocab_size(tokenizer, dataset):
    tokenizer.fit_on_texts(dataset)


def data_process(_arr, _de_vocab, _en_vocab, _de_tokenizer, _en_tokenizer):
    data = []
    for [en, ger] in _arr:
        de_tensor_ = torch.tensor([_de_vocab[token] for token in _de_tokenizer(str(ger))],
                                  dtype=torch.long)
        en_tensor_ = torch.tensor([_en_vocab[token] for token in _en_tokenizer(str(en))],
                                  dtype=torch.long)
        data.append((de_tensor_, en_tensor_))
    return data


def tensor_to_words(tensor, vocab):
    # tensor: [max_len, batch_size] -> output of the model after argmax
    sentences = []
    # Transpose to get batch-first
    words = [vocab.itos[idx.item()] for idx in tensor if vocab.itos[idx.item()] != '<pad>']
    sentences.append(' '.join(words))
    return sentences


def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):
    model.train()

    epoch_loss = 0

    for _, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0)  # turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
