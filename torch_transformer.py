import math
import tempfile

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
from torchtext.data import Field

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import  Multi30k
from typing import Iterable, List


# Update URLs to point to data stored by user
# Multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
# Multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
# Multi30k.URL["test"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz"

# Update hash since there is a discrepancy between user hosted test split and that of the test split in the original dataset
# Multi30k.MD5["test"] = "6d1ca1dba99e2c5dd54cae1226ff11c2551e6ce63527ebb072a1f70f72a5cd36"
SRC = Field(tokenize="spacy", tokenizer_language="en", init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize="spacy", tokenizer_language="de", init_token='<sos>', eos_token='<eos>', lower=True)

data_train = Multi30k(split='train',path="/Users/rajshekhorroy/Downloads/tformer_data/training/")
data_val = Multi30k(split='valid',path="/Users/rajshekhorroy/Downloads/tformer_data/validation/")
data_test = Multi30k(split='test',path="/Users/rajshekhorroy/Downloads/tformer_data/mmt16_task1_test/")
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}
class Encoder(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Need (S, N) format for encoder.
        src = src.t()
        src = self.encoder(src) * math.sqrt(self.ninp)
        return self.pos_encoder(src)


class Decoder(nn.Module):
    def __init__(self, ntoken, ninp):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp):
        # Need batch dimension first for output of pipeline.
        return self.decoder(inp).permute(1, 0, 2)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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


import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


def data_process(raw_text_iter):

    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter if len(item) <30]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device("cuda")


def batchify(data, bsz):
    # Divide the dataset into ``bsz`` parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the ``bsz` batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


batch_size = 2
eval_batch_size = 2
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

bptt = 10


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    # Need batch dimension first for pipeline parallelism.
    return data.t(), target


ntokens = len(vocab)  # the size of vocabulary
emsize = 512  # embedding dimension
nhid = 512  # the dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 12  # the number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 4  # the number of heads in the Multihead Attention models
dropout = 0.2  # the dropout value

# from torch.distributed import rpc
# tmpfile = tempfile.NamedTemporaryFile()
# rpc.init_rpc(
#     name="worker",
#     rank=0,
#     world_size=1,
#     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
#         init_method="file://{}".format(tmpfile.name),
#         # Specifying _transports and _channels is a workaround and we no longer
#         # will have to specify _transports and _channels for PyTorch
#         # versions >= 1.8.1
#         _transports=["ibv", "uv"],
#         _channels=["cuda_ipc", "cuda_basic"],
#     )
# )

num_gpus = 1
partition_len = ((nlayers - 1) // num_gpus) + 1

# Add encoder in the beginning.
tmp_list = [Encoder(ntokens, emsize, dropout).cuda(0)]
module_list = []

# Add all the necessary transformer blocks.
for i in range(nlayers):
    transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
    if i != 0 and i % (partition_len) == 0:
        module_list.append(nn.Sequential(*tmp_list))
        tmp_list = []
    device = i // (partition_len)
    tmp_list.append(transformer_block.to(device))

# Add decoder in the end.
tmp_list.append(Decoder(ntokens, emsize).cuda(num_gpus - 1))
module_list.append(nn.Sequential(*tmp_list))

from torch.distributed.pipeline.sync import Pipe

# Build the pipeline.
chunks = 8
model = Pipe(torch.nn.Sequential(*module_list), chunks=chunks)


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


print('Total parameters in model: {:,}'.format(get_total_params(model)))
criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

import time


def train():
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(vocab)

    # Train only for 50 batches to keep script execution time low.
    nbatches = min(50 * bptt, train_data.size(0) - 1)

    for batch, i in enumerate(range(0, nbatches, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        # Since the Pipe is only within a single host and process the ``RRef``
        # returned by forward method is local to this node and can simply
        # retrieved via ``RRef.local_value()``.
        output = model(data).local_value()
        # Need to move targets to the device where the output of the
        # pipeline resides.
        loss = criterion(output.view(-1, ntokens), targets.cuda(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 10
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, nbatches // bptt, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(vocab)
    # Evaluate only for 50 batches to keep script execution time low.
    nbatches = min(50 * bptt, data_source.size(0) - 1)
    with torch.no_grad():
        for i in range(0, nbatches, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data).local_value()
            output_flat = output.view(-1, ntokens)
            # Need to move targets to the device where the output of the
            # pipeline resides.
            total_loss += len(data) * criterion(output_flat, targets.cuda(1)).item()
    return total_loss / (len(data_source) - 1)


best_val_loss = float("inf")
epochs = 3  # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
