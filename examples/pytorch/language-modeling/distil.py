import datasets
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import pipeline


batch_size = 8
n_epochs = 30
lr = 1e-3
device = torch.device("cuda:1")

pipe = pipeline(model="gpt2", device=device)
distilled = pipeline(model="./starting_rank_10_diag", task="text-generation", device=device).model
# Modifying internals
encodings = pipe.tokenizer("test", truncation=True, padding="max_length")


def tokenize(example):
    encodings = pipe.tokenizer(example["src"], truncation=True, padding="max_length")
    return encodings


train_ds = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
train_ds.map(tokenize, batched=True)
train_ds = train_ds.with_format("torch", device=device)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

test_ds = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
test_ds.map(tokenize, batched=True)
test_ds = test_ds.with_format("torch", device=device)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

optimizer = torch.optim.AdamW(distilled.parameters(), lr=lr)

writer = SummaryWriter()

train_iter = 0
for epoch in range(n_epochs):
    for batch in train_loader:
        targets = pipe.model(**batch)
        results = distilled(**batch)

        target = targets.softmax(dim=-1)
        results = results.softmax(dim=-1)

        optimizer.zero_grad()
        loss = loss_fn(target, results)
        optimizer.step()
        writer.add_scalar("Loss/train", loss.item(), train_iter)
        train_iter += 1

    for batch in test_loader:
        targets = pipe.model(**batch)
        results = distilled(**batch)

        target = targets.softmax(dim=-1)
        results = results.softmax(dim=-1)

        loss = loss_fn(target, results)
        writer.add_scalar("Loss/test", loss.item(), train_iter)
