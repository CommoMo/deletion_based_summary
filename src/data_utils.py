import json
import os

from torch.utils.data import Dataset, DataLoader
import torch


import logging
logger = logging.getLogger('[System]')


def init_tokenizer(tokenize):
    global tokenizer
    tokenizer = tokenize

def get_dataloader(args, tokenizer, data_type):
    init_tokenizer(tokenizer)
    if data_type == 'train':
        with open(os.path.join(args.data_dir, args.train_file), 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        dataset = DeletionBasedSummaryDataset(args, tokenizer, dataset)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_fn)
    else:
        with open(os.path.join(args.data_dir, args.valid_file), 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        dataset = DeletionBasedSummaryDataset(args, tokenizer, dataset)
        dataloader = DataLoader(dataset, batch_size=args.valid_batch_size, collate_fn=collate_fn)
    
    return dataloader


class DeletionBasedSummaryDataset(Dataset):
    def __init__(self, args, tokenizer, dataset):
        self.args = args
        self.max_length = args.max_seq_length
        self.tokenizer = tokenizer

        self.dataset= dataset
        
    def __getitem__(self, idx):
        data = self.dataset[idx]
        ext = data['ext']
        sentences = data['sentences']
        
        
        sentence_ids = [self.tokenizer.tokenize(sent) for sent in sentences]
        ext_ids = [self.tokenizer.tokenize(sent) for sent in ext]
        label_list = []
        for sent_id in sentence_ids:
            if sent_id in ext_ids:
                indices = [1 for _ in range(len(sent_id))]
            else:
                indices = [0 for _ in range(len(sent_id))]
            label_list.append(indices)

        content = ' '.join(sentences)
        label = [0] + sum(label_list, [])
        label = torch.tensor(label[:self.max_length-1]+[0], dtype=torch.float)
        target_text = ' '.join(ext)

        return content, label, target_text
    
    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    content = [_[0] for _ in batch]
    labels = [_[1] for _ in batch]
    target_texts = [_[2] for _ in batch]
    # max_length = max([label.shape[0] for label in labels])
    inputs = tokenizer(content, truncation=True, max_length=512, return_tensors='pt', padding=True)

    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return inputs, labels, target_texts

