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
    dataset = DeletionBasedSummaryDataset(args, tokenizer, data_type=data_type)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_fn)
    return dataloader


class DeletionBasedSummaryDataset(Dataset):
    def __init__(self, args, tokenizer, data_type='train'):
        self.args = args

        if data_type == 'train':
            with open(os.path.join(args.data_dir, args.train_file), 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
        else:
            with open(os.path.join(args.data_dir, args.valid_file), 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
                
        logger.info(f"Checking {data_type} dataset")
        self.tokenizer = tokenizer
        
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
        label = torch.tensor(label[:512], dtype=torch.float)

        return content, label
    
    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    content = [_[0] for _ in batch]
    labels = [_[1] for _ in batch]
    # max_length = max([label.shape[0] for label in labels])
    inputs = tokenizer(content, truncation=True, max_length=512, return_tensors='pt', padding=True)

    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return inputs, labels

