import json
import os
import pandas as pd

from torch.utils.data import Dataset
import torch

import logging
logger = logging.getLogger('[System]')

class DeletionBasedSummaryDataset(Dataset):
    def __init__(self, args, tokenizer, data_type='train'):
        self.args = args

        if data_type == 'train':
            dataset = pd.read_csv(os.path.join(args.data_dir, args.train_file), sep='\t')
        else:
            data_type = 'valid'
            dataset = pd.read_csv(os.path.join(args.data_dir, args.predict_file), sep='\t')
        logger.info(f"Checking {data_type} dataset")

        self.sentences, self.labels = self.check_class_map(self.args, dataset)
        self.tokenizer = tokenizer
        
    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.sentences[idx])
        input_ids, attn_mask = tokenized['input_ids'], tokenized['attention_mask']
        return input_ids, attn_mask, self.labels[idx]
    
    def __len__(self):
        return len(self.labels)


    def check_class_map(self, args, dataset):
        sentences, labels = [], []
        dataset = dataset.values.tolist()
        class_map = {}
        class_cnt = 0
        # class_len = len(class_map)
        for data in dataset:
            sentence = data[1]
            label = data[0]
            if len(sentence) <= 5:
                logger.info(f"[Passed data] Index: {dataset.index(data)}, Sentence: {sentence}, Label: {label}")
                continue
            if label not in class_map:
                class_map[label] = class_cnt
                class_cnt += 1
            sentences.append(sentence)
            labels.append(class_map[label])

        # if class_len != len(class_map):
        with open(os.path.join(args.output_dir, 'class_map.json'), 'w', encoding='utf-8') as f:
            json.dump(class_map, f, indent='\t')

        return sentences, labels

def collate_fn(batch):
    input_ids = [torch.tensor(_[0]) for _ in batch]
    attention_masks = [torch.tensor(_[1]) for _ in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
    labels = [torch.tensor(_[2]) for _ in batch]
    return input_ids, attention_masks, torch.LongTensor(labels)

