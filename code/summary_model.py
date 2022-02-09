import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class DeletionBasedSummaryModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False, use_fast=False)
        
        self.hidden_size = self.model.config.hidden_size
        self.classification_layer = nn.Linear(self.hidden_size, 1)

        sigmoid = nn.Sigmoid()

    def forward(inputs):
        print(1)
