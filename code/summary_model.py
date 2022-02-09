import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class DeletionBasedSummaryModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False, use_fast=False)
        
        self.hidden_size = self.model.config.hidden_size
        self.classification_layer = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        outputs = self.classification_layer(outputs)
        return self.sigmoid(outputs)        
