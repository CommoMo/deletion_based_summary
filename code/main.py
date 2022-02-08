import json

import argparse
from attrdict import AttrDict
from transformers.models.auto.configuration_auto import AutoConfig
from utils import init_logger, set_seed, init_system
from data_utils import IntentDataset, collate_fn

import torch
from torch.utils.data import DataLoader
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import os
from fastprogress.fastprogress import master_bar, progress_bar

import datetime

import logging

from model import ClassificationModel

LOG_PATH = './logs'

def main(cli_args):
    # read from config file and make args
    with open(os.path.join('configs', cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
        args['output_dir'] = cli_args.output_dir

    with open(os.path.join('configs', cli_args.config_file), 'w') as f:
        json.dump(args, f, indent='\t')

    ### Set Loggers ###
    logger = logging.getLogger('[System]')
    logger_path = os.path.join(LOG_PATH, f'{args.model_type}.logs')
    file_handler = logging.FileHandler(logger_path)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    
    logger.info(f"  >> Current path: {os.getcwd()}")
    logger.info("Training/evaluation parameters {}".format(args))


    model = DeletionBasedSummaryModel(args, pretrained_model).to(device)


    if args.do_train:
        ### Training configs ###
        ### Data Loading ###
        global_step, train_loss = train(args, train_dataloader, model, tokenizer, test_dataloader)
        logger.info(" global_step = %s, average loss = %s", global_step, train_loss)

    if args.do_eval:
        best_acc = 0
        best_checkpoint = ''
        checkpoints = sorted([_ for _ in os.listdir(args.output_dir) if _.startswith('best')])
        logger.info(f"Evaluate the following checkpoint: {args.output_dir}")

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = torch.load(os.path.join(args.output_dir, checkpoint, 'intent_class_model.pt'))
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.output_dir, checkpoint))
            
            eval_loss, eval_acc = evaluate(args, model, tokenizer, test_dataloader)
            
            if eval_acc > best_acc:
                best_acc = eval_acc
                best_checkpoint = checkpoint
            # result.append(f'[{checkpoint}] Loss: {eval_loss:.3f}, Accuaracy: {eval_acc:.3f}')
            logger.info(f'[{checkpoint}] Loss: {eval_loss:.4f}, Accuaracy: {eval_acc:.4f}')
        logger.info(f"[VALID] All checkpoint validations are done.")
        logger.info(f"Best checkpoint: {best_checkpoint}, Accuracy: {best_acc:.2f}")
        
    logger.info(f"finished")
    train_logger.info('finished')

    

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--config_file", type=str, default='configs/koelectra_small.json')
    cli_parser.add_argument("--output_dir", type=str, default='koelectra_small_ckpt')
    cli_args = cli_parser.parse_args()

    main(cli_args)
