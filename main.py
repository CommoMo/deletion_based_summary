import argparse
import sys
sys.path.append('./src')
import torch
from utils import *
from trainer import train
from summary_model import DeletionBasedSummaryModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(cli_args):
    args = init_setting(cli_args)
    model = DeletionBasedSummaryModel(args).to(device)

    if args.do_train:
        train(args, model)
    

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--config_file", type=str, default='klue_roberta_large.json')
    # cli_parser.add_argument("--config_file", type=str, default='koelectra_small.json')
    # cli_parser.add_argument("--output_dir", type=str, default='koelectra_small_ckpt')
    cli_args = cli_parser.parse_args()

    main(cli_args)
