import json
import os
import numpy
from tqdm import tqdm
import argparse
import torch

from data_structure import simulation_model
from util import *

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=2381, help='random seed')
    #
    parser.add_argument('--expand_width', type=int, default=4, help='tree expanding width')
    parser.add_argument('--simulation_examples', type=int, default=5000)
    parser.add_argument('--rollout_times', type=int, default=100)
    parser.add_argument('--save_iter', type=int, default=100)
    parser.add_argument('--start_iter', type=int, default=0)
    # path
    parser.add_argument('--raw_data_path', type=str, default='/data/ljx/musique/musique_ans_v1.0_train.jsonl')
    parser.add_argument('--save_path', type=str, default='/data/ljx/result/qd_mcts')

    parser.add_argument('--text_source', type=str, default="ES")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--mrc_type', type=str, default="longformer")

    # checkpoints
    parser.add_argument('--filter_model_name_or_path', type=str, default="/data/ljx/roht/musique/checkpoints_filter/roberta_large_btz1_epoch5_softmax/ckpt-best")
    parser.add_argument('--filter_max_length', type=int, default=256)
    parser.add_argument('--filter_select_topk', type=int, default=1)

    parser.add_argument('--cq_filter_model_name_or_path', type=str, default="/data/ljx/roht/musique/checkpoints_filter/cq_roberta_large_btz16_epoch5/ckpt-best")
    parser.add_argument('--cq_filter_max_length', type=int, default=384)
    parser.add_argument('--cq_filter_select_topk', type=int, default=1)

    parser.add_argument('--mrc_model_name_or_path', type=str, default="/data/ljx/roht/musique/checkpoints_mrc/pretrained_longformer_large_btz16_epoch5_support_random/ckpt-best")
    parser.add_argument('--mrc_max_question_length', type=int, default=150)
    parser.add_argument('--mrc_max_context_length', type=int, default=400)
    parser.add_argument('--mrc_max_length_generate', type=int, default=32)
    parser.add_argument('--mrc_use_predicted_topk_contexts', type=int, default=3)
    parser.add_argument('--mrc_supervise_support', default=True, action="store_true")
    parser.add_argument('--mrc_supervise_answerable', default=False, action="store_true")

    parser.add_argument('--cq_mrc_model_name_or_path', type=str, default="/data/ljx/roht/musique/checkpoints_mrc/cq_pretrained_longformer_large_btz16_epoch5_support_random/ckpt-best")
    parser.add_argument('--cq_mrc_max_question_length', type=int, default=150)
    parser.add_argument('--cq_mrc_max_context_length', type=int, default=400)
    parser.add_argument('--cq_mrc_max_length_generate', type=int, default=32)
    parser.add_argument('--cq_mrc_use_predicted_topk_contexts', type=int, default=7)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sm = simulation_model(args)

    sm.simulate()
    
    save_data()

if __name__=='__main__':
    main()