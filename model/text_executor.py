from transformers import BartTokenizer, BartForConditionalGeneration
from torch import nn
import torch
import random
import os
import numpy as np
from .text_retriever import TextRetriver
from .roberta_filter import RobertaFilter
#from .bart_mrc import BartMRC
from .longformer_mrc import LongformerMRC
#from .deberta_mrc import DeBertaMRC

name2mrc = {
    #"bart": BartMRC,
    "longformer": LongformerMRC,
    #"deberta": DeBertaMRC
}

class TextExecutor(nn.Module):
    def __init__(self, args, mode='single-hop'):
        super().__init__()
        self.args = args
        self.source = args.text_source
        self.retriever = TextRetriver()
        self.filter = RobertaFilter(args, mode)
        self.mrc = name2mrc[args.mrc_type](args, mode)
        if mode == 'single-hop':
            self.mrc_use_predicted_topk_contexts = self.args.mrc_use_predicted_topk_contexts
            self.filter_select_topk = self.args.filter_select_topk 
        else:
            self.mrc_use_predicted_topk_contexts = self.args.cq_mrc_use_predicted_topk_contexts
            self.filter_select_topk = self.args.cq_filter_select_topk 
    
    def predict(self, question, program, contexts=None):
        if contexts is None:
            contexts = self.retriever.retrieve(question, program, self.text_source)
        if self.mrc_use_predicted_topk_contexts is not None:
            #print("question", question)
            #print("contexts", contexts)
            #exit()
            assert self.filter_select_topk is not None
            contexts = self.filter.predict(question, contexts, k=self.mrc_use_predicted_topk_contexts)
            pred, score, support_contexts = self.mrc.predict(question, contexts, k=self.filter_select_topk)
            #print(pred, score, len(contexts))
            return [(pred, score, support_contexts)] if pred not in {"No answer", "[SEP]", ""} else []
        else:
            support_contexts = self.filter.predict(question, contexts, k=self.filter_select_topk)
            if len(support_contexts) == 0:
                return []
            #print('!!!\n', support_contexts)
            #exit()
            preds, scores = self.mrc.predict([(question, context) for context in support_contexts])
            preds = [(pred[0], score[0], [context]) for pred, score, context in zip(preds, scores, support_contexts) if pred[0] not in {"No answer", "[SEP]", ""}]
        preds.sort(key=lambda x: x[1], reverse=True)
        return preds
        