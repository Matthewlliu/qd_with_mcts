from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch import nn
import torch
import random
import os
import numpy as np
import torch.nn.functional as F

class RobertaFilter(nn.Module):
    def __init__(self, args, mode='single-hop'):
        super().__init__()
        self.args = args
        if mode == 'single-hop':
            model_name_or_path = args.filter_model_name_or_path
            self.max_length = self.args.filter_max_length
        else:
            model_name_or_path = args.cq_filter_model_name_or_path
            self.max_length = self.args.cq_filter_max_length
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
        except:
            self.tokenizer = RobertaTokenizer.from_pretrained(os.path.join(model_name_or_path, 'tokenizer'))
        try:
            self.model = RobertaForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1)
        except:
            self.model = RobertaForSequenceClassification.from_pretrained(os.path.join(model_name_or_path, 'model'), num_labels=1)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.device = args.device
        self.model.to(self.device)
        
    def save_pretrained(self, save_dir):
        self.model.save_pretrained(os.path.join(save_dir, 'model'))
        self.tokenizer.save_pretrained(os.path.join(save_dir, 'tokenizer'))
        
    @classmethod
    def from_pretrained(cls, args):
        return cls(args)
    
    def forward(self, input_texts, labels=None, k=None):
        btz = len(input_texts)
        inputs = self.tokenizer(input_texts, max_length=self.max_length, truncation='only_second', padding='longest', return_tensors='pt')
        input_ids, attention_mask = inputs.input_ids.to(self.device), inputs.attention_mask.to(self.device)
        output = {}

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1) # (B)
        probs = logits.sigmoid() # (B)
        output["logits"] = logits.tolist()
        output["probs"] = probs.tolist()
        if k is None:
            output["preds"] = (probs >= 0.5).tolist()
        else:
            pred_idxs = probs.topk(k=k)[1]
            preds = torch.zeros(btz, dtype=torch.long, device=self.device)
            preds[pred_idxs] = 1
            output["preds"] = preds.tolist()
            output["pred_idxs"] = pred_idxs.tolist()
            output["pred_ordered_idxs"] = probs.topk(k=btz)[1].tolist()
        if labels is not None:
            labels = labels.to(self.device)
            output["loss"] = self.loss(logits, labels.float()) 
            output["labels"] = labels.tolist()
        return output

    @torch.no_grad()
    def predict(self, question, contexts, k=None):
        self.model.eval()
        num_para = len(contexts)
        btz = 128
        context_probs = []
        for i in range(0, num_para, btz):
            input_texts = []
            for context in contexts[i: i+btz]:
                input_texts.append((question, context))
            probs = self.forward(input_texts)["probs"]
            context_probs.extend((context, prob) for context, prob in zip(contexts[i: i+btz], probs))
        if k is None:
            return [context for context, prob in context_probs if prob >= 0.5]
        else:
            context_probs.sort(key=lambda x:x[1], reverse=True)
            return [context for context, prob in context_probs[:k]]
            

def main():
    args = get_args()
    args.device = "cpu"
    filter = RobertaFilter(args)
    input_texts = [[("how are you?", "I'm fine."),
                ("how are you2?", "I'm fine2.")]]

    output = filter(input_texts)
    print(output)
    

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from param import get_args
    main()

