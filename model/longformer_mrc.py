from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering
from torch import nn
import torch
import random
import os
import numpy as np
import sys
#sys.path.append('..')
from .utils import answer_offset_in_context

class LongformerMRC(nn.Module):
    def __init__(self, args, mode='single-hop'):
        super().__init__()
        self.args = args
        if mode == 'single-hop':
            model_name_or_path = args.mrc_model_name_or_path
            self.max_question_length = self.args.mrc_max_question_length
            self.max_context_length = self.args.mrc_max_context_length
            self.max_length_generate = self.args.mrc_max_length_generate
            self.use_predicted_topk_contexts = self.args.mrc_use_predicted_topk_contexts
        else:
            model_name_or_path = args.cq_mrc_model_name_or_path
            self.max_question_length = self.args.cq_mrc_max_question_length
            self.max_context_length = self.args.cq_mrc_max_context_length
            self.max_length_generate = self.args.cq_mrc_max_length_generate
            self.use_predicted_topk_contexts = self.args.cq_mrc_use_predicted_topk_contexts
        try:
            self.tokenizer = LongformerTokenizerFast.from_pretrained(model_name_or_path)
        except:
            self.tokenizer = LongformerTokenizerFast.from_pretrained(os.path.join(model_name_or_path, 'tokenizer'))
        try:
            self.model = LongformerForQuestionAnswering.from_pretrained(model_name_or_path)
        except:
            self.model = LongformerForQuestionAnswering.from_pretrained(os.path.join(model_name_or_path, 'model'))
        self.device = args.device
        new_tokens = ['[SEP]']
        added_tokens_num = self.tokenizer.add_tokens(new_tokens, special_tokens = False)
        print('added_tokens_num:', added_tokens_num)
        if added_tokens_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.qa_loss = nn.CrossEntropyLoss()
        self.supervise_support = self.args.mrc_supervise_support
        if self.supervise_support:
            self.support_fc = nn.Linear(self.model.config.hidden_size, 1).to(self.args.device)
            self.support_loss = nn.BCEWithLogitsLoss()
        self.supervise_answerable = self.args.mrc_supervise_answerable
        if self.supervise_answerable:
            self.answerable_fc = nn.Linear(self.model.config.hidden_size, 2).to(self.args.device)
            self.answerable_loss = nn.CrossEntropyLoss()
        self.model.to(self.args.device)
        
    def save_pretrained(self, save_dir):
        self.model.save_pretrained(os.path.join(save_dir, 'model'))
        self.tokenizer.save_pretrained(os.path.join(save_dir, 'tokenizer'))
        
    @classmethod
    def from_pretrained(cls, args):
        return cls(args)

    def clip(self, text, max_length):
        return self.tokenizer.decode(self.tokenizer(text, add_special_tokens=False).input_ids[:max_length], skip_special_tokens=True)
    
    def preprocess(self, questions, batch_contexts, is_training=False, answers=None, batch_support_labels=None):
        if not is_training:
            input_texts = []
            for question, contexts in zip(questions, batch_contexts):
                input_text = self.clip(question.strip(), self.max_question_length) + "</s></s>"
                for context in contexts:
                    input_text += "[SEP]" + self.clip(context.strip(), self.max_context_length)
                input_texts.append(input_text)
            return self.tokenizer(input_texts, padding=True, return_tensors='pt')
        else:
            input_texts, is_support_labels, answer_start_offsets, answer_end_offsets = [], [], [], []
            for question, contexts, answer, support_labels in zip(questions, batch_contexts, answers, batch_support_labels):
                question = self.clip(question.strip(), self.max_question_length)
                answer = answer.strip()
                input_text = question + "</s></s>"
                answer_start_offset, answer_end_offset = None, None
                for context, is_support in zip(contexts, support_labels):
                    context = self.clip(context.strip(), self.max_context_length)
                    if is_support:
                        local_answer_offset = answer_offset_in_context(answer, context)
                        if local_answer_offset is not None:
                            answer_start_offset = len(input_text) + len("[SEP]") + local_answer_offset
                            answer_end_offset = answer_start_offset + len(answer) - 1
                    input_text += "[SEP]" + context
                input_texts.append(input_text)
                is_support_labels.append(support_labels)
                answer_start_offsets.append(answer_start_offset)
                answer_end_offsets.append(answer_end_offset)
            encoding = self.tokenizer(input_texts, padding=True, return_offsets_mapping=True, return_tensors='pt')
            answer_start_positions, answer_end_positions = [], []
            for ans_st, ans_ed, token_spans in zip(answer_start_offsets, answer_end_offsets, encoding.offset_mapping):
                if ans_st is None:
                    ans_st_pos, ans_ed_pos = 0, 0
                else:
                    ans_st_pos, ans_ed_pos = None, None
                    #print(ans_st, ans_ed, token_spans)
                    for i, token_span in enumerate(token_spans):
                        token_st, token_ed = token_span.tolist()
                        if ans_st >= token_st and ans_st < token_ed:
                            ans_st_pos = i
                        if ans_ed >= token_st and ans_ed < token_ed:
                            ans_ed_pos = i
                    assert ans_st_pos is not None and ans_ed_pos is not None
                answer_start_positions.append(ans_st_pos)
                answer_end_positions.append(ans_ed_pos)
            return {
                "input_texts": input_texts,
                "input_ids": encoding.input_ids,
                "attention_mask": encoding.attention_mask,
                "start_positions": torch.LongTensor(answer_start_positions),
                "end_positions": torch.LongTensor(answer_end_positions),
                "support_labels": torch.LongTensor(is_support_labels),
            }  
            
    def get_best_span(
        self, 
        span_start_logits,
        span_end_logits,
        max_length=None, 
    ):
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        # (batch_size, passage_length, passage_length)
        span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
        # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
        # the span ends before it starts.
        span_mask = torch.triu(torch.ones((passage_length, passage_length), device=self.device))

        if max_length is not None:
            range_vector = torch.arange(passage_length, device=self.device)
            range_matrix = range_vector.unsqueeze(0)-range_vector.unsqueeze(1)
            length_mask = ((range_matrix < max_length) & (range_matrix >= 0))
            span_mask = (span_mask.long() & length_mask).float()

        span_log_mask = span_mask.log()

        valid_span_log_probs = span_log_probs + span_log_mask
        #valid_span_log_probs[:, 0, :] = -np.inf
        # Here we take the span matrix and flatten it, then find the best span using argmax.  We
        # can recover the start and end indices from this flattened list using simple modular
        # arithmetic.
        # (batch_size, passage_length * passage_length)
        best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
        span_start_indices = best_spans // passage_length
        span_end_indices = best_spans % passage_length
        return span_start_indices, span_end_indices
        
    def forward(self, questions, batch_contexts, is_training=False, answers=None, batch_support_labels=None, batch_answerable_labels=None, k=1):
        inputs = self.preprocess(questions, batch_contexts, is_training, answers, batch_support_labels)
        input_ids, attention_mask = inputs["input_ids"].to(self.device), inputs["attention_mask"].to(self.device)
        btz, num_para = len(input_ids), len(batch_contexts[0])
        if is_training:
            start_positions, end_positions = inputs["start_positions"].to(self.device), inputs["end_positions"].to(self.device)
            outputs = self.model(input_ids,
                                 attention_mask=attention_mask,
                                 start_positions=start_positions,
                                 end_positions=end_positions,
                                 output_hidden_states=self.supervise_support)
            loss = outputs.loss
            #print(loss)
            if self.supervise_support:
                support_hidden_states = outputs.hidden_states[-1][input_ids==self.sep_token_id].view(btz, num_para, -1) # (B, k, d)
                support_logits = self.support_fc(support_hidden_states).squeeze(-1) # (B, k)
                support_loss = self.support_loss(support_logits.view(-1), inputs["support_labels"].to(self.device).float().view(-1))
                '''
                print("")
                print(support_logits.view(-1), inputs["support_labels"].to(self.device).float().view(-1))
                print(support_loss)
                '''
                loss += support_loss
            if self.supervise_answerable:
                cls_hidden_states = outputs.hidden_states[-1][:, 0, :] # (B, d)
                answerable_logits = self.answerable_fc(cls_hidden_states) # (B, 2)
                answerable_loss = self.answerable_loss(answerable_logits, torch.LongTensor(batch_answerable_labels).to(self.device)) # (B)
                loss += answerable_loss
            return {
                "loss": loss, 
                "data_num": len(input_ids)
            }
        else:
            outputs = self.model(input_ids, 
                            attention_mask=attention_mask,
                            output_hidden_states=self.supervise_support)
            start_logits = outputs.start_logits.squeeze(-1).masked_fill(~attention_mask.bool(), -np.inf) # (B, L)
            end_logits = outputs.end_logits.squeeze(-1).masked_fill(~attention_mask.bool(), -np.inf) # (B, L)
            start_probs, end_probs = start_logits.softmax(-1), end_logits.softmax(-1) # (B, L)
            '''
            start_probs, start_positions = start_probs.max(-1)
            end_probs, end_positions = end_probs.max(-1) #(B)
            '''
            start_positions, end_positions = self.get_best_span(start_logits, end_logits, max_length=self.max_length_generate) # (B)
            pred_answer_probs = []
            for input_id, st_pos, ed_pos, st_prob, ed_prob in zip(input_ids, start_positions, end_positions, start_probs, end_probs):
                pred_answer = self.tokenizer.decode(input_id[st_pos:ed_pos+1], skip_special_tokens=True).strip()
                pred_answer_prob = st_prob[st_pos] * ed_prob[ed_pos]
                pred_answer_probs.append((pred_answer, pred_answer_prob.item()))
            output = {"pred_answer_probs": pred_answer_probs}
            
            pred_is_supports = torch.zeros(btz, num_para, dtype=torch.long, device=self.device) # (B, n)
            if self.supervise_support:
                support_hidden_states = outputs.hidden_states[-1][input_ids==self.sep_token_id].view(btz, num_para, -1) # (B, n, d)
                support_logits = self.support_fc(support_hidden_states).squeeze(-1) # (B, n)
                output["support_probs"] = support_logits.sigmoid() #(B, n)
                pred_support_idxs = support_logits.topk(k=k)[1] # (B, k)
                #print(pred_support_idxs.size())
                for i, idxs in enumerate(pred_support_idxs):
                    pred_is_supports[i][idxs] = 1
                output["pred_is_supports"] = pred_is_supports.tolist()
            else:
                pred_is_supports[:, :k] = 1
            output["pred_is_supports"] = pred_is_supports.tolist()
            
            if self.supervise_answerable:
                cls_hidden_states = outputs.hidden_states[-1][:, 0, :] # (B, d)
                answerable_logits = self.answerable_fc(cls_hidden_states) # (B, 2)
                output["pred_answerable_probs"] = answerable_logits.softmax(dim=-1)[:, 1].tolist()
            else:
                output["pred_answerable_probs"] = [1]*btz
        return output
    
    @torch.no_grad()
    def predict(self, question, contexts, para_idxs=None, k=1):
        self.eval()
        #print(input_texts)
        output = self.forward([question], [contexts], is_training=False, k=k)
        prediction, score = output["pred_answer_probs"][0]
        support_contexts = [context for i, context in enumerate(contexts) if output["pred_is_supports"][0][i] == 1]
        if para_idxs is not None:
            pred_support_idxs = [idx for i, idx in enumerate(para_idxs) if output["pred_is_supports"][0][i] == 1]
            return prediction, score, support_contexts, pred_support_idxs
        else:
            return prediction, score, support_contexts
    

if __name__ == "__main__":
    from param import get_args
    args = get_args()
    args.device = "cpu"
    mrc = LongformerMRC(args)
    input_ids = [2206, 1934, 1580]
    print(mrc.tokenizer.decode(input_ids))
    exit()
    question = "What is a pig?"
    contexts = ["how are you?", "Mary Henry am a pig."]
    answer = "Mary Henry"
    inputs = mrc.preprocess([question], [contexts], is_training=True, answers=[answer], batch_support_labels=[[False, True]])
    print(inputs)
    '''
    print(mrc.clip(context, 2))
    tokens = mrc.tokenizer(context, add_special_tokens = False, return_offsets_mapping=True, return_tensors='pt', padding=True)
    print(tokens)
    
    input_texts = [("how are you?", "I'm fine."),
                ("how are you2?", "I'm fine2.")]

    output_texts, scores = mrc.predict(input_texts)
    print(output_texts, scores)
    '''

