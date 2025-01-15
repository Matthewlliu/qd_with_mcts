from tqdm import tqdm
import numpy as np
import re
from torch import nn
import torch
from collections import defaultdict, Counter
from text_executor import TextExecutor

import sys
sys.path.append("/data/ljx/qd_mcts")
from util import merge_question, load_json_line

class Avg():
    def __init__(self, sum=None, cnt=1):
        if sum is not None:
            self.sum, self.cnt = sum, cnt
        else:
            self.sum, self.cnt = 0, 0
            
    def update(self, value, k=1):
        self.sum += value * k
        self.cnt += k
    
    def merge(self, avg1):
        sum = self.sum + avg1.sum
        cnt = self.cnt + avg1.cnt
        return Avg(sum, cnt)
        
    def avg(self):
        return self.sum / max(1, self.cnt)

class rollout_model_toy(object):
    def __init__(self):
        pass

    def forward(self, q_list):
        base_score = 1 - 1/len(q_list)
        score = base_score + np.random.uniform()/len(q_list)

        error_rate = 0.5
        if np.random.uniform() < error_rate:
            return 0
        else:
            return score

class rollout_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = args
        
        self.singlehop_text_executor = TextExecutor(args, mode='single-hop')
        self.multihop_text_executor = TextExecutor(args, mode='multi-hop')

    def inference(self, tree, node):
        q_list = tree.list[node].q_list
        given_contexts = tree.context

        print("inference:\n", q_list)

        if tree.list[node].simulation_tree is None:
            q_tree = self.build_tree(q_list, tree.list, node)
        else:
            q_tree = tree.list[node].simulation_tree
        
        print("q_tree:\n", q_tree)
        # inference
        memo = defaultdict(list)
        success_memory = []
        for q_node in q_tree:
            memo = self._inference_iter(q_node, memo, given_contexts)
        success_memory.append(memo)
        if len(success_memory[0][q_tree[-1]["idx"]]) > 0:
            print("simul result:", success_memory[0][q_tree[-1]["idx"]])
            return success_memory[0][q_tree[-1]["idx"]][0][0], q_tree

    def _inference_iter(self, decomp_tree, memory, given_contexts=None):
        node = decomp_tree
        idx, question, ent_question, son_idxs, hop = node["idx"], node["tree_question"], node["ent_question"], node["son_idxs"], node["hop"]

        source2score = {"text":1}
        if hop > 1:
            source2score["son"] = 1
        
        if "text" in source2score:
            #s = time.perf_counter()
            if hop == 1:
                text_preds = self.singlehop_text_executor.predict(question, None, given_contexts)
            else:
                text_preds = self.multihop_text_executor.predict(question, None, given_contexts)
            memory[idx].extend((pred, Avg(score), support_contexts) for (pred, score, support_contexts) in text_preds)
        if "son" in source2score:
            func = "Bridge" if len(son_idxs) == 1 else "QueryRelation"
            try:
                #s = time.perf_counter()
                #print(func + " | " + ent_question)
                preds = getattr(self, func)(node, memory, given_contexts=given_contexts)
                #e = time.perf_counter()
                #print("SON execute time: %s" % (e-s))
                memory[idx].extend(preds)
            except:
                pass 
            
            
        memory[idx] = self.santitize_preds(memory[idx])
        
        return memory

    def build_tree(self, q_list, tree, v):
        ret = []
        # decide q_list answering order
        ordered_q = []
        start_ind = []
        rest = set()
        for ind, q in enumerate(q_list):
            son_idxs = [int(x) for x in re.findall(r"\#(\d+)", q)]
            if len(son_idxs) == 0:
                ordered_q.append(q)
                start_ind.append(ind)
            else:
                rest.add(ind)

        def find_next(start_ind, rest, q_list):
            for ind in rest:
                q = q_list[ind]
                son_idxs = [int(x) for x in re.findall(r"\#(\d+)", q)]

                flag = True
                for son in son_idxs:
                    if son not in start_ind:
                        flag = False
                    
                if flag:
                    return ind, son_idxs
            return None, None

        next_id, sons = find_next(start_ind, rest, q_list)
        while next_id is not None:
            rest.remove(next_id)
            start_ind.append(next_id)
            sub_q = q_list[next_id]

            # change tag index
            for son in sons:
                for j, st in enumerate(start_ind):
                    if son == st:
                        sub_q.replace('#%s' % son, '#%s' % j)

            ordered_q.append(sub_q)
            next_id, sons = find_next(start_ind, rest, q_list)

        print("ordered_q:", ordered_q)
        q_tree = []
        # after the order is decided
        for ind, sub_q in enumerate(ordered_q):
            son_idxs = [int(x) for x in re.findall(r"\#(\d+)", sub_q)]
            hop = 1 + sum([ret[idx]["hop"] for idx in son_idxs])

            # tree_question
            # find all sons
            if len(son_idxs) > 0:
                all_sons = []
                son_idx = son_idxs
                while len(son_idx) > 0:
                    all_sons.extend(son_idx)
                    tmp = []
                    for son in son_idx:
                        tmp.extend(ret[son]["son_idxs"])
                    son_idx = tmp
                print("all_sons", all_sons)

                rest = [ i for i in range(ind) if i not in all_sons]
                print("rest", rest)
                if len(rest) == 0: # root node
                    tree_question = tree[0].q_list[0]
                else:
                    candidate = []
                    n = v
                    while n > 0: #and tree[n].parent is not None:
                        candidate.append(tree[n].parent)
                        n = tree[n].parent

                    # condition 1: exist in tree parent
                    tree_question = None
                    for c in candidate:
                        if len(tree[c].q_list) == len(rest) + 1:
                            #tmp_ind = list(range(len(tree[c].q_list)))
                            flag2 = True
                            for r in rest:
                                tmp_q = ordered_q[r]
                                if tmp_q not in tree[c].q_list:
                                    flag2 = False
                                    break
                            if flag2 is True:
                                for i in range(len(tree[c].q_list)):
                                    if tree[c].q_list[i] not in ordered_q:
                                        tree_question =  tree[c].q_list[i]

                    # condition 2: generate a parent
                    if len(son_idxs) >= 2: # query relation
                        tree_question = sub_q
                        for s in son_idxs:
                            sq = ret[s]["tree_question"]
                            tree_question = tree_question.replace("#%s"%s, sq)
                    elif len(son_idxs) == 1:
                        #son_q = [ordered_q[a] for a in sorted(all_sons)]
                        second_q = sub_q.replace("#%s" % son_idxs[0], "#1")
                        son_q = [ordered_q[son_idxs[0]], sub_q]
                        tree_question = merge_question(son_q)
            else: # leaf node
                tree_question = sub_q
    
            
            node = {
                "idx": ind, 
                "ent_question": sub_q,
                "son_idxs": son_idxs, 
                "tree_question": tree_question,
                "hop": hop,
            }
            ret.append(node)
            print(node)
            #input()
        return ret

    def santitize_preds(self, preds):
        pred2attr = {}
        for pred, score, support_contexts in preds:
            if (pred) not in pred2attr:
                pred2attr[(pred)] = (score, support_contexts)
            else:
                pred2attr[(pred)] = (max(score, pred2attr[(pred)][0], key=lambda x:x.avg()), support_contexts+pred2attr[(pred)][1])
        else:
            #排序
            preds = [(pred, score, list(set(support_contexts))) for (pred), (score, support_contexts) in pred2attr.items()]
            return sorted(preds, key=lambda x:x[1].avg(), reverse=True)

    def Bridge(self, node, memory, given_contexts=None):
        idx, question, ent_question, son_idxs, hop = node["idx"], node["question"], node["ent_question"], node["son_idxs"], node["hop"]
        preds = []
        assert len(son_idxs) == 1
        son_idx = son_idxs[0]
        for ent, last_score, _ in memory[son_idx]:
            question = ent_question.replace(f"#{son_idx}", ent)
            
            text_preds = self.singlehop_text_executor.predict(question, program, given_contexts)
            preds.extend((pred, last_score.merge(Avg(score)), support_contexts) for (pred, score, support_contexts) in text_preds)
        return preds
    
    def QueryRelation(self, node, memory, given_contexts=None):
        idx, question, ent_question, son_idxs, hop = node["idx"], node["question"], node["ent_question"], node["son_idxs"], node["hop"]
        preds = []
        assert len(son_idxs) == 2
        son_idx1, son_idx2 = son_idxs
        preds = []
        for ent1, score1, _ in memory[son_idx1]:
            for ent2, _, score2, _ in memory[son_idx2]:
                question = ent_question.replace(f"#{son_idx1}", ent1).replace(f"#{son_idx2}", ent2)
                #program, parse_score = self.kb_executor.parse(question)

                last_score = score1.merge(score2)
                
                text_preds = self.singlehop_text_executor.predict(question, program, given_contexts)
                preds.extend((pred, last_score.merge(Avg(score)), support_contexts) for (pred, score, support_contexts) in text_preds)
        return preds

if __name__=="__main__":
    import sys
    sys.path.append("/data/ljx/qd_mcts")
    from mcts_data_collection import get_args
    from data_structure import tree_list, node
    # args
    args = get_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # model
    model = rollout_model(args)

    # data
    raw_data = load_json_line(args.raw_data_path)
    raw_data = [d for d in raw_data if int(d['id'][0]) > 2]

    # build tree
    entry = raw_data[0]
    question = entry['question']
    answer = entry['answer']
    print("answer:", answer)
    context = [ p['title'] + '. ' + p['paragraph_text'] for p in entry['paragraphs']]
    cur_tree = tree_list(question, answer, context)

    #n1
    pid = 0

    node_id = 1
    q_list = ["Which religion is founded by the black community in the city that used to be the US capital", "Who started the Bethel branch of #0"]
    leaf = [False, True]
    new_node = node(q_list, pid, node_id, leaf)
    cur_tree.list[pid].child.append(node_id)
    cur_tree.list.append(new_node)

    node_id = 2
    q_list = ["What is the Bethel branch of the religion that is founded by the black community in the city that used to be the US capital", "Who started #0"]
    leaf = [False, True]
    new_node = node(q_list, pid, node_id, leaf)
    cur_tree.list[pid].child.append(node_id)
    cur_tree.list.append(new_node)

    node_id = 3
    q_list = ["What is the city that used to be the US capital", "Who started the Bethel branch of the religion that is founded by the black community in #0"]
    leaf = [True, False]
    new_node = node(q_list, pid, node_id, leaf)
    cur_tree.list[pid].child.append(node_id)
    cur_tree.list.append(new_node)

    node_id = 4
    q_list = ["the black community in the US old capital", "Who started the Bethel branch of the religion that is founded by #0"]
    leaf = [True, False]
    new_node = node(q_list, pid, node_id, leaf)
    cur_tree.list[pid].child.append(node_id)
    cur_tree.list.append(new_node)
    
    #cur_tree.iterate()
    n = 1
    cur_tree.show(1)
    model.inference(tree=cur_tree, node=n)

    '''
    q_list = [
        "a on #1",
        "b",
        "c on #3",
        "e on #0",
    ]
    o = order(q_list)
    print(o)
    '''