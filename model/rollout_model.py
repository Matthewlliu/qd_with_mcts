from tqdm import tqdm
import os
import numpy as np
import re
from torch import nn
import torch
from collections import defaultdict, Counter
from model.llm_executor import LLMReasoner
from model.retriever import Wikipedia_retriever
from termcolor import colored

import sys
sys.path.append("/data/ljx/qd_mcts")
from util import load_json_line

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
        
class Dummy_retriever():
    def __init__(self):
        pass

    def retrieve(self, query, k=10):
        return []

class rollout_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = args
        if args.use_context:
            self.retriever = Wikipedia_retriever()
        else:
            self.retriever = Dummy_retriever()
        
        # self.singlehop_text_executor = TextExecutor(args, mode='single-hop')
        # self.multihop_text_executor = TextExecutor(args, mode='multi-hop')

        # self.filter = RobertaFilter(args, mode='single-hop')

        self.glm_executor = LLMReasoner(model_name="glm")
        self.gpt_executor = LLMReasoner(model_name="gpt4o")
        self.gem_executor = LLMReasoner(model_name="gemini")

    def inference(self, tree, node, savepath_for_debug):
        q_list = tree.list[node].q_list
        root_q_id = tree.list[node].root_q_id

        # the simulation tree here is actually a topological sorted list of questions
        if not tree.list[node].simulation_tree is None:
            return tree.list[node].answer, tree.list[node].simulation_tree
        
        # topology sort the questions
        sorted_q_list = []
        unlocking = {}
        in_degree = []
        for ind, q in enumerate(q_list):
            son_idxs = [int(x)-1 for x in re.findall(r"\#(\d+)", q)]
            for son in son_idxs:
                # assert son + 1 != root_q_id
                if son not in unlocking:
                    unlocking[son] = []
                unlocking[son].append(ind)
            in_degree.append(len(son_idxs))

        # topological sort
        stack = []
        for i, d in enumerate(in_degree):
            if d == 0:
                stack.append(i)
        while stack:
            cur = stack.pop()
            sorted_q_list.append({"idx": cur+1, "question": q_list[cur]})
            if cur in unlocking:
                for son in unlocking[cur]:
                    in_degree[son] -= 1
                    if in_degree[son] == 0:
                        stack.append(son)

        # inference
        answer_memo = {}
        for sorted_idx, q in enumerate(sorted_q_list):    
            idx = q["idx"]
            question = q["question"]
            # replace the #idx with the answer
            for i, a in answer_memo.items():
                question = question.replace(f"#{i}", a)
            # predict
            text_pred = self.glm_executor.predict(question, savepath_for_debug)
            answer_memo[idx] = text_pred
            sorted_q_list[sorted_idx]["answer"] = text_pred
        # assert sorted_q_list[-1]["idx"] == root_q_id, f"The last question is not the root question:\n{sorted_q_list}\n{q_list}"
        return sorted_q_list[-1]["answer"], sorted_q_list

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