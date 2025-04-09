import os
from util import subq_decompose, llm_determine_leaf, check_repeating, reorder_index, valid_check
from util import insert
from model.rollout_model import rollout_model as rollout_model
import numpy as np
from termcolor import colored
import json
import pandas as pd
from tqdm import tqdm

class Node():
    def __init__(self, q_list, root_q_id, parent, self_id, if_leaf):
        self.q_list = q_list
        self.root_q_id = root_q_id
        self.score = 0
        self.child = []
        self.parent = parent
        self.simulation_tree = None
        self.id = self_id
        self.selected_times = 0
        self.if_leaf = if_leaf 
        self.answer = None
    def determine_leaf(self,):
        return self.if_leaf

    def get_child_score(self, p_id):
        return NotImplemented

    def add_child(self):
        return NotImplemented

    def get_child(self):
        return self.child

    def update(self, score):
        self.score += score #TODO
        self.selected_times += 1
        return 1
    
    def get_score(self):
        return self.score

    def score_adjust(self, score):
        pass

    def to_json_result(self):
        return {
            "id": self.id,
            "q_list": self.q_list,
            "root_q_id": self.root_q_id,
            "answer": self.answer,
            "score": self.score,
            "selected_times": self.selected_times,
            "parent": self.parent,
            "child": self.child,
            "if_leaf": self.if_leaf,
            "simulation_tree": self.simulation_tree
        }

class TreeList():
    def __init__(self, question, answer, max_layer=10):
        root_node = Node([question], 1, None, 0, [False])
        self.list = [root_node]
        self.rollout_times = 0
        self.answer = answer
        self.question = question
        self.max_layer = max_layer
  
    def get_child_score(self, pid):
        pnode = self.list[pid]
        children_id = pnode.get_child()
        score = {}
        for cid in children_id:
            s = self.list[cid].get_score()
            # HACK: what does this function do?
            s = self.list[cid].score_adjust(s)
            score[cid] = s
        return score

    def update_node(self, pid, score):
        pnode = self.list[pid]
        pnode.update(score)

    def expand(self, pid, total_expand_width, do_soft_valid_check):
        pnode = self.list[pid]
        leaf_cond = pnode.if_leaf
        cands = [ind for ind, tmp in enumerate(leaf_cond) if tmp is False]
        # randomly distribute the total expand_width to each candidate
        expand_width_list = [0] * len(cands)
        for _ in range(total_expand_width):
            expand_width_list[np.random.randint(0, len(cands))] += 1
        expand_id_list = [] # all expanded nodes

        for cand, expand_width in zip(cands, expand_width_list): # the index number of the candidate and the expand width
            if expand_width == 0:
                continue
            subq = pnode.q_list[cand]
            subqs = subq_decompose(subq, expand_width, do_soft_valid_check)
            for res in subqs:
                q_list = insert(pnode.q_list, cand, res)
                try:
                    q_list, root_q_id = reorder_index(q_list, cand, pnode.root_q_id)
                except Exception as e:
                    print(colored("Error: %s" % e, 'red'))
                    print(colored("subq: %s" % subq, 'red'))
                    print(colored("res: %s" % res, 'red'))
                    continue

                res_leaf = [llm_determine_leaf(r) for r in res]

                # new node
                leaf = insert(leaf_cond, cand, res_leaf)
                node_id = len(self.list)

                new_node = Node(q_list, root_q_id, pid, node_id, leaf)
                self.list[pid].child.append(node_id)
                self.list.append(new_node)

                expand_id_list.append(node_id)
        return expand_id_list

    def tree_policy(self, expand_width, do_soft_valid_check):
        """
        expand the tree
        """
        v = 0 # root node
        while not all(self.list[v].if_leaf) and len(self.list[v].q_list) < self.max_layer: # when dividable
            if len(self.list[v].child) == 0:
                cnode = self.expand(v, expand_width, do_soft_valid_check) # expand multiple nodes, and randomly return one
                return cnode
            else:
                v = self.select_best_child(v, self.list[v].child)
        return v

    def select_best_child(self, pid, children):
        N = self.list[pid].selected_times
        coe = 0.6

        score = {}
        for c in children:
            n = self.list[c].selected_times
            if n == 0:
                s = np.inf
            else:
                s = self.list[c].score / n + coe * np.sqrt(2*np.log(N) / n )
            score[c] = s
        sort_score = sorted(score.items(), key=lambda item: item[1])
        return sort_score[-1][0]

    def update_tree(self, v_l, delta):
        for v, d in zip(v_l, delta):
            self.list[v].update(d)
            parent = self.list[v].parent
            while parent is not None:
                self.list[parent].update(d)
                parent = self.list[parent].parent

    def show(self, pid):
        print("Node #%s" % pid)
        print("\tQuestion list:")
        for is_leaf, q in zip(self.list[pid].if_leaf, self.list[pid].q_list):
            print(colored("\t\t*" if is_leaf else "\t\t ", 'red'), colored("%s" % q, 'green'))
        print("\tChildren:", self.list[pid].child)
        print("\tScore / Times: %s / %s" % (self.list[pid].score, self.list[pid].selected_times) )
    
    def iterate(self, func):
        """
        Iterate through the tree and print the nodes in a breadth-first manner. 
        """
        root = [0]
        func(0)
        while len(root) > 0:
            next_r = []
            for r in root:
                for c in self.list[r].child:
                    next_r.append(c)
                    func(c)
            root = next_r
        

class simulation_model(object):
    def __init__(self, args):
        self.args = args
        # self.raw_data = load_json_line(self.args.raw_data_path)
        # # data sample
        # # HACK: what does this do?
        # self.raw_data = [d for d in self.raw_data if int(d['id'][0]) > 2]
        self.raw_data = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")
        print("length of raw data:", len(self.raw_data))
        self.do_soft_valid_check = args.do_soft_valid_check

        self.rollout_model = rollout_model(args) # TODO
        self.cur_tree = None
        self.his_tree = []
        self.savedir_for_debug = self.args.debug_save_path if self.args.save_llm_result_for_debug else None

    def simulate(self,):
        for exp_id, entry in tqdm(self.raw_data.iterrows()):
            question = entry['Prompt']
            answer = [entry["Answer"]]
            answer = [a.lower() for a in answer]
            error_times = 100

            
            if self.savedir_for_debug:
                savepath_for_debug = os.path.join(self.savedir_for_debug, f"{exp_id}.txt")
            else:
                savepath_for_debug = None
            
            self.cur_tree = TreeList(question, answer)
            self.his_tree = self.cur_tree.list
            while self.cur_tree.rollout_times < self.args.rollout_times:
                try:
                    self.run(savepath_for_debug)
                    self.his_tree = self.cur_tree.list
                except Exception as e:
                    print(colored("Error: %s" % e, 'red'))
                    self.cur_tree.list = self.his_tree
                    error_times -= 1
                    if error_times == 0:
                        break
                # self.run(savepath_for_debug)
                    
            if error_times == 0:
                with open(os.path.join(self.args.save_path, f"error_{exp_id}.json"), "w") as f:
                    json.dump([node.to_json_result() for node in self.cur_tree.list], f, indent=4)
            else:
                with open(os.path.join(self.args.save_path, f"tree_{exp_id}.json"), "w") as f:
                    json.dump([node.to_json_result() for node in self.cur_tree.list], f, indent=4)

            print(colored(exp_id, 'red'))
            for k,v in entry.items():
                if not "wikipedia" in k:
                    print(colored(k, 'red'), v)
            print("Score: ", colored(self.cur_tree.list[0].score, 'green'))
    
    def run(self, savepath_for_debug):
        # select and expand
        v_l = self.cur_tree.tree_policy(self.args.expand_width, self.do_soft_valid_check)
        # print("Tree policy: ", v_l)
        # print("Rollout times: ", self.cur_tree.rollout_times)

        # roll out
        if isinstance(v_l, int):
            v_l = [v_l]
        
        delta = []
        for v in v_l:
            res, q_tree = self.rollout_model.inference(tree=self.cur_tree, node=v, savepath_for_debug=savepath_for_debug) # TODO
            self.cur_tree.rollout_times += 1
            if self.cur_tree.list[v].simulation_tree is None:
                # print("root_q_id: ", self.cur_tree.list[v].root_q_id)
                # for q in q_tree:
                #     print(q)
                # print("-"*100)
                self.cur_tree.list[v].simulation_tree = q_tree
                self.cur_tree.list[v].answer = res
            delta.append(res) # TODO

        # back propagate
        score = [int(d.lower() in self.cur_tree.answer) for d in delta]
        # for v, d in zip(v_l, score):
        #     if d == 1:
        #         print(colored("Correct: %s" % self.cur_tree.list[v].q_list, 'green'))
        self.cur_tree.update_tree(v_l, score)

if __name__=='__main__':
    pass