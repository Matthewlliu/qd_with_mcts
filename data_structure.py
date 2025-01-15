import os
from util import subq_decompose, llm_determine_leaf, check_repeating, reorder_index, valid_check
from util import load_json_line, save_data, insert
from model.rollout_model import rollout_model as rollout_model
import numpy as np

class node():
    def __init__(self, q_list, parent, self_id, if_leaf):
        self.q_list = q_list
        self.score = 0
        self.child = []
        self.parent = parent
        self.simulation_tree = None
        self.id = self_id
        self.selected_times = 0
        self.if_leaf = if_leaf #self.determine_leaf()

    def determine_leaf(self,):
        return [False]
        pass

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

class tree_list(node):
    def __init__(self, question, answer, context, max_layer=4):
        root_node = node([question], None, 0, [False])
        self.list = [root_node]
        self.rollout_times = 0
        self.answer = answer
        self.question = question
        self.context = context
        self.max_layer = max_layer
    
    def get_child_score(self, pid):
        pnode = self.list[pid]
        children_id = pnode.get_child()
        score = {}
        for cid in children_id:
            s = self.list[cid].get_score()
            s = self.list[cid].score_adjust(s)
            score[cid] = s
        return score

    def update_node(self, pid, score):
        pnode = self.list[pid]
        pnode.update(score)

    def expand(self, pid, expand_width):
        pnode = self.list[pid]
        leaf_cond = pnode.if_leaf
        cands = [ind for ind, tmp in enumerate(leaf_cond) if tmp is False]
        expand_id_list = [] # all expanded nodes

        for cand in cands: # the index number 
            subq_pool = []
            subq = pnode.q_list[cand]
            print("expand on:", subq)
            for it in range(expand_width):
                res = subq_decompose(subq)
                #while len(res) != 2:
                while not valid_check(res):
                    res = subq_decompose(subq)
                #print(res)
                if not check_repeating(res, subq_pool):
                    subq_pool.append(res)
                    # reorder q index
                    #print(q_list)
                    q_list = insert(pnode.q_list, cand, res)
                    try:
                        q_list = reorder_index(q_list, cand)
                    except:
                        continue

                    res_leaf = [llm_determine_leaf(r) for r in res]

                    # new node
                    leaf = insert(leaf_cond, cand, res_leaf)
                    node_id = len(self.list)

                    new_node = node(q_list, pid, node_id, leaf)
                    self.list[pid].child.append(node_id)
                    self.list.append(new_node)

                    expand_id_list.append(node_id)
        return expand_id_list


    #def score_adjust(self, score):
    #    pass

    def tree_policy(self, expand_width):
        v = 0 # root node
        tree_depth = 0
        while all(self.list[v].if_leaf) is False and len(self.list[v].q_list) < self.max_layer: # when dividable
            if len(self.list[v].child) == 0:
                print("expand on node #%s" % v)
                self.show(v)
                cnode = self.expand(v, expand_width) # expand multiple nodes, and randomly return one
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
        #print(score)
        sort_score = sorted(score.items(), key=lambda item: item[1])
        #print(sort_score)
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
        print("\tQuestion list:", self.list[pid].q_list)
        print("\tif leaf:", self.list[pid].if_leaf)
        print("\tchildren:", self.list[pid].child)
        print("\tscore / times: %s / %s" % (self.list[pid].score, self.list[pid].selected_times) )

    def iterate(self,):
        root = [0]
        self.show(0)
        while len(root) > 0:
            next_r = []
            for r in root:
                for c in self.list[r].child:
                    next_r.append(c)
                    self.show(c)
            root = next_r
        

class simulation_model(object):
    def __init__(self, args):
        self.args = args
        self.raw_data = load_json_line(self.args.raw_data_path)
        # data sample
        self.raw_data = [d for d in self.raw_data if int(d['id'][0]) > 2]
        print(len(self.raw_data))
        #exit()

        self.rollout_model = rollout_model(args) #TODO
        self.cur_tree = None
        self.his_tree = []

    def simulate(self,):
        print("Starting from iteration #%s" % self.args.start_iter)
        for exp_id in range(self.args.start_iter, self.args.simulation_examples):
            entry = self.raw_data[exp_id]
            question = entry['question']
            answer = entry['answer']
            context = entry['paragraphs']
            #question = "When did Napoleon occupy the city where the mother of the woman who brough Louis XVI style to the court died?"
            print(question)
            print(answer)
            self.cur_tree = tree_list(question, answer, context)
            #for _ in self.args.rollout_times:
            while self.cur_tree.rollout_times < self.args.rollout_times:
                #try:
                self.run()
                #except:
                #    pass
            # save
    
    def run(self):
        # select and expand
        v_l = self.cur_tree.tree_policy(self.args.expand_width)
        print("Tree policy: ", v_l)
        #input()

        # roll out
        if isinstance(v_l, int):
            v_l = [v_l]
        
        delta = []
        for v in v_l:
            #q = self.cur_tree.list[v].q_list
            res, q_tree = self.rollout_model.inference(tree=self.cur_tree, node=v) # TODO
            self.cur_tree.rollout_times += 1
            if self.cur_tree.list[v].simulation_tree is None:
                self.cur_tree.list[v].simulation_tree = q_tree
            delta.append(res) # TODO

        # back propagate
        self.cur_tree.update_tree(v_l, delta)

        self.cur_tree.iterate()
        input()
        


if __name__=='__main__':
    pass