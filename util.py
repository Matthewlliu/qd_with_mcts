import json
from llm_call import call_api, call_api_gpt4o
import re

def collect_data():
    pass

def save_data():
    pass

def load_json_line(raw_data_path):
    data = []
    with open(raw_data_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data

def subq_decompose(q):
    def post_process(r):
        r = r.split('\n')
        tmp = []
        for rr in r:
            if '1:' in rr:
                rr = rr.split('1:')[1]
            elif '2:' in rr:
                rr = rr.split('2:')[1]
            elif rr.startswith('1.'):
                rr = rr.split('1.')[1]
            elif rr.startswith('2.'):
                rr = rr.split('2.')[1]
            tmp.append(rr.strip())
        return tmp

    prompt = "Given a possibly multi-hop question, decompose it into TWO sub-questions. You may use [E] to represent the answer entity of one sub-question that appears in another. Give 4 different ways of decomposing, not just paraphrase. \
        \nInput: {} \
        \nOutput:"
    p = prompt.format(q)
    res = call_api(p, 2)
    #res = call_api_gpt4o(p, 2)
    #print(res)
    #res = post_process(res)
    return res

def merge_question(q):
    def post_process(r):
        r = r.strip()
        return r

    prompt = "Given two sub-questions, merge them into a whole question. The #1 represents the answer of the 1st sub question. \
        \nFor example: \
        \nInput: \
        \n1. The country whose capital is Rome? \
        \n2. When did Napoleon conquer #1? \
        \nOutput: \
        \nWhen did Napoleon conquer the country whose capital is Rome? \
        \nTry this one: \
        \nInput: \
        \n1. {} \
        \n2. {} \
        \nOutput:"

    p = prompt.format(q[0], q[1])
    res = call_api(p, temperature=0)
    res = post_process(res)
    return res


def llm_determine_leaf(q):
    def post_process(r):
        r = r.strip()
        if r.startswith("No"):
            return True
        else:
            return False

    prompt = "Given a natural language question, determine whether it should be decomposed by the hops needed to answer it. Don't give explanation. \
        \nExamples: \
        \n1. \
        \nInput: How many Germans live in [E]? \
        \nOutput: No. This question is one-hop. \
        \n2. \
        \nInput: When was [E]'s first movie as a director released? \
        \nOutput: Yes. This question is two-hop. \
        \nTry this one: \
        \nInput: {} \
        \nOutput: "
    p = prompt.format(q)
    res = call_api(p, temperature=0)
    res = post_process(res)
    return res

def check_repeating(res, pool):
    for p in pool:
        a = res[0][1:-1]
        b = res[1][1:-1]
        if a in p[0] and b in p[1]:
            return True
    return False

def insert(origin_list, pos, res_list):
    out = origin_list[:pos] + res_list + origin_list[pos+1:]
    return out

def reorder_index(q_list, cand_pos):
    p = r"(\[E[0-9]\])"
    for ind, q in enumerate(q_list):
        match = re.findall(p, q)
        for m in match:
            q_list[ind] = q_list[ind].replace(m, '[E]')

    if '[E]' in q_list[cand_pos]:
        q_list[cand_pos] = q_list[cand_pos].replace('[E]', '#%s' % (cand_pos+1))
        new_index = cand_pos
    elif '[E]' in q_list[cand_pos+1]:
        q_list[cand_pos+1] = q_list[cand_pos+1].replace('[E]', '#%s' % (cand_pos))
        new_index = cand_pos + 1
    #print(q_list)

    p = r"\#[0-9]"
    for ind, q in enumerate(q_list):
        if ind == cand_pos or ind == cand_pos+1:
            continue
        match = re.findall(p, q)
        for m in match:
            i = int(m[1:])
            if i < cand_pos :
                new = '#%s' % i
            elif i == cand_pos :
                new = '#%s' % new_index
            else:
                new = '#%s' % (i+1)
            q_list[ind] = q_list[ind].replace(m, new)
    return q_list

def valid_check(res):
    if len(res) != 2:
        return False
        
    if ('[E' in res[0]) and ('[E' in res[1]):
        return False
    if ('[E' not in res[0]) and ('[E' not in res[1]):
        return False
    return True

if __name__=='__main__':
    #question = "When did Napoleon occupy the city where Loui XV's mother died"
    #question = "When was the institute that owned The Collegian founded?"
    question = "Who started the Bethel branch of the religion founded by the black community in the city that used to be the US capitol?"
    #question = [
    #    "Which institute that owned the Collegian?",
    #    "When was #1 founded?"
    #    ]
    #res = merge_question(question)
    res = subq_decompose(question)
    print(question)
    print(res)

    '''
    q_list = [
        "who brought Louis XVI style to the court",
        "When did Napoleon occupy the city where #1's mother died"
    ]
    res = [
        "When did Napoleon occupy [E]?",
        "Where did #1's mother die?"
    ]
    q_list = insert(q_list, 1, res)
    print(q_list)
    res = reorder_index(q_list, 1)
    print(res)
    '''