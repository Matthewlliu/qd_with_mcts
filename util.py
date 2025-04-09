import json
from llm_call import call_api_glm, call_api_gpt4o
import re
import random
from termcolor import colored
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

def valid_check(input, res):
    """
    Check if the two sub-questions are valid, i.e., one of them contains [E] and the other one doesn't. 
    All the #1, #2, etc. in the input should also appear in the output.
    @param res: a list of two sub-questions
    @return: True if the two sub-questions are valid, False otherwise
    """
    if len(res) != 2:
        return False
        
    if ('[E]' in res[0]) and ('[E]' in res[1]):
        return False
    if ('[E]' not in res[0]) and ('[E]' not in res[1]):
        return False

    input_idxs = set([int(x) for x in re.findall(r"\#(\d+)", input)])
    res_idxs = set([int(x) for x in re.findall(r"\#(\d+)", res[0])]) | set([int(x) for x in re.findall(r"\#(\d+)", res[1])])
    if input_idxs - res_idxs:
        return False
    
    return True

def soft_valid_check(input, decomposition):
    """
    Check if the two sub-questions contains exactly the information of the original question. 
    @param input: the original question
    @param decomposition: a list of two sub-questions
    @return: True if the two sub-questions are valid, False otherwise
    """

    prompt_for_merging_questions = """
    Given two sub-questions, you should merge them into a single question.
    The decomposition is trying to break down a more complex question into two sub-questions and therefore reasoning about the answer of the original question step by step.
    And your job is to reverse the process and get the original question from the two sub-questions.

    One of the the sub-questions will contain [E], while the other doesn't. 
    To get the merged question, you should somehow rephrase the sub-question that doesn't contain [E] and replace the [E] in the other sub-question with this rephrased sub-question.
    Most importantly, make sure the merged question is a natural question that a human can understand and don't miss any information or specification in the two sub-questions.

    Some questions in the input might contain #1, #2, etc. to represent the answer of the previous question. You should treat them as some known entities and keep them as is in the merged question.
    Here are some examples:
    
    Input:
    1. Who was the first secretary of #1 as of August 4, 2024?
    2. In what state was [E] born?
    Output: As of August 4, 2024, in what state was the first secretary of #1 born?

    Input:
    1. What is the first name of the 15th first lady of the United States' mother? 
    2. If my future wife's surname is the same as the second assassinated president's mother's maiden name, and her first name is [E], what is my future wife's name?
    Output: If my future wife's surname is the same as the second assassinated president's mother's maiden name, and her first name is the first name of the 15th first lady of the United States' mother, what is my future wife's name?

    You should just output the merged question and nothing else.
    
    Here is the input:
    1. {}
    2. {}
    Output:
    """

    prompt_for_checking_identity = """
    Given two questions, you should check whether they are asking about the same thing, i.e., including the same specifications for the answer.
    You should use a integer between 1 and 10 to rate how confident you are in that they are asking about the same thing, 
    where 1 means you are very confident that they are not asking about the same thing and 10 means you are very confident that they are asking about the same thing.
    Here are some examples:
    
    Input:
    1. If my future wife's surname is the same as the second assassinated president's mother's maiden name, and her first name is the first name of the 15th first lady of the United States' mother, what is my future wife's name?
    2. My future wife will have the same first name as the 15th first lady of the United States' mother, and her surname will be the same as the second assassinated president's mother's maiden name. What is my future wife's name?
    Output: 10
    
    Input:
    1. Who was the first secretary of #1 as of August 4, 2024?
    2. In what state was the first secretary of #1 born?
    Output: 1

    Input: 
    1. Where was the first president of #3 born?
    2. What is the birthplace of the first president of #1?
    Output: 1
    
    You should just output a single integer between 1 and 10 and nothing else.
    
    Here is the input:
    1. {}
    2. {}
    Output:
    """

    merging_prompt = prompt_for_merging_questions.format(decomposition[0], decomposition[1])
    merged_question = call_api_glm(merging_prompt, temperature=0)

    checking_prompt = prompt_for_checking_identity.format(input, merged_question)
    confidence = call_api_glm(checking_prompt, temperature=0)
    confidence = int(confidence.strip())
    return confidence >= 6 or (confidence // 2 >= random.randint(1, 10))

def subq_decompose(q, num_decompositions, do_soft_valid_check=False, max_attempts=10):
    """
    Decompose the question into two sub-questions. 
    @param q: the question to decompose
    @param num_decompositions: the number of decompositions
    @return: a list of decompositions
    """
    def post_process(input, r, ret):
        decompositions = re.split(r'Decomposition \d+:', r)
        for d in decompositions:
            if not d.strip():
                continue
            # 修改正则表达式以处理前导空格和多行
            matches = re.findall(r'\s*Q1:\s*(.*?)\s*\n\s*Q2:\s*(.*?)(?:\n|$)', d.strip(), re.DOTALL)
            if matches:
                subq1, subq2 = matches[0]
                if valid_check(input, [subq1.strip(), subq2.strip()]) and \
                (not do_soft_valid_check or soft_valid_check(input, [subq1.strip(), subq2.strip()])):
                    ret.append([subq1.strip(), subq2.strip()])
        return ret

    prompt = """
                Decompose the following question into TWO sub-questions, 
                such that the original question can be broken down into two sequential sub-questions. 
                You should use [E] to represent the answer entity of one sub-question that appears in another. 
                For the two sub-questions you generate, one shouldn't contain [E] and the other one should contain [E].
                By answering the one without [E] first and then the one with [E] (which will be replaced by the answer entity of the first sub-question), 
                one should be able to get the answer to the original question.
                Some questions in the input might contain #1, #2, etc. to represent the answer of the previous question. You should ignore them and keep them as is in the sub-questions.
                Give exactly {} different decompositions. 
                Follow the format below:

                Input:
                If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name?

                Output:
                Decomposition 1:
                Q1: What's the first name of the 15th first lady of the United States' mother?
                Q2: If my future wife's surname is the same as the second assassinated president's mother's maiden name, and her first name is [E], what is my future wife's name?
                
                Decomposition 2:
                Q1: Who's the second assassinated president of the United States?
                Q2: If my future wife's surname is the same as [E]'s mother's maiden name, and her first name is the same as the 15th first lady of the United States' mother's first name, what is my future wife's name?
                
                And so on...
                
                Question to decompose: {}
            """
    ret = []
    attempts = 0
    while len(ret) < num_decompositions and attempts < max_attempts:
        p = prompt.format(num_decompositions - len(ret), q)
        res = call_api_glm(p, 1)
        res = post_process(q, res, ret)
        attempts += 1
    if len(ret) < num_decompositions:
        print(colored("Warning: Only generated {}/{} decompositions for question: {}".format(len(ret), num_decompositions, q), 'red'))
    return ret

def llm_determine_leaf(q):
    """
    Determine whether a question is a simple one-hop question.
    @param q: the question to determine
    @return: True if the question is a simple one-hop question, False otherwise
    """
    def post_process(r):
        r = r.strip()
        if r.startswith("Yes"):
            return True
        else:
            return False

    prompt = """Given a natural language question, determine whether it is a atomic question that not be decomposed in a meaningful and straightforward way.
        Some questions in the input might contain #1, #2, etc. You should view them as some known and fixed entities.

        Examples: 
        1. 
        Input: How many Germans live in Paris in 2024? 
        Output: Yes. 
        Rationale: The question cannot be meaningfully decomposed.
        2. 
        Input: When was Yimou Zhang's first movie as a director released? 
        Output: No. 
        Rationale: The question can be meaningfully decomposed into two sub-questions; for example, "What is Yimou Zhang's first movie as a director?" and "When was it released?"
        3. 
        Input: Who founded the CS department of #1? 
        Output: Yes. 
        Rationale: This question cannot be meaningfully decomposed. It's asking the name of a person that is fully specified.

        You should just output "Yes" or "No" and nothing else. Don't give any explanation or reasoning.
        Input: {} 
        Output: """
    p = prompt.format(q)
    res = call_api_glm(p, temperature=0)
    res = post_process(res)
    return res

def check_repeating(res, pool):
    """
    Check if the two sub-questions are repeating.
    @param res: a list of two sub-questions
    @param pool: a list of sub-questions
    @return: True if the two sub-questions are repeating, False otherwise
    """
    a = res[0][1:-1]
    b = res[1][1:-1]
    for p in pool:
        if a in p[0] and b in p[1]:
            return True
    return False

def insert(origin_list, pos, res_list):
    # FIXME: this should not be exposed to the outside  
    """
    Replace the decomposed question at the specified position with the result list.
    @param origin_list: the original list
    @param pos: the index of the decomposed question to be replaced
    @param res_list: the result list
    @return: the list after replacement
    """
    out = origin_list[:pos] + res_list + origin_list[pos+1:]
    return out

def reorder_index(q_list, cand_pos, root_qid):
    """
    Replace the [E] in the list with the index of the question.
    @param q_list: the list of questions
    @param cand_pos: the index of the decomposed question
    @return: the list after replacement
    """
    # FIXME: this should not be exposed to the outside  
    new_root_qid = root_qid

    # HACK: where does the (\[E[0-9]\]) come from? 
    # p = r"(\[E[0-9]\])"
    # for ind, q in enumerate(q_list):
    #     match = re.findall(p, q)
    #     for m in match:
    #         q_list[ind] = q_list[ind].replace(m, '[E]')

    if '[E]' in q_list[cand_pos]:
        q_list[cand_pos] = q_list[cand_pos].replace('[E]', '#%s' % (cand_pos+2))
        new_pos = cand_pos
        if root_qid == cand_pos+1:
            new_root_qid = cand_pos+1
    else:
        q_list[cand_pos+1] = q_list[cand_pos+1].replace('[E]', '#%s' % (cand_pos+1))
        new_pos = cand_pos + 1
        if root_qid == cand_pos+1:
            new_root_qid = cand_pos+2

    if root_qid > cand_pos+1:
        new_root_qid = root_qid+1

    p = r"\#[0-9]"
    for pos, q in enumerate(q_list):
        if pos == cand_pos or pos == cand_pos+1:
            continue
        match = re.findall(p, q)
        for m in match:
            i = int(m[1:]) # index starting from 1
            if i < cand_pos+1 :
                new = '#%s' % i
            elif i == cand_pos+1 :
                new = '#%s' % (new_pos + 1)
            else:
                new = '#%s' % (i+1)
            q_list[pos] = q_list[pos].replace(m, new)
    
    return q_list, new_root_qid

if __name__=='__main__':
    #question = "When did Napoleon occupy the city where Loui XV's mother died"
    #question = "When was the institute that owned The Collegian founded?"
    question = "Who started the Bethel branch of the religion founded by the black community in the city that used to be the US capitol?"
    #question = [
    #    "Which institute that owned the Collegian?",
    #    "When was #1 founded?"
    #    ]
    #res = merge_question(question)
    print(colored("Testing question decomposition", 'yellow'))
    res = subq_decompose(question, 3)
    print(colored(question, 'red'))
    for r in res:
        print(colored(llm_determine_leaf(r[0]), 'green'), colored(r[0], 'blue'))
        print(colored(llm_determine_leaf(r[1]), 'green'), colored(r[1], 'blue'))
        print('-'*100)


    print(colored("Testing question insertion", 'yellow'))
    q_list = [
        "What is the maiden name of the second assassinated president of the United States' mother?",
        "If my future wife's surname is #1 and her first name is the same as the 15th first lady of the United States' mother's first name, what is my future wife's name?"
    ]
    res = [
        "Who is the second assassinated president of the United States?",
        "What is the maiden name of [E]'s mother?"
    ]
    q_list = insert(q_list, 0, res)
    print(q_list)
    res = reorder_index(q_list, 0, 2)
    print(res)