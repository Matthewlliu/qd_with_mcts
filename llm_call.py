import zhipuai
from zhipuai import ZhipuAI
from termcolor import colored
from openai import OpenAI

def call_api(prompt, temperature=1):
    client = ZhipuAI(api_key="50aa6ea8cb7dec390f35329b8848584c.ZbyeInuAG7hKCExI")
    response = client.chat.completions.create(
        model="glm-4-plus",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    res = response.choices[0].message.content
    #print(response.choices[0].message.content)
    #input()
    return res

def call_api_gpt4o(prompt, temperature=1):
    client = OpenAI(
        api_key='sk-8BsfGhks9p4Ex0UfEb2277De91734aA0Bb5a1a1b26172fBb',
        base_url="https://svip.xty.app/v1",
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                #"content": "3.11和3.9哪个大",
                "content": prompt,
            }
        ],
        model="gpt-4o-2024-08-06",
        temperature=temperature,
    )
    res = chat_completion.choices[0].message.content
    return res

if __name__=='__main__':
    '''
    prompt = "Given a possibly multi-hop question, decompose it into TWO sub-questions. You may use [E] to represent the answer entity of one sub-question that appears in another. Give the answer only. \
        \nInput: {} \
        \nOutput:"
    question = [
        "When did Napoleon occupy the city where [E2]'s mother died",
        "Who brought Louis XVI style to the court?",
        "When did Napoleon occupy the city where the mother of the woman who brough Louis XVI style to the court died?",
        "How many Germans live in the colonial holding in Aruba's continent that was governed by Prazere's country?"
    ]
    #for q in question:
    print('\n')
    print("original question:", question[0])
    for i in range(5):
        q = question[0]
        p = prompt.format(q)
        #print(p)
        res = call_api(p)
        print('\n')
        print("output: ")
        print(colored(res, 'yellow'))
        input()
    '''
    prompt = "Given a natural language question, determine whether it should be decomposed by the hops needed to answer it. Don't give explanation. \
        \nExamples: \
        \n1. \
        \nInput: How many Germans live in France? \
        \nOutput: No. This question is one-hop. \
        \n2. \
        \nInput: When was David Cameron's first movie as a director released? \
        \nOutput: Yes. This question is two-hop. \
        \nTry this one: \
        \nInput: {} \
        \nOutput: "
    question = [
        #"When did Napoleon occupy the city where [E2]'s mother died",
        #"Who brought Louis XVI style to the court?",
        #"When did Napoleon occupy the city where the mother of the woman who brough Louis XVI style to the court died?",
        #"How many Germans live in France?",
        "When did WWII begin",
        "Which artist did Beyonce marry?",
        "What artist did Beyonce duet with in the single, \"Deja Vu''?"
    ]
    for q in question:
        print(q)
        p = prompt.format(q)
        res = call_api(p)
        print("output: ")
        print(colored(res, 'yellow'))