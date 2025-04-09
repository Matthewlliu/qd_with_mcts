import sys
sys.path.append("/home/lhw/qd_with_mcts")
from llm_call import call_api_glm, call_api_gpt4o, call_gemini
import time
from termcolor import colored
PROMPT = """Answer the following quesiton with only a short and concise answer(entity). The given context might include some information about the answer, 
if not, use your own knowledge. 
The result should be exactly in the following format and nothing else, do not provide any other information or explanation:

Input:
Question: Who is the president of the United States?
Context: No context provided

Output:
Donald Trump

Now answer the question:
Question: %s\n\
Context: %s\n\
Answer: """

class LLMReasoner():
    def __init__(self, model_name):
        if model_name == 'gpt4o':
            self.call_api = call_api_gpt4o
        elif model_name == 'gemini':
            self.call_api = call_gemini
        elif model_name == 'glm':
            self.call_api = call_api_glm
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def predict(self, query, context=[], savepath_for_debug=None):
        con = ' '.join([ "%s. " % (ind+1) + c  for ind, c in enumerate(context)]) if context else "No context provided"
        p = PROMPT % (query, con)
        resp = self.call_api(p)
        ans = self.postprocess(resp)
        if savepath_for_debug:
            with open(savepath_for_debug, "a") as f:
                f.write(f"{query}\n{con}\n{ans}\n\n")
        return ans

    def postprocess(self, res):
        return res.strip()

if __name__=='__main__':
    glmr = LLMReasoner('glm')
    #gptr = GPTReasoner(None)
    gemr = LLMReasoner('gemini')
    q = 'What religion was founded by the black community in the city that used to be the US capitol?'
    c = ['Religion of black Americans: Urban churches:  time when 50 cents a day was good pay for unskilled physical labor. Increasingly the Methodists reached out to college or seminary graduates for their ministers, but most of Baptists felt that education was a negative factor that undercut the intense religiosity and oratorical skills they demanded of their ministers. After 1910, as black people migrated to major cities in both the North and the South, there emerged the pattern of a few very large churches with thousands of members and a paid staff, headed by an influential preacher. At the same time there were many "storefront" churches with a few dozen members.', 'Religion of black Americans: Pentecostalism:  Giggie finds that black Methodists and Baptists sought middle class respectability. In sharp contrast the new Holiness Pentecostalism, in addition to the Holiness Methodist belief in entire sanctification, which was based on a sudden religious experience that could empower people to avoid sin, also taught a belief in a third work of grace accompanied by glossolalia. These groups stressed the role of the direct witness of the Holy Spirit, and emphasized the traditional emotionalism of black worship. William J. Seymour, a black preacher, traveled to Los Angeles where his preaching sparked the three-year-long Azusa Street Revival in 1906. Worship at the racially integrated Azusa ']
    
    print("glm")
    s = time.perf_counter()
    r = glmr.predict(q, c)
    e = time.perf_counter()
    print(r)
    print("time: %ss" % (e-s))

    print("gemini")
    s = time.perf_counter()
    r = gemr.predict(q, c)
    e = time.perf_counter()
    print(r)
    print("time: %ss" % (e-s))