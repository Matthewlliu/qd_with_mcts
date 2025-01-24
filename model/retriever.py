import pandas as pd
import zhipuai
import time
from tqdm import tqdm
#from pandas.core.frame import DataFrame
#from datasets import Dataset
import datasets

def load_dataset(path):
    #data=pd.read_csv(path,sep='\t')
    ids = []
    content = []
    count = 0
    with open(path) as f:
        for line in tqdm(f):
            if count == 0:
                count += 1
                continue
            l = line.split('\t')
            ids.append(l[0].strip())
            title = l[-1].strip()
            text = ' '.join(l[1:-1])
            con = title + ': ' + text
            #print(con)
            #input()
            content.append(con)
    data = {
        "id": ids,
        "content": content
    }
    df = pd.DataFrame(data=data)
    print("build datasets")
    data = datasets.Dataset.from_pandas(df)
    return data

if __name__=='__main__':
    path = '/data1/amy/WIKI_data/processed_tsv/atlas_202112_title_section+text.tsv'
    s = time.perf_counter()
    wiki_dataset = load_dataset(path)
    e = time.perf_counter()
    print("LOADING time: %ss" % (e-s))
    #print(len(data))
    #print(data[0:5])
    #wiki_dataset = Dataset.from_pandas(data=df)
    print(wiki_dataset)