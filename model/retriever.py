import pandas as pd
import zhipuai
from zhipuai import ZhipuAI
import time
from tqdm import tqdm
#from pandas.core.frame import DataFrame
#from datasets import Dataset
import datasets
import numpy as np
import os
import math
from multiprocessing import Pool
#import multiprocessing
import faiss
import pickle
import json

def get_embeddings(query, dim=1024):
    #client = ZhipuAI(api_key="9e12872b9b314a138c19db0a674deac6.YT3cCTU4Lrqsgglo")
    client = ZhipuAI(api_key="4078bda64bf14bf588ec5e10bad0b645.BeM0YmYkZDCLDuGz")
    response = client.embeddings.create(
        model="embedding-2", #填写需要调用的模型编码
        input=[
            query
        ],
        #dimensions=dim,
    )
    emd = np.array(response.data[0].embedding)
    emd = np.float32(emd)
    return emd

def load_dataset(path, num=33180000):
    #data=pd.read_csv(path,sep='\t')
    ids = []
    content = []
    count = 0
    with open(path) as f:
        for line in tqdm(f):
            l = line.split('\t')
            ids.append(l[0].strip())
            title = l[-1].strip()
            text = ' '.join(l[1:-1])
            con = title + ': ' + text
            #print(con)
            #input()
            content.append(con)
            count += 1
            if count == num:
                break
    data = {
        "id": ids,
        "content": content
    }
    df = pd.DataFrame(data=data)
    #print("build datasets")
    data = datasets.Dataset.from_pandas(df)
    return data

def batched_func(rank, data, emd_path, worker):    
    print('Process rank {} PID {} begin... data num: {}'.format(rank, os.getpid(), len(data)))
    path = os.path.join(emd_path, 'rank_%s.hf' % rank)
    if len(data) == 0:
        print("Overworked")
        return 0 
    elif os.path.exists(path):
        print("Existed")
        return 0
    else:
        data = data.map(
            lambda x: {"embeddings": get_embeddings(x["content"])})

        #path = os.path.join(emd_path, 'rank_%s.hf' % rank)
        data.save_to_disk(path)

def build_dataset_embedding():
    path = '/data/ljx/wikipedia/atlas_202112_title_section+text.tsv'
    emd_path = '/data/ljx/wikipedia/embedding/full_set'
    sam_num = 33176581

    wiki_dataset = load_dataset(path, num=sam_num)
    print(wiki_dataset)
    split = 20

    start_split = 0
    gap = math.ceil(sam_num/split)
    #for sp in range(start_split, split):
    for sp in [14, 15, 16, 17, 18, 19]:
        #sp = 13
        print("Work on data split {}".format(sp))
        sub_path = os.path.join(emd_path, 'split_%s' % sp)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        start = sp * gap
        end = min((sp+1) * gap, len(wiki_dataset))
        sub_dataset = wiki_dataset[start : end]
        sub_dataset = datasets.Dataset.from_dict(sub_dataset)

        print(sub_dataset)
        #input()

        worker = 50
        #print("?")

        # subsets
        subsets = []
        for i in range(worker):
            j = math.ceil(len(sub_dataset)/worker)
            end = min(len(sub_dataset), (i + 1) * j)
            data = sub_dataset[int(i * j) : int(end)]
            data = datasets.Dataset.from_dict(data)
            subsets.append(data)
        print("Subsets built")

        #del wiki_dataset

        s = time.perf_counter()
        
        
        p = Pool(worker)
        for i in range(worker):
            p.apply_async(batched_func, (i, subsets[i], sub_path, worker))
        p.close()
        p.join()
        
        '''
        for i in range(worker):
            batched_func(i, subsets[i], sub_path, worker)
        '''

        e = time.perf_counter()
        print("Embedding time: %ss" % (e-s))

def load_dataset_embedding(root_path, output_path):
    if os.path.exists(output_path):
        dataset = datasets.load_from_disk(os.path.join(output_path, 'dataset.hf'))
        with open(os.path.join(output_path, 'emds.pkl'), 'r') as f:
            emds = pickle.load(f)
    else:
        #splits = []
        dataset = []
        emds = []
        for f in os.listdir(root_path):
            sp = os.path.join(root_path, f)
            if f.startswith('split_0') and (not os.path.isfile(sp)) :
                print(sp)
                #ranks = []
                for r in tqdm(os.listdir(sp)):
                    if r.endswith('.hf'):
                        rank = datasets.load_from_disk(os.path.join(sp, r))
                        #data = rank.remove_columns(["embeddings"])
                        for ddd in rank:
                            dataset.append({"id": ddd["id"], "content": ddd["content"]})
                        #emd = rank["embeddings"]
                        #print(data)
                        #print(len(emd))
                        #print(type(emd[0]))
                        #print(type(emd[0][0]))
                        #exit()
                        emds.extend(rank["embeddings"])
                        #ranks.append(data)
                        del rank
                #splits.append(datasets.concatenate_datasets(ranks))
        #dataset = datasets.concatenate_datasets(splits)
        #dataset.save_to_disk(output_path)
        with open(os.path.join(output_path, 'dataset.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
        with open(os.path.join(output_path, 'emds.pkl'), 'wb') as f:
            pickle.dump(emds, f)
    return dataset, emds

def dump_embedding_split(root_path, index_list=[]):
    splits_file = ['split_%s' % i for i in index_list]
    for f in os.listdir(root_path):
        sp = os.path.join(root_path, f)
        if f in splits_file and (not os.path.isfile(sp)):
            print(sp)
            ds_path = os.path.join(sp, 'ds.hf')
            emd_path = os.path.join(sp, 'emd.pkl')
            print(ds_path)
            print(emd_path)

            emds = []
            ranks = []
            for r in tqdm(os.listdir(sp)):
                if r.startswith('rank_'):
                    rank = datasets.load_from_disk(os.path.join(sp, r))
                    data = rank.remove_columns(["embeddings"])
                    emds.extend(rank["embeddings"])
                    del rank
                    ranks.append(data)
            dataset = datasets.concatenate_datasets(ranks)
            dataset.save_to_disk(ds_path)

            emds = np.array(emds, dtype=np.float32)
            with open(emd_path, 'wb') as f:
                pickle.dump(emds, f)

def gather_emd_splits(root_path, index_list, output_path):
    splits_file = ['split_%s' % i for i in index_list]
    emds = None
    for sf in splits_file:
        sp = os.path.join(root_path, sf)
        emd_path = os.path.join(sp, 'emd.pkl')
        with open(emd_path, 'rb') as f:
            emd = pickle.load(f)
        print(emd.shape)
        if emds is None:
            emds = emd
        else:
            emds = np.concatenate((emds, emd), axis=0)

    of = os.path.join(output_path, 'all_emds.pkl')
    with open(of, 'wb') as f:
        pickle.dump(emds, f)

def gather_ds_splits(root_path, index_list, output_path):
    '''
    ds_path =os.path.join(root_path, 'all_data.hf')
    ds = datasets.load_from_disk(ds_path)
    new_ds = ds.remove_columns(['embeddings'])
    del ds
    
    of = os.path.join(output_path, 'dataset.hf')
    new_ds.save_to_disk(of)
    '''
    splits_file = ['split_%s' % i for i in index_list]
    dataset = []
    for sf in tqdm(splits_file):
        sp = os.path.join(root_path, sf)
        ds_path = os.path.join(sp, 'ds.hf')
        ds_split = datasets.load_from_disk(ds_path)
        dataset.append(ds_split)
    dataset = datasets.concatenate_datasets(dataset)
    of = os.path.join(output_path, 'dataset.hf')
    dataset.save_to_disk(of)

class Wikipedia_retriever():
    def __init__(self, root_path='/data1/ljx/wikipedia_embedding', nlist=256, nprobe=16):
        self.dataset, self.emds = self.load(root_path)
        #self.dataset.add_faiss_index(column="embeddings")
        #print(self.dataset)
        #print(self.emds.shape)
        #print(self.emds.dtype)
        self.faiss_index = self.index(nl=nlist, npr=nprobe)
        print("Finished loading retriever")

    def load(self, root):
        path = os.path.join(root, 'all_data_emd_split')
        dataset = datasets.load_from_disk(os.path.join(path, 'dataset.hf'))
        with open(os.path.join(path, 'all_emds.pkl'), 'rb') as f:
            emds = pickle.load(f)
        return dataset, emds

    def index(self, nl, npr):
        dim = 1024
        quantizer = faiss.IndexFlatL2(dim)
        faiss_index = faiss.IndexIVFFlat(quantizer, dim, nl, faiss.METRIC_L2)
        faiss_index.train(self.emds)
        faiss_index.add(self.emds)
        faiss_index.nprobe = npr
        return faiss_index

    def retrieve(self, query, k=3):
        emd = get_embeddings(query)
        #s = time.perf_counter()
        #scores, samples = self.dataset.get_nearest_examples(
        #    "embeddings", emd, k=k
        #)
        distances, indices = self.faiss_index.search(np.array([emd], dtype=np.float32), k)
        #e = time.perf_counter()
        #print("Retrieval time: %ss" % (e-s))
        #return e-s
        ret = []
        for i, index in enumerate(indices[0]):
            ret.append(self.dataset[int(index)]['content'])
        #samples_df = pd.DataFrame.from_dict(samples)
        #samples_df["scores"] = scores
        #samples_df.sort_values("scores", ascending=False, inplace=True)
        #for _, row in samples_df.iterrows():
        #    print(f"Content:  {row.content}")
        return ret
    
if __name__=='__main__':
    
    path = '/data/ljx/wikipedia/atlas_202112_title_section+text.tsv'
    emd_path = '/data/ljx/wikipedia/embedding/test_100'
    sam_num = 100

    root_path = '/data1/ljx/wikipedia_embedding'
    emd_path = "/data1/ljx/wikipedia_embedding/all_data_emd_split" #os.path.join(root_path, 'all_data.hf')
    retriever = Wikipedia_retriever(root_path)
    
    #dump_embedding_split(root_path, list(range(15,20)))
    #gather_emd_splits(root_path, list(range(20)), emd_path)
    #gather_ds_splits(root_path, list(range(20)), emd_path)
    #dataset, emds = load_dataset_embedding(root_path, emd_path)
    #input()
    #exit()
    #print(type(retriever.dataset[0]['embeddings']))
    
    num_vectors = len(retriever.dataset)
    dim = 1024 #len(all_emd[0])
    
    '''
    query = ["Which programme made its first broadcast on 28 July 1940, consisting of a speech by Queen Wilhelmina",
             "What is the name of Batman's mother",
             "What was the original name of Beatles the rock band?"
        ]
    '''
    query_path = "/data/ljx/musique/dev_test_singlehop_questions_v1.0.json"
    tmp = json.load(open(query_path))
    query = [q["question"] for q in tmp["natural_questions"]]
    query = query[:30]
    emds = [ get_embeddings(q) for q in query ]

    """
    #faiss_index = faiss.IndexFlatIP(dim)
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(retriever.emds)
    #exit()
    """
    with open("/data/ljx/qd_with_mcts/result/low_rank_faiss_res30.json", 'rb') as f:
        low_rank_res = pickle.load(f)
    
    nlist = [1024]#[64, 128, 256, 512]
    nprobe = [4, 8 ,16]#[2, 4, 8, 16, 32, 64]
    quantizer = faiss.IndexFlatL2(dim)

    all_result = {}
    all_score = {}
    for nl in nlist:
        print("Start training index...")
        faiss_index = faiss.IndexIVFFlat(quantizer, dim, nl, faiss.METRIC_L2)
        faiss_index.train(retriever.emds)
        faiss_index.add(retriever.emds)
        print("Built...")
        for npr in nprobe:
            if nl == npr:
                continue
            faiss_index.nprobe = npr
            print("nl: %s, np: %s" % (nl, npr) )
    
            times = []
            scores = []
            if (nl, npr) not in all_result:
                all_result[(nl, npr)] = []
            for ind, emd in enumerate(emds):
                #print(q)
                #emd = get_embeddings(q)
                k = 3
                s = time.perf_counter()
                distances, indices = faiss_index.search(np.array([emd], dtype=np.float32), k)
                e = time.perf_counter()
                #print("Retrieval time: %ss" % (e-s))
                times.append(e-s)
                res = indices[0]
                #result.append(res)

                all_result[(nl, npr)].append(res)
                l_res = low_rank_res[ind]
                score = len(set(res).intersection(set(l_res)))/len(set(res).union(set(l_res)))
                scores.append(score)
                #for i, index in enumerate(indices[0]):
                    #distance = distances[0][i]
                    #print(f"Nearest neighbor {i+1}: {retriever.dataset[int(index)]['content']}, Distance {distance}")
            print("Average time: %s " % np.average(times))
            print("Average score: %s " % np.average(scores))
            all_score[(nl, npr)] = np.average(scores)
    res_path = "/data/ljx/qd_with_mcts/result/high_rank_search.pkl"
    with open(res_path, 'wb') as f:
        pickle.dump(all_result, f)

    sorted_score = sorted(all_score.items(), key = lambda x: x[1], reverse=True)
    print(sorted_score[0])

    '''
    for k in [1, 3, 5]:
        t = []
        for _ in range(100):
            tt = retriever.retrieve(query)
            t.append(tt)
        print(np.average(t))
    '''

    '''
    wiki_dataset = load_dataset(path, num=sam_num)
    print(wiki_dataset)
    worker = 16

    subsets = []
    for i in range(worker):
        j = math.ceil(len(wiki_dataset)/worker)
        end = min(len(wiki_dataset), (i + 1) * j)
        data = wiki_dataset[int(i * j) : int(end)]
        data = datasets.Dataset.from_dict(data)
        subsets.append(data)

    p = Pool(worker)
    #print("?")
    for i in range(worker):
        p.apply_async(batched_func, args=(i, subsets[i], emd_path, worker))
    p.close()
    p.join()
    '''
    #root = '/data/ljx/wikipedia/embedding/full_set'
    #data_emd_path = '/data/ljx/wikipedia/embedding/full_set/all_data.hf'
    #dataset = load_dataset_embedding(root, data_emd_path)
    #print(dataset)
    #exit()
    #build_dataset_embedding()

    '''
    count = 0
    word = 0
    with open(path) as f:
        for line in tqdm(f):
            if count < sam_num:
                l = line.split('\t')
                #ids.append(l[0].strip())
                title = l[-1].strip()
                text = ' '.join(l[1:-1])
                con = title + ' ' + text
                word += len(con.split())
                count += 1
            else:
                break
    print("word number:", word)
    print("Token number:", round(word * 4/3))
    '''

    '''
    data_file = '/data/ljx/wikipedia/wiki_dataset_num%s.hf' % sam_num
    if os.path.exists(data_file):
        wiki_dataset =datasets.load_from_disk(data_file)
    else:
        wiki_dataset = load_dataset(path, num=sam_num)#num=3317658)

        s = time.perf_counter()
        wiki_dataset = wiki_dataset.map(
            lambda x: {"embeddings": get_embeddings(x["content"])})
        e = time.perf_counter()
        print("Embedding time: %ss" % (e-s))

        wiki_dataset.save_to_disk(data_file)
    
    #query = "Who was the mother of Napoleon"
    query = "Which programme made its first broadcast on 28 July 1940, consisting of a speech by Queen Wilhelmina"
    emd = get_embeddings(query)
    #print(emd.dtype)
    #print(type(emd))
    #exit()
    s = time.perf_counter()
    wiki_dataset.add_faiss_index(column="embeddings")
    e = time.perf_counter()
    print("Adding index time: %ss" % (e-s))
    #print(len(data))
    #print(data[0:5])
    #wiki_dataset = Dataset.from_pandas(data=df)
    print(wiki_dataset)

    s = time.perf_counter()
    scores, samples = wiki_dataset.get_nearest_examples(
        "embeddings", emd, k=5
    )
    e = time.perf_counter()
    print("Retrieval time: %ss" % (e-s))

    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)
    for _, row in samples_df.iterrows():
        print(f"Content:  {row.content}")
    '''

    '''
    slices = '/data/ljx/wikipedia/wiki_slice_%s'
    data = []
    for i in range(10):
        with open(slices % i, 'r') as f:
            tmp = f.readlines()
            #print(tmp[0])
            #exit()
            data.extend(tmp)
    print(len(data))
    with open(path, 'w') as f:
        for line in data:
            f.write(line)
    '''