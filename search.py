import argparse
from operator import itemgetter
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from tqdm.auto import tqdm
import torch, os
from model.dataset import VQADataset, DataCollator
from torch.utils.data import DataLoader
from misc.eval import PRR, wiki_recall
from misc.util import dict_to_device, postprocess_retrieval_result
import torch.nn.functional as F
from transformers import AutoProcessor

parser = argparse.ArgumentParser(description='Search documents')
parser.add_argument('-i','--index', help='peft Checkpoint')
args = vars(parser.parse_args())
dataset_name = args['index'].split('_')[0]
dataset_path = os.path.join('assets/data', dataset_name)
model_name = args['index'].split('_')[2]
'''
This file is used for testdataset label generation with any pre-trained models and fine-tuned models, including: Pre-trained BLIP, BLIP2 and fine-tuned BLIP, BLIP2
'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create a list to store the results
batch_size = 96

dataset= VQADataset(path=dataset_path, split='val', model=model_name)
processor = AutoProcessor.from_pretrained(f"Salesforce/{model_name}-flan-t5-xl")
processor.tokenizer.add_special_tokens({'additional_special_tokens':['<Q>', '<D>', '<C>']})
dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=DataCollator(processor = processor))

ks = [1, 5, 10]

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment=".", root="assets/experiments", name=args['index'])):

        searcher = Searcher(index=f"{args['index']}")

        torch.cuda.empty_cache()
        for idx, batch in enumerate(tqdm(dl, desc="Searching... ")):
            qid, query = itemgetter("qid", "query")(batch)
            dict_to_device(query, device)
            ranking = searcher.search_all(query, k=max(ks), qid_batch=qid)
            tmp_path = ranking.save(f"tmp_@{max(ks)}.tsv")
        
        result_path = postprocess_retrieval_result(raw_result_path=tmp_path,result_name=f"{args['index']}_result_@{max(ks)}.csv", k=max(ks))
        print(f'results saved at {result_path}')
        for k in ks:
            hit, total = PRR(k, result_path, data_path=dataset_path)
            print(f'PRR@{k}: {hit/total} = {hit}/{total}')
            if dataset_name == 'infoseek':
                hit, total = wiki_recall(k, result_path, data_path=dataset_path)
                print(f'wiki_recall@{k}: {hit/total} = {hit}/{total}')