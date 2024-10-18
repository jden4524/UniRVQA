import transformers, os, random, json, ast
from PIL import Image
from torch.utils.data import Dataset
from misc.util import read_csv

random.seed(42)


class VQADataset(Dataset):
    def __init__(self, path, split, retrieval_result = None, model='blip2'):
        self.path = path
        self.split = split
        self.dataset_name = 'okvqa' if 'okvqa' in self.path else 'infoseek'
        self.retrieval_result = retrieval_result
        if self.dataset_name == "okvqa":
            self.data = json.load(open(os.path.join(self.path, f'google_corpus/retriever_{split}.json'), "r"))
            self.data = [i for i in self.data if len(i["ctxs"])>0]
            self.answer = json.load(open(os.path.join(self.path, f'mscoco_{split}2014_annotations.json')))["annotations"]
            self.answer = {i["question_id"]: [j["answer"] for j in i["answers"]] for i in self.answer}
        if self.dataset_name == "infoseek":
            with open(os.path.join(self.path, f'infoseek_{split}.jsonl'), "r") as f:
                self.data = [json.loads(line) for line in f]
            self.answer = {i["data_id"]: i["answer"] for i in self.data}
            # disable for caption gen
            caption_split = "val" if "val" in split else split
            with open(os.path.join(self.path, f'infoseek_{caption_split}_caption.json'), "r") as f:
                self.caption = json.load(f)
            self.corpus = {row[2]:row[1] for row in read_csv(os.path.join(self.path, f'wiki_100k_short.csv') ,header=True)}
        
        if self.retrieval_result: 
            self.retrieval_result = read_csv(retrieval_result, header=False)
            self.retrieval_result = {row[0]: ast.literal_eval(row[1]) for row in self.retrieval_result} # qid:pid
            corpus_path = os.path.join(self.path, f'google_corpus/okvqa_full_clean_corpus.csv') if self.dataset_name == "okvqa" else os.path.join(self.path, f'wiki_100k_short.csv') 
            self.corpus = {row[0]:row[1] for row in read_csv(corpus_path,header=True)}
        self.model = model
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.dataset_name == "okvqa":
            qid = sample["question_id"] 
            img_path = os.path.join(self.path, f"{self.split}2014/{sample['img_id']}.jpg")
            caption = sample['caption']
            ans = sample['answers']
        if self.dataset_name == "infoseek":
            qid = sample["data_id"]
            img_path = os.path.join(self.path, f"OVEN_images/{self.split}/{sample['image_id']}.jpg")
            # disable for caption gen
            # caption = ''
            caption = self.caption[qid]
            if isinstance(sample['answer_eval'], list):
                if isinstance(sample['answer_eval'][0], str): 
                    ans = {i.lower():1 for i in sample['answer_eval']}
                else: # numerical
                    ans = sample['answer_eval'][0]["range"]
            else:
                ans = sample['answer_eval']["range"]

        query_str = f"<Q> {caption} <C> Question: {sample['question']} Short Answer: "
        img = Image.open(img_path)
        
        if self.retrieval_result: # return provided doc and all ans
            docs = ["<D> " + self.corpus[pid] for pid in self.retrieval_result[str(qid)]]
            num_of_doc = len(docs)
            dict_to_return = {"qid": [qid]*num_of_doc,
                                "img": [img]*num_of_doc,
                                "query": [query_str]*num_of_doc,
                                "doc": docs,
                                "ans": [ans]
                                }
        
        else: # random sampling doc and ans
            if self.dataset_name == "okvqa":
                doc = "<D> " + random.choice(sample["ctxs"])["text"]
            
                # popular_ans = [key for key, value in sample['answers'].items() if value == max(sample['answers'].values())]
                ans = random.choice(self.answer[qid])
                # for a in self.answer[qid]:
                #     if a in doc:
                #         ans = a
            if self.dataset_name == "infoseek":
                if 'entity_id' in sample: # training
                    doc = "<D> " + self.corpus[sample['entity_id']]
                    ans = self.answer[qid][0]
                else: # doesn't need doc and ans for search
                    doc = ''
                    ans = ''
                
            
            dict_to_return = {"qid": qid,
                                "img": img,
                                "query": query_str,
                                "doc": doc,
                                "ans": ans,
                                'ans_full': self.answer[qid]}
        if self.model != 'blip2':
            question_str = f"Question: {sample['question']} Short Answer: "
            if self.retrieval_result:
                dict_to_return['question']=[question_str]*num_of_doc
            else:
                dict_to_return['question']=question_str                  
        return dict_to_return
        

class DataCollator:
    """
        Data collator that will dynamically pad the inputs
    """
    def __init__(self, processor= transformers.Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl"), sample=True ,max_doc_len = 256):
        self.processor = processor
        self.sample = sample
        self.max_doc_len = max_doc_len

    def __call__(self, batch):
        qid = [b["qid"] for b in batch]
        img = [b["img"] for b in batch]
        query = [b["query"] for b in batch]
        doc = [b["doc"] for b in batch]
        ans = [b['ans'] for b in batch]
        
        if self.sample: # only return ans_input_ids
            ans = self.processor.tokenizer(ans, padding=True, return_tensors='pt')
            ans = ans.pop("input_ids")
            ans[ans == self.processor.tokenizer.pad_token_id] = -100
            ans_full = [self.processor.tokenizer(b['ans_full'], padding=True,return_tensors='pt')['input_ids'] for b in batch]

        else:
            qid = sum(qid, [])
            img = sum(img, [])
            query = sum(query, [])
            doc = sum(doc, [])
            ans = sum(ans, [])
        
        query = self.processor(img, query, padding=True, truncation=True, max_length= 256, return_tensors="pt")
        doc = self.processor.tokenizer(doc, padding=True, truncation=True, max_length= self.max_doc_len, add_special_tokens=False, return_tensors='pt')
        to_return ={"qid": qid, "query": query, "doc": doc}
        
        if self.sample:
            to_return["ans_full"] = ans_full
            to_return["ans_input_ids"] = ans
        else:
            to_return["gt_ans_dict"] = ans
        
        if 'question' in batch[0].keys():
            question = [b['question'] for b in batch]
            if not self.sample:
                question = sum(question, [])
            question = self.processor.qformer_tokenizer(question, padding=True, truncation=True, max_length= 256, return_tensors='pt')
            to_return["query"]['qformer_input_ids'] = question['input_ids']
            to_return["query"]['qformer_attention_mask'] = question['attention_mask']
        return to_return
