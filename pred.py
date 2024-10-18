import json
from operator import itemgetter
from transformers import AutoProcessor
from model.mmRAG_instructblip import InstructBlipForConditionalGeneration
from model.mmRAG_blip2 import Blip2ForConditionalGeneration
from tqdm import tqdm
import torch, ast, argparse
from model.dataset import VQADataset, DataCollator
from torch.utils.data import DataLoader
from peft import PeftModel, PeftConfig
from misc.util import  *


'''
This file is used for generating answers for VQA using retrieved documents
'''
parser = argparse.ArgumentParser(description='Generate answers for VQA using retrieved documents')
parser.add_argument('-r','--retrieval_result_path', help='Path to retrieval (search) results')
parser.add_argument('-k','--k', help='use top k results', default=5)
parser.add_argument('-c','--checkpoint', help='model checkpoint', default=None)

args = vars(parser.parse_args())
dataset_name = args['retrieval_result_path'].split('/')[-1].split('_')[0]
model_name = args['retrieval_result_path'].split('/')[-1].split('_')[2]
if args['checkpoint'] is not None:
    cp_name = args['checkpoint']
else:
    cp_name = args['retrieval_result_path'].split('/')[-1].split('_index_')[1].split('_result_')[0]
k_max = int(args['retrieval_result_path'].split('@')[1].split('.')[0])
k = k_max if args['k'] == 0 else int(args['k'])
pred_file_name = os.path.join(os.path.dirname(args['retrieval_result_path']), f"{dataset_name}_pred_{cp_name}.jsonl")

processor = AutoProcessor.from_pretrained(f"Salesforce/{model_name}-flan-t5-xl")
processor.tokenizer.add_special_tokens({'additional_special_tokens':['<Q>', '<D>', '<C>']})
if model_name == 'blip2':
    model = Blip2ForConditionalGeneration.from_pretrained(f"Salesforce/{model_name}-flan-t5-xl", device_map='auto',torch_dtype=torch.bfloat16)
else:
    model = InstructBlipForConditionalGeneration.from_pretrained(f"Salesforce/{model_name}-flan-t5-xl", device_map='auto',torch_dtype=torch.bfloat16)

model.resize_token_embeddings(len(processor.tokenizer))
model = PeftModel.from_pretrained(model, f"assets/checkpoints/{cp_name}")   
model = model.merge_and_unload(safe_merge=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Create a list to store the results
results = []
batch_size = 2

rel_score = load_corpus_dict(args['retrieval_result_path'], 0, 2, only_one=False) 
testdataset= VQADataset(path=f'assets/data/{dataset_name}', split='val' , retrieval_result = args['retrieval_result_path'], model=model_name)
test_dataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=DataCollator(processor = processor,sample=False))

model.eval()

EM=0
VQA =0

pred_dict = []
for idx, batch in enumerate(tqdm(test_dataloader, desc='Generating Prediction: ...')):
    qid, query, doc, gt_ans_dict = itemgetter("qid", "query", "doc", "gt_ans_dict")(batch)
    dict_to_device(doc, device)
    dict_to_device(query, device)
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        qid, query, doc = keep_topk(qid, query, doc, k=k, k_max=k_max)
        sliced_query = slice_tensor_in_dict(query, slice(None, None, k))

        answers_ouputs, verify = model.generate_with_verify(**sliced_query, num_beams=3)
        
        generated_text_self = processor.batch_decode(answers_ouputs, skip_special_tokens=True)
        generated_verify = processor.batch_decode(verify, skip_special_tokens=True)
        # print(generated_text_self)
        # print(generated_verify)

        select_qid = []
        generated_ans = {}
        for i, verify in enumerate(generated_verify): 
            if verify == 'correct':
                generated_ans[qid[::k][i]] = generated_text_self[i]
            else:
                select_qid.append(qid[::k][i])

        # select the questions need RAG:
        if select_qid !=[]:
            new_batch_size=len(select_qid)
            indices = [i for i, value in enumerate(qid) if value in select_qid]
            
            sliced_query = slice_tensor_in_dict(query, indices)
            sliced_doc = slice_tensor_in_dict(doc, indices)
            
            generated_text_rag=model.generate(**sliced_query,
                                              doc_input_ids=sliced_doc['input_ids'], doc_attention_mask =sliced_doc['attention_mask'],    
                                              num_beams=3, return_dict_in_generate=True, output_scores=True)

            reshape_scores = generated_text_rag.sequences_scores.view(new_batch_size,k)
            reshape_sequences = generated_text_rag.sequences.view(new_batch_size,k, -1)
            
            rel_scores = torch.Tensor([ast.literal_eval(rel_score[str(qid)]) for qid in select_qid]).to(device)
            rel_scores = keep_topk(rel_scores.T, k=k, k_max=k_max).T

            joint_scores = reshape_scores+rel_scores
            
            _ , max_idx = torch.max(joint_scores, 1)

            ans_seq=[]
            for i in range(reshape_sequences.size(0)):
                selected_ans = reshape_sequences[i][max_idx[i].item()]
                ans_seq.append(selected_ans)
            
            generated_text_rag = processor.batch_decode(ans_seq, skip_special_tokens=True)
        

        #evaluation
        # if len(generated_ans ==[]:
        #     generated_ans=generated_text_rag
        # elif generated_ans !=[] and len(generated_ans)==batch_size:
        #     generated_ans = generated_ans
        # else:
        #     generated_ans+=generated_text_rag
        for i, qid_s in enumerate(select_qid):
            generated_ans[qid_s] = generated_text_rag[i]
        
        for i in range(len(generated_ans)):
            gt_ans = gt_ans_dict[i]
            pred_ans = generated_ans[qid[::k][i]]
            if isinstance(gt_ans,dict):
                gt_ans = dict(sorted(gt_ans.items(), key=lambda item: item[1], reverse=True))
                for ans in gt_ans.keys():
                    if ans.lower() in pred_ans.lower():
                        VQA+=gt_ans[ans]
                        EM+=1
                        break
            else:
                try:
                    pred_ans_num = float(pred_ans)
                    if (gt_ans[0]<=pred_ans_num<=gt_ans[1]) or (gt_ans[0]>=pred_ans_num>=gt_ans[1]):
                        VQA+=1
                        EM+=1
                except ValueError:
                    pass
        
            with open(pred_file_name, 'a') as file:
                entry = {"data_id": qid[::k][i], "prediction": pred_ans}
                print(json.dumps(entry), file=file)
               
        # if idx%100 ==0:
        #     print('Current EM = {}'.format(EM/((idx+1)*batch_size)))
        #     print('Current VQA = {}'.format(VQA/((idx+1)*batch_size)))

print('Current EM = {}'.format(EM/len(testdataset)))
print('Current VQA = {}'.format(VQA/len(testdataset)))