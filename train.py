import argparse
from model.mmRAG_instructblip import InstructBlipForConditionalGeneration
from model.mmRAG_blip2 import Blip2ForConditionalGeneration
import torch, transformers
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model.dataset import VQADataset, DataCollator
from misc.util import dict_to_device
from transformers.utils.logging import set_verbosity
set_verbosity(40)
import os
from peft import LoraConfig, get_peft_model
import wandb
from collections import defaultdict
from operator import itemgetter

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('-c','--checkpoint', help='Name of checkpoint to save')
parser.add_argument('-b','--bsize', help='Batch size', default=16)
parser.add_argument('-s','--step', help='Max steps', default=6000)
parser.add_argument('-v','--val_every', help='Steps between validation', default=200)
args = vars(parser.parse_args())
dataset_name = args['checkpoint'].split('_')[-1]
model_name = args['checkpoint'].split('_')[0]
data_path = f"assets/data/{dataset_name}"

wandb.login(key="b3851425eab9176831e112eaedf2b9adb2363c88", verify=True)
wandb.init(project="qrag", name=args['checkpoint'], dir="assets")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if model_name == 'blip2':
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-flan-t5-xl', torch_dtype=torch.bfloat16, device_map="auto")
elif model_name == 'instructblip':
    model = InstructBlipForConditionalGeneration.from_pretrained('Salesforce/instructblip-flan-t5-xl', torch_dtype=torch.bfloat16, device_map="auto")
processor = transformers.AutoProcessor.from_pretrained(f"Salesforce/{model_name}-flan-t5-xl")
processor.tokenizer.add_special_tokens({'additional_special_tokens':['<Q>', '<D>', '<C>']})
dataset = VQADataset(path=data_path, split='train' if dataset_name=='okvqa' else 'train_subset', model=model_name)
rand_gen = torch.Generator().manual_seed(42)
if dataset_name == 'infoseek':
    train = dataset
    val = VQADataset(path=data_path, split='val_2000', model=model_name)
else:
    train, val = random_split(dataset, [0.9, 0.1], generator=rand_gen)
dl_train = DataLoader(train, batch_size=int(args['bsize']), shuffle=True, pin_memory=True, num_workers=4, collate_fn=DataCollator(processor=processor, max_doc_len=256))
dl_val = DataLoader(val, batch_size=int(args['bsize']), shuffle=False, pin_memory=True, num_workers=4,collate_fn=DataCollator(processor=processor, max_doc_len=256))

model.resize_token_embeddings(len(processor.tokenizer))
peft_config = LoraConfig(
    use_dora=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=['q', 'k', 'v', 'o', 'wi_0', 'wi_1', 'wo','lm_head'], 
    modules_to_save=['retrieval_projection', 'embed_tokens']
)
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

for n,module in model.named_modules():
    if 'vision_model' in n: # or ('qformer' in n):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 100, int(args['step']))
# scaler = torch.cuda.amp.GradScaler() # needed only for fp16



step=0
while step < int(args['step']):
    model.train()
    for idx, batch in enumerate(pbar := tqdm(dl_train, desc="Training... ")):
        if step >= int(args['step']):
            break
        qid, query, doc, ans_input_ids, ans_full = itemgetter("qid", "query", "doc", "ans_input_ids", "ans_full")(batch)
        dict_to_device(query, device)
        dict_to_device(doc, device)
        ans_input_ids = ans_input_ids.to(device)
        optimizer.zero_grad()
        step+=1

        retrieval_loss, inputs_embeds, attention_mask= model.retrieval_loss(**query, doc_input_ids=doc['input_ids'], doc_attention_mask = doc['attention_mask'])
        (retrieval_loss).backward()
        loss_dict = {"retrieval_loss": retrieval_loss}
        if step >= 100: # warm-up steps for language model generation.
            query.pop("attention_mask")
            ra_gen_loss = model.gen_loss(**query, attention_mask = attention_mask, inputs_embeds=inputs_embeds, doc_input_ids=doc['input_ids'], doc_attention_mask = doc['attention_mask'], labels = ans_input_ids)
            (ra_gen_loss).backward()
            nodoc_gen_loss, logits = model.gen_loss(**query, attention_mask = attention_mask, inputs_embeds=inputs_embeds, labels = ans_input_ids, return_logits = True)
            (nodoc_gen_loss).backward()
            loss_dict["ra_gen_loss"] = ra_gen_loss
            loss_dict["nodoc_gen_loss"] = nodoc_gen_loss
        
        if step >= 200:
            self_verify_loss = model.self_verify_loss(**query, ans_full=ans_full, logits = logits, attention_mask = attention_mask, inputs_embeds=inputs_embeds, labels = ans_input_ids)
            (self_verify_loss).backward()
            loss_dict["self_verify_loss"] = self_verify_loss

        wandb.log({"train/"+k:v.item() for k,v in loss_dict.items()}, step=step)

        optimizer.step()
        scheduler.step()

        if step % args['val_every'] == 0:  
            model.eval()
            with torch.no_grad():
                val_loss_dict = defaultdict(int)

                for idx, batch in enumerate(pbar := tqdm(dl_val, desc="Validating... ")):
                    qid, query, doc, ans_input_ids, ans_full = itemgetter("qid", "query", "doc", "ans_input_ids", "ans_full")(batch)
                    dict_to_device(query, device)
                    dict_to_device(doc, device)
                    ans_input_ids = ans_input_ids.to(device)

                    retrieval_loss, inputs_embeds, attention_mask= model.retrieval_loss(**query, doc_input_ids=doc['input_ids'], doc_attention_mask = doc['attention_mask'])
                    val_loss_dict["retrieval_loss"] += retrieval_loss.item()
                    if step >= 100: # warm-up steps for language model generation.
                        query.pop("attention_mask")
                        ra_gen_loss = model.gen_loss(**query, attention_mask = attention_mask, inputs_embeds=inputs_embeds, doc_input_ids=doc['input_ids'], doc_attention_mask = doc['attention_mask'], labels = ans_input_ids)
                        nodoc_gen_loss, logits = model.gen_loss(**query, attention_mask = attention_mask, inputs_embeds=inputs_embeds, labels = ans_input_ids, return_logits = True)
                        val_loss_dict["ra_gen_loss"] += ra_gen_loss.item()
                        val_loss_dict["nodoc_gen_loss"] += nodoc_gen_loss.item()

                    if step >= 150:
                        self_verify_loss = model.self_verify_loss(**query, ans_full=ans_full,logits = logits, attention_mask = attention_mask, inputs_embeds=inputs_embeds, labels = ans_input_ids)
                        val_loss_dict["self_verify_loss"] += self_verify_loss.item()


                wandb.log({"val/"+k:v/len(dl_val) for k,v in val_loss_dict.items()}, step=step)


            model.save_pretrained(f"assets/checkpoints/{args['checkpoint']}_{step}", from_pt=True)
