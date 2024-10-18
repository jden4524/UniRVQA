import os
import torch
import sys
import csv, random
import torch.nn.functional as F

def dict_to_device(dictionary, device):
    for key in dictionary:
        dictionary[key] = dictionary[key].to(device)

def slice_tensor_in_dict(d, slice_obj):
    sliced_d = {k:v[slice_obj] for k,v in d.items()}
    return sliced_d

def keep_topk(*arg, k, k_max):
    to_return = []
    for d in arg:
        if isinstance(d,list):
            idx = [i + j for i in range(0, len(d), k_max) for j in range(k)]
            to_return.append([d[i] for i in idx])
        elif isinstance(d,torch.Tensor):
            idx = [i + j for i in range(0, len(d), k_max) for j in range(k)]
            to_return.append(d[idx])
        else:
            idx = [i + j for i in range(0, len(next(iter(d.values()))), k_max) for j in range(k)]
            to_return.append(slice_tensor_in_dict(d, idx))
    if len(to_return)==1:
        return to_return[0]
    return to_return

def pad_embeds_and_sequences(input_list, emb_layer):
    device = input_list[0][0].device
    longest = max([e.size(0)+s.size(0) for e,s in input_list])
    result = []
    mask = torch.ones((len(input_list),longest), dtype=torch.long)
    for i, (e,s) in enumerate(input_list):
        pad_len = longest - (e.size(0)+s.size(0))
        seq_embs = emb_layer(torch.cat([s, torch.zeros(pad_len,dtype=torch.long,device=device)]))
        result.append(torch.cat([e, seq_embs], dim=0))
        mask[i,-pad_len:]=0
    return torch.stack(result).to(device), mask.to(device)

def postprocess(result_path):
    import os
    import csv
    import re
    pattern = r'[^a-zA-Z0-9\s]'
    clean_file = result_path + ".clean"
    contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
							 "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
							 "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
							 "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
							 "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
							 "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
							 "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
							 "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
							 "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
							 "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
							 "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
							 "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
							 "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
							 "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
							 "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
							 "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
							 "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
							 "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
							 "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
							 "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
							 "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
							 "youll": "you'll", "youre": "you're", "youve": "you've"}
    manualMap    = { 'none': '0',
							  'zero': '0',
							  'one': '1',
							  'two': '2',
							  'three': '3',
							  'four': '4',
							  'five': '5',
							  'six': '6',
							  'seven': '7',
							  'eight': '8',
							  'nine': '9',
							  'ten': '10'
							}
    articles     = ['a',
							 'an',
							 'the'
							]
    periodStrip  = re.compile("(?!<=/d)(/.)(?!/d)")
    commaStrip   = re.compile("(/d)(/,)(/d)")
    punct        = [';', r"/", '[', ']', '"', '{', '}',
							 '(', ')', '=', '+', '//', '_', '-',
							 '>', '<', '@', '`', ',', '?', '!']


    with open(result_path, 'r') as infile, open(clean_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            #cleaned_text = re.sub(pattern, '', row[2])
            cleaned_text = row[1].replace('</s>', '')
            cleaned_text = cleaned_text.replace('<pad> ', '')
            cleaned_text = cleaned_text.replace('<pad>', '')
            cleaned_text = cleaned_text.strip()

            for p in punct:
                if (p + ' ' in cleaned_text or ' ' + p in cleaned_text) or (re.search(commaStrip, cleaned_text) != None):
                    cleaned_text = cleaned_text.replace(p, '')
                else:
                    cleaned_text = cleaned_text.replace(p, ' ')
            cleaned_text = periodStrip.sub("",
									  cleaned_text,
									  re.UNICODE)
                            
            outText=[]
            for word in cleaned_text.lower().split():
                if word in list(manualMap.keys()):
                    nword=manualMap[word]
                else:
                    nword = word

                if nword not in articles:
                    outText.append(word)
                else:
                    pass
            for wordId, word in enumerate(outText):
                if word in contractions:
                    outText[wordId] = contractions[word]
                    
            outText = ' '.join(outText)

            row[1]=outText
            writer.writerow(row)
    
def colbert_score_reduce(scores_padded, doc_mask):
    doc_padding = ~doc_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[doc_padding] = -65504
    scores = scores_padded.max(1).values
    #input_max, max_indices = torch.max(scores, dim=-1)
    return scores.sum(-1)


def colbert_score(query, doc_padded, doc_mask, use_gpu=False):
    """
    Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
    If Q.size(0) is 1, the matrix will be compared with all passages.
    Otherwise, each query matrix will be compared against the *aligned* passage.

    EVENTUALLY: Consider masking with -inf for the maxsim (or enforcing a ReLU).
    """
    if use_gpu:
        query, doc_padded, doc_mask = query.cuda(), doc_padded.cuda(), doc_mask.cuda()
    assert query.dim() == 3, query.size()
    assert doc_padded.dim() == 3, doc_padded.size()
    assert query.size(0) in [1, doc_padded.size(0)]

    scores = doc_padded @ query.to(dtype=doc_padded.dtype).permute(0, 2, 1)

    return colbert_score_reduce(scores, doc_mask)

def in_batch_loss(query, doc_padded, doc_mask):
    # query: batch_size x q_len x dim
    # doc_padded: batch_size x i_len x dim >>batch_size*n_docs x i_len x dim
    # doc_mask: batch_size x i_len x dim >>batch_size*n_docs x i_len x dim

    # currently each question only has one positive documents, might need to increase the number here.

    batch_size, i_len, dim = doc_padded.size()
    scores = (doc_padded.float().unsqueeze(0) @ query.float().permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)  # query-major unsqueeze
    
    scores = colbert_score_reduce(scores, doc_mask.repeat(query.size(0), 1, 1)) #batch X batch_doc
    

    in_batch_scores = scores.reshape(query.size(0), -1) # batch X documents in batch

    batch_size = query.shape[0]
    batch_size_doc = doc_padded.shape[0]
    num_pos_and_neg = batch_size_doc // batch_size

    # batch_size x dim  matmul  dim x (num_pos+num_neg)*batch_size
    # -->  batch_size x (num_pos+num_neg)*batch_size
    in_batch_labels = torch.zeros(batch_size, batch_size_doc).to(scores.device)
    step = num_pos_and_neg
    for i in range(batch_size):
        in_batch_labels[i, step * i] = 1

    in_batch_labels = torch.argmax(in_batch_labels, dim=1)
    loss_fn = torch.nn.CrossEntropyLoss()                           
    loss = loss_fn(in_batch_scores, in_batch_labels)

    return loss
    
def read_csv(file_path, delimiter=",", header = True):
    rows = []
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        if header:
            next(reader)
        for row in reader:
            rows.append(row)
    return rows

def load_corpus_dict(file_path, key_col, value_col, only_one=False):
    '''key_col & value_col can either be index number or string'''
    random.seed(42)
    csv.field_size_limit(sys.maxsize)
    documents = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if only_one:
                if (len(row[value_col])>2):
                    id_list = row[value_col][1:-1].split(',')
                    id_list = [int(num) for num in id_list]
                    documents[row[key_col]]=random.choice(id_list)
                else:
                    documents[row[key_col]]=[]
            else:
                documents[row[key_col]] = row[value_col]
    return documents

def filter_documents_with_keywords(docs, keywords):
    '''docs >> list; keywords >> set'''
    filtered_documents = []
    for doc in docs:
        # Check if any of the keywords are present in the document
        if any(keyword in docs for keyword in keywords):
            filtered_documents.append(doc)
    return filtered_documents

def cast(tensor_batch, device):
    yes =torch.tensor([ 0, 2163,1]).to(device)
    no = torch.tensor([0,465,1]).to(device)
    cast_results = torch.zeros(tensor_batch.shape[0])
    for idx, tensor in enumerate(tensor_batch):
        if torch.equal(yes, tensor):
            cast_results[idx] = 1
        elif torch.equal(no, tensor):
            cast_results[idx] = 0
        else:
            print('Discriminator generate problem!')
    return cast_results

def postprocess_retrieval_result(raw_result_path, result_name, k):
    '''This is function is used to merge multiple results into a row and mapping index id to passenge id (pid)'''
    result_path = os.path.join(os.path.dirname(raw_result_path),result_name)
    with open(raw_result_path, newline='') as csvfile, open(result_path, 'w', newline='') as outfile:
        reader = csv.reader(csvfile, delimiter='\t')
        writer = csv.writer(outfile)
        rows = list(reader)

        for i in range(0, len(rows), k):
            group = rows[i:i+k]
            doc_list =[]
            r_score = []
            for row in group:
                doc_list.append(row[1])    
                r_score.append(float(row[3]))

            r_score = F.log_softmax(torch.Tensor(r_score))
            
            output = [group[0][0], doc_list, r_score.tolist()]
            writer.writerow(output)
    return result_path

