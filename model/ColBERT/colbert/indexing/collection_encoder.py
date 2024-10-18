import torch

from colbert.infra.run import Run
from colbert.utils.utils import print_message, batch
from tqdm import tqdm
from torch.utils.data import DataLoader
from colbert.parameters import DEVICE


class CollectionEncoder():
    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint
        self.use_gpu = self.config.total_visible_gpus > 0

    def encode_passages(self, passages):
        Run().print(f"#> Encoding {len(passages)} passages..")

        if len(passages) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []

            # Batch here to avoid OOM from storing intermediate embeddings on GPU.
            # Storing on the GPU helps with speed of masking, etc.
            # But ideally this batching happens internally inside docFromText.
            def collate_fn(batch):
                batch = ["<D> "+b for b in batch]
                tokenized = self.checkpoint.doc_tokenizer.tok(batch, padding=True, truncation=True, max_length= self.config.doc_maxlen, add_special_tokens=False, return_tensors='pt')
                return tokenized
            dl = DataLoader(passages, batch_size=self.config.index_bsize, shuffle=False, pin_memory=True, collate_fn=collate_fn)
            for batch in tqdm(dl):
                # batch["input_ids"] = batch["input_ids"].to(DEVICE)
                # batch["attention_mask"] = batch["attention_mask"].to(DEVICE)
                embs_, doclens_ = self.checkpoint.docFromText(batch)
                embs.append(embs_)
                doclens.extend(doclens_)
                
            # for offset, passages_batch in tqdm(batch(passages, self.config.index_bsize * 50, provide_offset=True), total=len(passages)//(self.config.index_bsize * 50)+1):
            #     embs_, doclens_ = self.checkpoint.docFromText(passages_batch, bsize=self.config.index_bsize,
            #                                                   keep_dims='flatten', showprogress=(not self.use_gpu))
            #     embs.append(embs_)
            #     doclens.extend(doclens_)
                # Run().print(offset)

            embs = torch.cat(embs)

            # embs, doclens = self.checkpoint.docFromText(passages, bsize=self.config.index_bsize,
            #                                                   keep_dims='flatten', showprogress=(self.config.rank < 1))

        # with torch.inference_mode():
        #     embs = self.checkpoint.docFromText(passages, bsize=self.config.index_bsize,
        #                                        keep_dims=False, showprogress=(self.config.rank < 1))
        #     assert type(embs) is list
        #     assert len(embs) == len(passages)

        #     doclens = [d.size(0) for d in embs]
        #     embs = torch.cat(embs)

        return embs, doclens
