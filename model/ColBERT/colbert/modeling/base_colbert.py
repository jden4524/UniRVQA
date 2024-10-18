import os
import torch
import sys
from peft import PeftModel, PeftConfig

from colbert.utils.utils import torch_load_dnn

from transformers import AutoProcessor
from model.mmRAG_blip2 import Blip2ForConditionalGeneration
from model.mmRAG_instructblip import InstructBlipForConditionalGeneration
# from colbert.modeling.hf_colbert import class_factory
from colbert.infra.config import ColBERTConfig

from colbert.parameters import DEVICE

class BaseColBERT(torch.nn.Module):
    """
    Shallow module that wraps the ColBERT parameters, custom configuration, and underlying tokenizer.
    This class provides direct instantiation and saving of the model/colbert_config/tokenizer package.

    Like HF, evaluation mode is the default.
    """

    def __init__(self, name_or_path, colbert_config=None):
        super().__init__()

        self.colbert_config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint(name_or_path), colbert_config)
        self.name = self.colbert_config.model_name#  or name_or_path
        
        if "blip2" in self.name:
            self.model = Blip2ForConditionalGeneration.from_pretrained(self.name, device_map="cuda", torch_dtype=torch.bfloat16) # load_in_8bit=True causes vit to produce nan
            
        elif "instruct" in self.name:
            self.model = InstructBlipForConditionalGeneration.from_pretrained(self.name, device_map="cuda", torch_dtype=torch.bfloat16)
        self.raw_tokenizer = AutoProcessor.from_pretrained(self.name).tokenizer
        self.raw_tokenizer.add_special_tokens({'additional_special_tokens':['<Q>', '<D>', '<C>']})
        self.model.resize_token_embeddings(len(self.raw_tokenizer))
        # config = PeftConfig.from_pretrained(name_or_path)
        self.model = PeftModel.from_pretrained(self.model, name_or_path)
        self.model = self.model.merge_and_unload(safe_merge=True)

        self.eval()

    @property
    def device(self):
        return self.model.device

    # def encode(self, **kwargs):
    #     return self.model.get_retrieval_embed(**kwargs)

    @property
    def score_scaler(self):
        return self.model.score_scaler

    def save(self, path):
        assert not path.endswith('.dnn'), f"{path}: We reserve *.dnn names for the deprecated checkpoint format."

        self.model.save_pretrained(path)
        self.raw_tokenizer.save_pretrained(path)

        self.colbert_config.save_for_checkpoint(path)
