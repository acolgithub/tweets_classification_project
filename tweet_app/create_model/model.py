from create_model.parameters import Params

import time
import datetime
import numpy as np
from typing import Union, Tuple

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import DebertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from transformers import logging


logging.set_verbosity_error() 

# create model
class Make_model(nn.Module):
    
    def __init__(self, params: Params) -> None:
        super(Make_model, self).__init__()
        self.model = DebertaForSequenceClassification.from_pretrained(
            "microsoft/deberta-base",               # base model
            num_labels = params.n_output_labels,  # number of outputs
            output_attentions = False,              # returns attention weights of all layers
            output_hidden_states = False            # returns hidden states of all layers
        )


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        target: Union[torch.FloatTensor, None]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # if there is a target then return loss and prediction
        if target != None:
            output = self.model(
                input_ids=input_ids,
                token_type_ids=None,
                attention_mask=attention_masks,
                labels=target,
                return_dict=None
            )
            
            return output["loss"], output["logits"]
        
        else:
            output = self.model(
                input_ids=input_ids,
                token_type_ids=None,
                attention_mask=attention_masks,
                labels=None,
                return_dict=None
            )
            
            return output["logits"]
