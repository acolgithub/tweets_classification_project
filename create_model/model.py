from parameters import Params

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



# # get deberta
# class Deberta_model(nn.Module):
#     def __init__(self, params: Params) -> None:
#         super(Deberta_model, self).__init__()
#         self.model = DebertaForSequenceClassification.from_pretrained(
#             "microsoft/deberta-base",               # base model
#             num_labels = params.n_output_labels,  # number of outputs
#             output_attentions = False,              # returns attention weights of all layers
#             output_hidden_states = False            # returns hidden states of all layers
#         )


#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         attention_masks: torch.Tensor,
#         target: Union[torch.FloatTensor, None]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
        
#         # if there is a target then return loss and prediction
#         if target != None:
#             output = self.deberta(
#                 input_ids=input_ids,
#                 token_type_ids=None,
#                 attention_mask=attention_masks,
#                 labels=target,
#                 return_dict=None
#             )
            
#             return output["loss"], output["logits"]
        
#         else:
#             output = self.deberta(
#                 intput_ids=input_ids,
#                 token_type_ids=None,
#                 attention_mask=attention_masks,
#                 labels=None,
#                 return_dict=None
#             )
            
#             return output["logits"]
    



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
                intput_ids=input_ids,
                token_type_ids=None,
                attention_mask=attention_masks,
                labels=None,
                return_dict=None
            )
            
            return output["logits"]


    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.AdamW,
        scheduler: get_linear_schedule_with_warmup,
        params: Params
    ) -> Tuple[float, float, float]:
        
        # time each training epoch
        t0 = time.time()

        # track accuracy, loss, and f1
        total_train_accuracy = 0
        total_train_loss = 0
        total_train_f1_score = 0
        
        # indicate training
        self.model.train()
        
        for X_batch, y_batch in dataloader:

            # zero gradient
            optimizer.zero_grad()

            # reshape data and targets for model
            tuple_ids = X_batch[:,:params.max_len]
            attention_masks = X_batch[:,params.max_len:]
            labels = y_batch.flatten().to(float)

            # add to device
            tuple_ids = tuple_ids.to(params.device)
            attention_masks = attention_masks.to(params.device)
            labels = labels.to(params.device)

            # get loss value and prediction
            loss, logits = self.model(tuple_ids, attention_mask=attention_masks, labels=labels).to_tuple()

            # add train loss
            total_train_loss += loss.item()

            # get prediction
            y_batch_pred = (torch.argmax(logits, axis=1).flatten() > 0.5).to(float)

            # detach computational graph, copy to cpu, make numpy array
            y_batch_pred = y_batch_pred.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            
            # compute training accuracy
            total_train_accuracy += np.sum(y_batch_pred == labels) / len(labels)

            # calculate weighted f1 score of prediction
            total_train_f1_score += f1_score(labels, y_batch_pred, average="weighted")

            # accumulate gradient
            loss.backward()

            # update parameters
            optimizer.step()

            # update learning rate
            scheduler.step()

        # gather data
        average_train_accuracy = total_train_accuracy / len(dataloader)
        average_train_loss = total_train_loss / len(dataloader)
        average_train_f1_score = total_train_f1_score / len(dataloader)

        # print results
        print(f"Average train accuracy: {average_train_accuracy}")
        print(f"Average train loss: {average_train_loss}")
        print(f"Average train f1 score: {average_train_f1_score}")
        print(f"Training time: {datetime.timedelta(seconds = time.time()-t0)}\n")

        return average_train_accuracy, average_train_loss, average_train_f1_score


    def test(
            self,
            dataloader: torch.utils.data.DataLoader,
            params: Params
    ) -> Tuple[float, float, float]:
        
        # track validation accuracy, validation loss, and f1
        total_val_accuracy = 0
        total_val_loss = 0
        total_val_f1_score = 0
        
        # indicate testing
        self.model.eval()
        
        # disable gradient computation and reduce memory consumption
        with torch.inference_mode():

            for X_val, y_val in dataloader:

                # reshape data and targets for model
                val_tuple_ids = X_val[:,:params.max_len]
                val_attention_masks = X_val[:,params.max_len:]
                val_labels = y_val.flatten().to(float)
                
                # add to device
                val_tuple_ids = val_tuple_ids.to(params.device)
                val_attention_masks = val_attention_masks.to(params.device)
                val_labels = val_labels.to(params.device)

                # get loss value and prediction
                val_loss, val_logits = self.model(val_tuple_ids, attention_mask=val_attention_masks, labels=val_labels).to_tuple()

                # add train loss
                total_val_loss += val_loss.item()

                # get prediction
                y_val_batch_pred = (torch.argmax(val_logits, axis=1).flatten() > 0.5).to(float)

                # detach computational graph, copy to cpu, make numpy array
                y_val_batch_pred = y_val_batch_pred.detach().cpu().numpy()
                val_labels = val_labels.detach().cpu().numpy()
                
                # calculate accuracy
                total_val_accuracy += np.sum(y_val_batch_pred == val_labels) / len(val_labels)

                # calculate weighted f1 score of prediction
                total_val_f1_score += f1_score(val_labels, y_val_batch_pred, average="weighted")
                
        # gather validation data
        average_val_accuracy = total_val_accuracy / len(dataloader)
        average_val_loss = total_val_loss / len(dataloader)
        average_val_f1_score = total_val_f1_score / len(dataloader)
        
        # print results
        print(f"Average validation accuracy: {average_val_accuracy}")
        print(f"Average validation loss: {average_val_loss}")
        print(f"Average validation f1 score: {average_val_f1_score}\n")
        
        return average_val_accuracy, average_val_loss, average_val_f1_score