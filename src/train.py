import sklearn
import pandas as pd
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange,tnrange,tqdm_notebook
import random

from utils import Emotion_dict

def train_model(model, args, train_dataloader, valid_dataloader, test_dataloader):
    
    num_warmup_steps = int(0.05*args.train_length)
    num_training_steps = len(train_dataloader)*args.epochs

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
    
    train_logs = []
    valid_logs = []
    test_logs  = []

    model.zero_grad()

    for _ in tnrange(1, args.epochs+1, desc='Epoch'):
        print("<" + "="*22 + F" Epoch {_} "+ "="*22 + ">")
        # Calculate total loss for this epoch
        batch_loss = 0
        
        train_accuracy, nb_train_steps = 0, 0
        
        pred_list = np.array([])
        labels_list = np.array([])
        
        for step, batch in enumerate(train_dataloader):
            # Set our model to training mode (as opposed to evaluation mode)
            model.train()
            
            # Add batch to GPU
            batch = tuple(t.cuda(args.device) for t in batch)
            b_input_ids_1, b_input_ids_2, b_input_ids_3, b_personality, b_init_emo, b_response_emo, b_labels = batch
            logits = model(b_input_ids_1, b_input_ids_2, b_input_ids_3, personality=b_personality, init_emo=b_init_emo)
            
            if args.loss_function == 'MSE':
                loss_fct     = nn.MSELoss()
                loss         = loss_fct(logits, b_response_emo)
                pred_emotion = logits.detach().to('cpu').numpy()
                pred_flat    = vad_to_emo(pred_emotion, Emotion_dict).flatten()
                labels_flat  = vad_to_emo(b_response_emo.to('cpu'), Emotion_dict).flatten()
            else:
                if args.loss_function == 'CE': # cross entropy:
                    loss_fct = nn.CrossEntropyLoss()
                elif args.loss_function == 'Focal': # Focal loss:
                    loss_fct = FocalLoss()
                
                loss        = loss_fct(logits, b_labels)
                logits      = logits.detach().to('cpu').numpy()
                label_ids   = b_labels.to('cpu').numpy()
                pred_flat   = np.argmax(logits, axis=1).flatten()
                labels_flat = label_ids.flatten()
            
            
            
            pred_list   = np.append(pred_list, pred_flat)
            labels_list = np.append(labels_list, labels_flat)
            df_metrics  = pd.DataFrame({'Epoch':args.epochs,'Actual_class':labels_flat,'Predicted_class':pred_flat})
           

            nb_train_steps += 1

            # Backward pass
            loss.backward()
            
            # Clip the norm of the gradients to 1.0
            # Gradient clipping is not in AdamW anymore
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            
            # Update learning rate schedule
            scheduler.step()
      
            # Clear the previous accumulated gradients
            optimizer.zero_grad()
            
            # Update tracking variables
            batch_loss += loss.item()
      
        #  Calculate the average loss over the training data.
        avg_train_loss = batch_loss / len(train_dataloader)

        #store the current learning rate
        for param_group in optimizer.param_groups:
            print("\n\tCurrent Learning rate: ",param_group['lr'])
        
        result = classification_report(pred_list, labels_list, digits=4, output_dict=True)
        for key in result.keys():
            if key !='accuracy':
                try:
                    train_logs.append([
                        labelencoder.classes_[int(eval(key))], 
                        result[key]['precision'], 
                        result[key]['recall'], 
                        result[key]['f1-score'], 
                        result[key]['support'] 
                    ])
                except:
                    train_logs.append([
                        key, 
                        result[key]['precision'], 
                        result[key]['recall'], 
                        result[key]['f1-score'], 
                        result[key]['support'] 
                    ])
        

        valid_logs = eval_model(model, valid_dataloader, args, valid_logs)
        test_logs  = test_model(model, test_dataloader, args, test_logs)

    df_train_logs = pd.DataFrame(train_logs, columns=['label', 'precision', 'recall', 'f1-score', 'support']).add_prefix('train_')
    df_valid_logs = pd.DataFrame(valid_logs, columns=['precision', 'recall', 'f1-score', 'support']).add_prefix('valid_')
    df_test_logs  = pd.DataFrame(test_logs, columns=['precision', 'recall', 'f1-score', 'support']).add_prefix('test_')


    df_all = pd.concat([df_train_logs, df_valid_logs, df_test_logs], axis=1)
    df_all.to_csv(args.result_name, index=False)



def vad_to_emo(emotion, Emotion_dict):
    label_list = []
    for emo in list(emotion):
        min_index = 0
        min_mse = 1000
        cnt = 0
        for k,v in Emotion_dict.items():
            mse = sklearn.metrics.mean_squared_error(list(emo), v)
            if mse < min_mse:
                min_mse = mse
                min_index = cnt
            cnt += 1
        label_list.append(min_index)
    return np.array(label_list)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
 
    def forward(self, output, target):
        # convert output to pseudo probability
        out_target = torch.stack([output[i, t] for i, t in enumerate(target)])
        probs = torch.sigmoid(out_target)
        focal_weight = torch.pow(1-probs, self.gamma)
 
        # add focal weight to cross entropy
        ce_loss = F.cross_entropy(output, target, weight=self.weight, reduction='none')
        focal_loss = focal_weight * ce_loss
 
        if self.reduction == 'mean':
            focal_loss = (focal_loss/focal_weight.sum()).sum()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
 
        return focal_loss


def eval_model(model, valid_dataloader, args, valid_logs):
    # Validation

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()
    
    pred_list = np.array([])
    labels_list = np.array([])
    
    for batch in valid_dataloader:
        batch = tuple(t.cuda(args.device) for t in batch)
        b_input_ids_1, b_input_ids_2, b_input_ids_3, b_personality, b_init_emo, b_response_emo, b_labels = batch
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          logits = model(b_input_ids_1, b_input_ids_2, b_input_ids_3, personality=b_personality, init_emo=b_init_emo)
        
        # if MSE:
        # loss_fct = nn.MSELoss()
        # loss = loss_fct(logits, b_response_emo)
        
        # if cross entropy:
        loss_fct = nn.CrossEntropyLoss()
        # loss_fct = FocalLoss()
        loss = loss_fct(logits, b_labels)
        logits = logits.detach().to('cpu').numpy()
        label_ids = b_labels.to('cpu').numpy()
  
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        
        # if MSE:
        # pred_emotion = logits.detach().to('cpu').numpy()
        # pred_flat = vad_to_emo(pred_emotion, Emotion_dict).flatten()
        # labels_flat = vad_to_emo(b_response_emo.to('cpu'), Emotion_dict).flatten()
        

        pred_list = np.append(pred_list, pred_flat)
        labels_list = np.append(labels_list, labels_flat)
    result = classification_report(pred_list, labels_list, digits=4, output_dict=True)
    for key in result.keys():
        if key !='accuracy':
            valid_logs.append([
                    result[key]['precision'], 
                    result[key]['recall'], 
                    result[key]['f1-score'], 
                    result[key]['support'] 
                ])
    return valid_logs


def test_model(model, test_dataloader, args, test_logs):
    # Test

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()
    
    pred_list = np.array([])
    labels_list = np.array([])
    
    for batch in test_dataloader:
        batch = tuple(t.cuda(args.device) for t in batch)
        b_input_ids_1, b_input_ids_2, b_input_ids_3, b_personality, b_init_emo, b_response_emo, b_labels = batch
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          logits = model(b_input_ids_1, b_input_ids_2, b_input_ids_3, personality=b_personality, init_emo=b_init_emo)
        
        # if MSE:
        # loss_fct = nn.MSELoss()
        # loss = loss_fct(logits, b_response_emo)
        
        # if cross entropy:
        loss_fct = nn.CrossEntropyLoss()
        
        # loss_fct = FocalLoss()
        
        loss = loss_fct(logits, b_labels)
        logits = logits.detach().to('cpu').numpy()
        label_ids = b_labels.to('cpu').numpy()
  
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        
        # if MSE:
        # pred_emotion = logits.detach().to('cpu').numpy()
        # pred_flat = vad_to_emo(pred_emotion, Emotion_dict).flatten()
        # labels_flat = vad_to_emo(b_response_emo.to('cpu'), Emotion_dict).flatten()
        
        pred_list = np.append(pred_list, pred_flat)
        labels_list = np.append(labels_list, labels_flat)

    result = classification_report(pred_list, labels_list, digits=4, output_dict=True)
    for key in result.keys():
        if key !='accuracy':
            test_logs.append([
                    result[key]['precision'], 
                    result[key]['recall'], 
                    result[key]['f1-score'], 
                    result[key]['support'] 
                ])
    return test_logs













