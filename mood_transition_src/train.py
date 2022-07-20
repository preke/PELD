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
from transformers import BertTokenizer, BertConfig
import random
import shutil
from utils import Emotion_dict




class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss


def get_losses_weights(losses):
    weights = torch.div(losses, torch.sum(losses)) * losses.shape[0]
    return weights


def train_model(model, args, train_dataloader, valid_dataloader, test_dataloader):
    
    num_warmup_steps = 0
    num_training_steps = len(train_dataloader)*args.epochs
    
            
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=args.adam_epsilon, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
    
    train_logs = []
    valid_logs = []
    test_logs  = []

    mood_cls_loss_list = []
    mood_mse_loss_list = []
    emo_cls_loss_list = []

    loss_list = []
    
    best_macro = 0.0
    best_epoch = 0
    model.zero_grad()

    for _ in tnrange(1, args.epochs+1, desc='Epoch'):
        print("<" + "="*22 + F" Epoch {_} "+ "="*22 + ">")
        # Calculate total loss for this epoch
        batch_loss = 0
        mood_batch_mse_loss = 0
        mood_batch_cls_loss = 0
        emo_batch_cls_loss = 0
        
        train_accuracy, nb_train_steps = 0, 0
        
        pred_list = np.array([])
        labels_list = np.array([])

        pred_mood_list   = np.array([])
        mood_labels_list = np.array([])
        
        for step, batch in enumerate(train_dataloader):

            model.train()
            batch = tuple(t.cuda(args.device) for t in batch)
            b_input_ids, b_input_ids_2, b_input_ids_3, b_attn_masks, b_attn_masks_2,\
            b_uttr_vad, b_personality, b_init_emo, b_user_emo, b_response_emo, \
            b_init_mood, b_response_mood_vad, b_response_mood_label, b_labels = batch
            if args.mode == '7': # given mood
                b_init_mood = b_response_mood_vad
            
            response_mood_vad, response_mood_logits, response_emo = model(b_input_ids, b_attn_masks, b_uttr_vad, b_user_emo, b_personality, b_init_mood)
            
            mood_mse_lf  = nn.MSELoss()
            mood_cls_lf  = MultiFocalLoss(num_class=4, gamma=2.0, reduction='mean') # nn.CrossEntropyLoss()
            emo_loss_fct = MultiFocalLoss(num_class=7, gamma=2.0, reduction='mean') # nn.CrossEntropyLoss()
            


            emo_loss      = emo_loss_fct(response_emo, b_labels)
            mood_mse_loss = mood_mse_lf(response_mood_vad, b_response_mood_vad)
            mood_cls_loss = mood_cls_lf(response_mood_logits, b_response_mood_label)
            
            
           
            
            # losses = torch.tensor([mood_mse_loss, emo_loss, mood_cls_loss])
            # loss_w = get_losses_weights(losses) 
            # new_losses = losses * loss_w
            # loss = mood_mse_loss*loss_w[0] + emo_loss*loss_w[1] + mood_cls_loss*loss_w[2]
            
            if args.mode == '1':
                loss = emo_loss 
            elif args.mode == '2':
                loss = mood_cls_loss + emo_loss
            elif args.mode == '5':
                loss = mood_cls_loss + emo_loss
            else: 
                loss = mood_mse_loss + emo_loss
            
            response_emo         = response_emo.detach().to('cpu').numpy()
            label_ids            = b_labels.to('cpu').numpy()                
            pred_flat            = np.argmax(response_emo, axis=1).flatten()
            labels_flat          = label_ids.flatten()

            response_mood_logits = response_mood_logits.detach().to('cpu').numpy()
            mood_labels          = b_response_mood_label.to('cpu').numpy()
            mood_pred            = np.argmax(response_mood_logits, axis=1).flatten()
            mood_labels          = mood_labels.flatten()


            pred_list        = np.append(pred_list, pred_flat)
            labels_list      = np.append(labels_list, labels_flat)

            pred_mood_list   = np.append(pred_mood_list, mood_pred)
            mood_labels_list = np.append(mood_labels_list, mood_labels)
            
            nb_train_steps += 1

            # Backward pass
            loss.backward()
            
            # Clip the norm of the gradients to 1.0
            # Gradient clipping is not in AdamW anymore
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            
            optimizer.step()
            scheduler.step()
            # Clear the previous accumulated gradients
            optimizer.zero_grad()
            # Update tracking variables
            batch_loss += loss.item()
            mood_batch_mse_loss += mood_mse_loss.item()
            mood_batch_cls_loss += mood_cls_loss.item()
            emo_batch_cls_loss  += emo_loss.item()
      
        #  Calculate the average loss over the training data.
        avg_train_loss = batch_loss / len(train_dataloader)
        
        avg_mood_mse_loss = mood_batch_mse_loss / len(train_dataloader)
        avg_mood_cls_loss = mood_batch_cls_loss / len(train_dataloader)
        avg_emo_cls_loss  = emo_batch_cls_loss / len(train_dataloader)

        #store the current learning rate
        for param_group in optimizer.param_groups:
            print("\n\tCurrent Learning rate: ",param_group['lr'])

        # print("\n\tCurrent overall loss: ", avg_train_loss)
        print("\n\tCurrent mood cls loss: ", avg_mood_cls_loss)
        print("\n\tCurrent mood mse loss: ", avg_mood_mse_loss)
        print("\n\tCurrent emo cls loss: ", avg_emo_cls_loss)

        mood_mse_loss_list.append(avg_mood_mse_loss)
        mood_cls_loss_list.append(avg_mood_cls_loss)
        emo_cls_loss_list.append(avg_emo_cls_loss)
        loss_list.append(avg_train_loss)
        
        
        
        print(classification_report(pred_list, labels_list, digits=4, output_dict=False))
        result = classification_report(pred_list, labels_list, digits=4, output_dict=True)

        print('Mood prediction\n')
        print(classification_report(pred_mood_list, mood_labels_list, digits=4, output_dict=False))
        mood_result = classification_report(pred_mood_list, mood_labels_list, digits=4, output_dict=True)
        for key in result.keys():
            if key !='accuracy':
                    train_logs.append([
                        key, 
                        result[key]['precision'], 
                        result[key]['recall'], 
                        result[key]['f1-score'], 
                        result[key]['support'] 
                    ])
            else:
                train_logs.append(['accuracy', 0,0,result[key],0]);

        for key in mood_result.keys():
            if key !='accuracy':
                    train_logs.append([
                        'mood_'+key,
                        mood_result[key]['precision'], 
                        mood_result[key]['recall'], 
                        mood_result[key]['f1-score'], 
                        mood_result[key]['support'] 
                    ])
            else:
                train_logs.append(['mood_'+'accuracy', 0,0,result[key],0]);

        
        valid_logs = eval_model(model, valid_dataloader, args, valid_logs)
        test_logs, pred_list, best_macro, best_epoch = test_model(model, test_dataloader, args, test_logs, best_macro, best_epoch, _)
        print('Current best macro is ', best_macro)
        print('Current best epoch is ', best_epoch)
        print('loss list', loss_list)
        print('mood mse loss list', mood_mse_loss_list)
        print('mood cls loss list', mood_cls_loss_list)
        print('emo cls loss list', emo_cls_loss_list)

    df_train_logs = pd.DataFrame(train_logs, columns=['label', 'precision', 'recall', 'f1-score', 'support']).add_prefix('train_')
    df_valid_logs = pd.DataFrame(valid_logs, columns=['label','precision', 'recall', 'f1-score', 'support']).add_prefix('valid_')
    df_test_logs  = pd.DataFrame(test_logs, columns=['label','precision', 'recall', 'f1-score', 'support']).add_prefix('test_')

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

def eval_model(model, valid_dataloader, args, valid_logs):
    # Validation
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()
    
    pred_list = np.array([])
    labels_list = np.array([])


    pred_mood_list   = np.array([])
    mood_labels_list = np.array([])
    
    for batch in valid_dataloader:
        batch = tuple(t.cuda(args.device) for t in batch)
        b_input_ids, b_input_ids_2, b_input_ids_3, b_attn_masks, b_attn_masks_2,\
        b_uttr_vad, b_personality, b_init_emo, b_user_emo, b_response_emo, \
        b_init_mood, b_response_mood_vad, b_response_mood_label, b_labels = batch
        if args.mode == '7': # given mood
                b_init_mood = b_response_mood_vad

        with torch.no_grad():
            response_mood_vad, response_mood_logits, response_emo = model(b_input_ids, b_attn_masks, b_uttr_vad, b_user_emo, b_personality, b_init_mood)
        
        mood_mse_lf  = nn.MSELoss()
        mood_cls_lf  = nn.CrossEntropyLoss()
        emo_loss_fct = nn.CrossEntropyLoss()

        emo_loss      = emo_loss_fct(response_emo, b_labels)
        mood_mse_loss = mood_mse_lf(torch.sign(response_mood_vad), b_response_mood_vad)
        mood_cls_loss = mood_cls_lf(response_mood_logits, b_response_mood_label)
        
        
        
        
        
        response_emo         = response_emo.detach().to('cpu').numpy()
        label_ids            = b_labels.to('cpu').numpy()                
        pred_flat            = np.argmax(response_emo, axis=1).flatten()
        labels_flat          = label_ids.flatten()

        response_mood_logits = response_mood_logits.detach().to('cpu').numpy()
        mood_labels          = b_response_mood_label.to('cpu').numpy()
        mood_pred            = np.argmax(response_mood_logits, axis=1).flatten()
        mood_labels          = mood_labels.flatten()


        pred_list        = np.append(pred_list, pred_flat)
        labels_list      = np.append(labels_list, labels_flat)

        pred_mood_list   = np.append(pred_mood_list, mood_pred)
        mood_labels_list = np.append(mood_labels_list, mood_labels)


    
    # print(classification_report(pred_list, labels_list, digits=4, output_dict=False))
    result = classification_report(pred_list, labels_list, digits=4, output_dict=True)
    # print('Mood prediction\n')
    # print(classification_report(pred_mood_list, mood_labels_list, digits=4, output_dict=False))
    mood_result = classification_report(pred_mood_list, mood_labels_list, digits=4, output_dict=True)

    for key in result.keys():
            if key !='accuracy':
                    valid_logs.append([
                        key, 
                        result[key]['precision'], 
                        result[key]['recall'], 
                        result[key]['f1-score'], 
                        result[key]['support'] 
                    ])
            else:
                valid_logs.append(['accuracy', 0,0,result[key],0]);

    for key in mood_result.keys():
            if key !='accuracy':
                    valid_logs.append([
                        'mood_'+key,
                        mood_result[key]['precision'], 
                        mood_result[key]['recall'], 
                        mood_result[key]['f1-score'], 
                        mood_result[key]['support'] 
                    ])
            else:
                valid_logs.append(['mood_'+'accuracy', 0,0,result[key],0]);
    return valid_logs


def test_model(model, test_dataloader, args, test_logs, best_macro=0.0, best_epoch=0, epoch=0):
    # Test

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()
    
    pred_list   = np.array([])
    labels_list = np.array([])

    pred_mood_list   = np.array([])
    mood_labels_list = np.array([])

    test_batch  = []
    tokenizer   =  BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    with open('result_file.txt', 'w') as f:
        for batch in test_dataloader:
            batch = tuple(t.cuda(args.device) for t in batch)
            b_input_ids, b_input_ids_2, b_input_ids_3, b_attn_masks, b_attn_masks_2,\
            b_uttr_vad, b_personality, b_init_emo, b_user_emo, b_response_emo, \
            b_init_mood, b_response_mood_vad, b_response_mood_label, b_labels = batch

            if args.mode == '7': # given mood
                b_init_mood = b_response_mood_vad
            with torch.no_grad():
                response_mood_vad, response_mood_logits, response_emo = model(b_input_ids, b_attn_masks, b_uttr_vad, b_user_emo, b_personality, b_init_mood)
            
            mood_mse_lf  = nn.MSELoss()
            mood_cls_lf  = nn.CrossEntropyLoss()
            emo_loss_fct = nn.CrossEntropyLoss()

            emo_loss      = emo_loss_fct(response_emo, b_labels)
            mood_mse_loss = mood_mse_lf(torch.sign(response_mood_vad), b_response_mood_vad)
            mood_cls_loss = mood_cls_lf(response_mood_logits, b_response_mood_label)
            
            response_emo         = response_emo.detach().to('cpu').numpy()
            label_ids            = b_labels.to('cpu').numpy()                
            pred_flat            = np.argmax(response_emo, axis=1).flatten()
            labels_flat          = label_ids.flatten()

            response_mood_logits = response_mood_logits.detach().to('cpu').numpy()
            mood_labels          = b_response_mood_label.to('cpu').numpy()
            mood_pred            = np.argmax(response_mood_logits, axis=1).flatten()
            mood_labels          = mood_labels.flatten()


            pred_list        = np.append(pred_list, pred_flat)
            labels_list      = np.append(labels_list, labels_flat)

            pred_mood_list   = np.append(pred_mood_list, mood_pred)
            mood_labels_list = np.append(mood_labels_list, mood_labels)

            for i in range(len(b_input_ids)):
                f.write(tokenizer.decode(b_input_ids[i]) + '\t')
                f.write(tokenizer.decode(b_input_ids_3[i])+ '\t')
                f.write(str(mood_pred[i]) + "\t" + str(mood_labels[i]) + '\t')
                f.write(str(pred_flat[i]) + "\t" + str(labels_flat[i]) + '\n')


    
    print(classification_report(pred_list, labels_list, digits=4, output_dict=False))
    result = classification_report(pred_list, labels_list, digits=4, output_dict=True)

    print('Mood prediction\n')
    print(classification_report(pred_mood_list, mood_labels_list, digits=4, output_dict=False))
    mood_result = classification_report(pred_mood_list, mood_labels_list, digits=4, output_dict=True)
    for key in result.keys():
            if key !='accuracy':
                    test_logs.append([
                        key, 
                        result[key]['precision'], 
                        result[key]['recall'], 
                        result[key]['f1-score'], 
                        result[key]['support'] 
                    ])
            else:
                test_logs.append(['accuracy', 0,0,result[key],0]);

    for key in mood_result.keys():
            if key !='accuracy':
                    test_logs.append([
                        'mood_'+key,
                        mood_result[key]['precision'], 
                        mood_result[key]['recall'], 
                        mood_result[key]['f1-score'], 
                        mood_result[key]['support'] 
                    ])
            else:
                test_logs.append(['mood_'+'accuracy', 0,0,result[key],0]);

    if result['macro avg']['f1-score'] > best_macro:
        best_macro = result['macro avg']['f1-score']
        best_epoch = epoch
        shutil.copyfile('result_file.txt', 'Mood_'+args.mode+'_best_result.txt')
    
    return test_logs, pred_list, best_macro, best_epoch
















