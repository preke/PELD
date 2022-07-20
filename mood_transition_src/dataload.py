import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import BertTokenizer, BertConfig
from sklearn.model_selection import train_test_split



from utils import Emotion_dict, get_vad, Mood_dict, get_vad_dict


def get_sent_vad(VAD_dict, input_ids, tokenizer):
    '''
    get the VAD score of one sentence through the word VAD vectors
    '''
    VAD_scores = []
    w_list = re.sub(r'[^\w\s\[\]]','',tokenizer.decode(input_ids)).split()
    v_score, a_score, d_score = 0, 0, 0
    for word in w_list:
        try:
            v_score += VAD_dict[word][0]
            a_score += VAD_dict[word][1]
            d_score += VAD_dict[word][2]
        except:
            v_score += 0
            a_score += 0
            d_score += 0

    v_score/=float(len(w_list))
    a_score/=float(len(w_list))
    d_score/=float(len(w_list))
    VAD_scores = [v_score, a_score, d_score]
    return VAD_scores


def get_sent_vad_attention(VAD_dict, input_id_2, tokenizer, user_emo):

    VAD_scores = []
    w_list = re.sub(r'[^\w\s\[\]]','',tokenizer.decode(input_id_2)).split()
    for word in w_list:
        try:
            VAD_scores.append([VAD_dict[word][0], VAD_dict[word][1], VAD_dict[word][2]])
        except:
            VAD_scores.append([0,0,0])

    VAD_scores = torch.Tensor(VAD_scores)
    user_emo = torch.Tensor(user_emo)
    # print(VAD_scores.shape) # sentence_length * 3
    # print(user_emo.shape) # 1 * 3
    # inner
    VAD_scores_weights = torch.inner(VAD_scores, user_emo) # sentence_length * 1
    # print(VAD_scores_weights.shape)
    VAD_scores_weights = F.softmax(VAD_scores_weights) # sentence_length * 1

    vad_attn = [0,0,0]
    for i in range(len(VAD_scores)):
        vad_attn[0] += VAD_scores[i][0] * VAD_scores_weights[i]
        vad_attn[1] += VAD_scores[i][1] * VAD_scores_weights[i]
        vad_attn[2] += VAD_scores[i][2] * VAD_scores_weights[i]
    
    return vad_attn


def personality_to_vad(personality):
    '''
    Convert 5-d personality to 3-d persoanlity parameters
    '''
    O, C, E, A, N = personality[:, 0], personality[:, 1], personality[:, 2], personality[:, 3], personality[:, 4]
    valence = 0.21 * 10 * E + 0.59 * 10 * A + 0.19 * 10 * N
    arousal = 0.15 * O + 0.30 * 10 * A - 0.57 * 10 * N
    dominance = 0.25 * O + 0.17 * C + 0.60 * 10 * E - 0.32 * 10 * A
    return torch.cat((valence.unsqueeze(-1), arousal.unsqueeze(-1), dominance.unsqueeze(-1)), 1)


def load_data(args, DATA_PATH):
    '''
    Load data...
    '''
    df = pd.read_csv(DATA_PATH, sep='\t').fillna('Nan')
    
    Utterance_1   = df['Utterance_1'].values 
    Utterance_2   = df['Utterance_2'].values
    if args.base == 'RoBERTa':
        uttr_input    = Utterance_1 + ' </s> ' + Utterance_2
    elif args.base == 'BERT':
        uttr_input    = Utterance_1 + ' [SEP] ' + Utterance_2
    
    Utterance_3   = df['Utterance_3'].values
    
    ## directly get persoanlity parameters
    personalities = [eval(i) for i in df['Personality']]
    personalities = torch.tensor(personalities)
    personalities = personality_to_vad(personalities) 
    

    ## Emotion in VAD vectors
    init_emo     = df['Emotion_1']
    user_emo     = df['Emotion_2']
    response_emo = df['Emotion_3']

    init_emo     = [Emotion_dict[i] for i in init_emo]
    user_emo     = [Emotion_dict[i] for i in user_emo]    
    response_emo = [Emotion_dict[i] for i in response_emo]
    
    
    init_mood     = df['Mood_1']
    response_mood = df['Mood_3']
    
    # discrete label
    
    from sklearn.preprocessing import LabelEncoder
    moodencoder  = LabelEncoder()
    response_mood_label = moodencoder.fit_transform(response_mood)
    
    
    init_mood     = [Mood_dict[i] for i in init_mood]
    response_mood_vad = [Mood_dict[i] for i in response_mood]

    from sklearn.preprocessing import LabelEncoder
    labelencoder  = LabelEncoder()
    label         = df['Emotion_3']
    label_enc     = labelencoder.fit_transform(label)
    labels        = label_enc
    
    if args.base == 'RoBERTa':
        tokenizer  = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    elif args.base == 'BERT':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        
    input_ids   = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN,pad_to_max_length=True) for sent in uttr_input]
    input_ids_2 = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN,pad_to_max_length=True) for sent in Utterance_2]
    input_ids_3 = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN,pad_to_max_length=True) for sent in Utterance_3]
    
    attention_masks = [[float(i>0) for i in seq] for seq in input_ids]
    attention_masks_2 = [[float(i>0) for i in seq] for seq in input_ids_2]
    attention_masks_3 = [[float(i>0) for i in seq] for seq in input_ids_3]
    
    
    ## utterance VAD values
    VAD_dict = get_vad_dict()
    
    
    # uttr_vad_1 = [get_sent_vad(VAD_dict, i, tokenizer) for i in input_ids]
    # uttr_vad = [get_sent_vad(VAD_dict, i, tokenizer) for i in input_ids_2]
    uttr_vad = [get_sent_vad_attention(VAD_dict, input_ids_2[i], tokenizer, user_emo[i]) for i in range(len(input_ids_2))]

    # i = 0

    # uttr_vad = [[uttr_vad_1[i][0]*0.5 + uttr_vad_2[i][0]*0.5, 
    #              uttr_vad_1[i][1]*0.5 + uttr_vad_2[i][1]*0.5,
    #              uttr_vad_1[i][2]*0.5 + uttr_vad_2[i][2]*0.5] for i in range(len(uttr_vad_1))]
    # uttr_vad = [[(user_emo[i][0] + init_emo[i][0])/2.0, (user_emo[i][1] + init_emo[i][1])/2.0, (user_emo[i][2] + init_emo[i][2])/2.0]for i in range(len(user_emo))]
    
    u3_vad = [get_sent_vad(VAD_dict, i, tokenizer) for i in input_ids_3]
    
    ## Train Test Split
    train_inputs,test_inputs,train_labels,test_labels = \
        train_test_split(input_ids, labels, random_state=args.SEED, test_size=0.1, stratify=labels)
    
    train_inputs_2,test_inputs_2,train_labels,test_labels = \
        train_test_split(input_ids_2, labels, random_state=args.SEED, test_size=0.1, stratify=labels)
    
    train_inputs_3,test_inputs_3,train_labels,test_labels = \
        train_test_split(input_ids_3, labels, random_state=args.SEED, test_size=0.1, stratify=labels) 
    
    train_attn_masks_2,test_attn_masks_2,train_labels,test_labels = \
        train_test_split(attention_masks_2, labels, random_state=args.SEED, test_size=0.1, stratify=labels)
    
    train_attn_masks,test_attn_masks,train_labels,test_labels = \
        train_test_split(attention_masks, labels, random_state=args.SEED, test_size=0.1, stratify=labels)
    
    train_personalities,test_personalities,_,_ = \
        train_test_split(personalities, labels, random_state=args.SEED, test_size=0.1, stratify=labels)
    
    train_init_emo,test_init_emo,_,_ = \
        train_test_split(init_emo, labels, random_state=args.SEED, test_size=0.1, stratify=labels)
    
    train_user_emo,test_user_emo,_,_ = \
        train_test_split(user_emo, labels, random_state=args.SEED, test_size=0.1, stratify=labels)
    
    train_response_emo,test_response_emo,_,_ = \
        train_test_split(response_emo, labels, random_state=args.SEED, test_size=0.1, stratify=labels)
    
    train_init_mood, test_init_mood,_,_ = \
        train_test_split(init_mood, labels, random_state=args.SEED, test_size=0.1, stratify=labels)
    
    train_response_mood_vad, test_response_mood_vad,_,_ = \
        train_test_split(response_mood_vad, labels, random_state=args.SEED, test_size=0.1, stratify=labels)

    train_response_mood_label, test_response_mood_label,_,_ = \
        train_test_split(response_mood_label, labels, random_state=args.SEED, test_size=0.1, stratify=labels)
    
    train_uttr_vad, test_uttr_vad,_,_ = \
        train_test_split(uttr_vad, labels, random_state=args.SEED, test_size=0.1, stratify=labels)
    
    train_u3_vad, test_u3_vad,_,_ = \
        train_test_split(u3_vad, labels, random_state=args.SEED, test_size=0.1, stratify=labels)
    
    train_set_labels = train_labels
    
    train_inputs,valid_inputs,train_labels,valid_labels = \
        train_test_split(train_inputs, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)
    
    train_inputs_2,valid_inputs_2,train_labels,valid_labels = \
        train_test_split(train_inputs_2, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)
    
    train_inputs_3,valid_inputs_3,train_labels,valid_labels = \
        train_test_split(train_inputs_3, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels) 
    
    train_attn_masks_2,valid_attn_masks_2,train_labels,valid_labels = \
        train_test_split(train_attn_masks_2, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)
    
    train_attn_masks,valid_attn_masks,train_labels,valid_labels = \
        train_test_split(train_attn_masks, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)
    
    train_personalities,valid_personalities,_,_ = \
        train_test_split(train_personalities, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)

    train_init_emo,valid_init_emo,_,_ = \
        train_test_split(train_init_emo, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)
    train_user_emo,valid_user_emo,_,_ = \
        train_test_split(train_user_emo, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)
    train_response_emo,valid_response_emo,_,_ = \
        train_test_split(train_response_emo, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)
    
    train_init_mood,valid_init_mood,_,_ = \
        train_test_split(train_init_mood, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)
    train_response_mood_vad,valid_response_mood_vad,_,_ = \
        train_test_split(train_response_mood_vad, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)

    train_response_mood_label,valid_response_mood_label,_,_ = \
        train_test_split(train_response_mood_label, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)
    
    train_uttr_vad, valid_uttr_vad,_,_ = \
        train_test_split(train_uttr_vad, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)
    
    train_u3_vad, valid_u3_vad,_,_ = \
        train_test_split(train_u3_vad, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)

    ## Tensor Wrapper
    train_inputs      = torch.tensor(train_inputs)
    valid_inputs      = torch.tensor(valid_inputs)
    test_inputs       = torch.tensor(test_inputs)
    
    train_inputs_2      = torch.tensor(train_inputs_2)
    valid_inputs_2      = torch.tensor(valid_inputs_2)
    test_inputs_2       = torch.tensor(test_inputs_2)
    
    train_inputs_3      = torch.tensor(train_inputs_3)
    valid_inputs_3      = torch.tensor(valid_inputs_3)
    test_inputs_3       = torch.tensor(test_inputs_3)
    
    train_attn_masks_2 = torch.tensor(train_attn_masks_2)
    valid_attn_masks_2 = torch.tensor(valid_attn_masks_2)
    test_attn_masks_2  = torch.tensor(test_attn_masks_2)
    
    train_attn_masks = torch.tensor(train_attn_masks)
    valid_attn_masks = torch.tensor(valid_attn_masks)
    test_attn_masks  = torch.tensor(test_attn_masks)

    train_uttr_vad        = torch.tensor(train_uttr_vad).squeeze(1)
    valid_uttr_vad        = torch.tensor(valid_uttr_vad).squeeze(1)
    test_uttr_vad         = torch.tensor(test_uttr_vad).squeeze(1)

    train_u3_vad        = torch.tensor(train_u3_vad).squeeze(1)
    valid_u3_vad        = torch.tensor(valid_u3_vad).squeeze(1)
    test_u3_vad         = torch.tensor(test_u3_vad).squeeze(1)
    
    train_labels        = torch.tensor(train_labels)
    valid_labels        = torch.tensor(valid_labels)
    test_labels         = torch.tensor(test_labels)
    
    train_init_emo      = torch.tensor(train_init_emo)
    valid_init_emo      = torch.tensor(valid_init_emo)
    test_init_emo       = torch.tensor(test_init_emo)
    
    train_user_emo      = torch.tensor(train_user_emo)
    valid_user_emo      = torch.tensor(valid_user_emo)
    test_user_emo       = torch.tensor(test_user_emo)
    
    train_response_emo  = torch.tensor(train_response_emo)
    valid_response_emo  = torch.tensor(valid_response_emo)
    test_response_emo   = torch.tensor(test_response_emo)
    
    train_init_mood      = torch.tensor(train_init_mood)
    valid_init_mood      = torch.tensor(valid_init_mood)
    test_init_mood       = torch.tensor(test_init_mood)
    
    train_response_mood_vad  = torch.tensor(train_response_mood_vad)
    valid_response_mood_vad  = torch.tensor(valid_response_mood_vad)
    test_response_mood_vad   = torch.tensor(test_response_mood_vad)

    train_response_mood_label  = torch.tensor(train_response_mood_label)
    valid_response_mood_label  = torch.tensor(valid_response_mood_label)
    test_response_mood_label   = torch.tensor(test_response_mood_label)
    

    train_data = TensorDataset(train_inputs, train_inputs_2, train_inputs_3, train_attn_masks, train_attn_masks_2, train_uttr_vad, train_personalities, train_init_emo, train_user_emo, train_response_emo, train_init_mood, train_response_mood_vad, train_response_mood_label, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    valid_data = TensorDataset(valid_inputs, valid_inputs_2, valid_inputs_3, valid_attn_masks, valid_attn_masks_2, valid_uttr_vad, valid_personalities, valid_init_emo, valid_user_emo, valid_response_emo, valid_init_mood, valid_response_mood_vad, valid_response_mood_label, valid_labels)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)

    test_data = TensorDataset(test_inputs, test_inputs_2, test_inputs_3, test_attn_masks, test_attn_masks_2, test_uttr_vad, test_personalities, test_init_emo, test_user_emo, test_response_emo, test_init_mood, test_response_mood_vad, test_response_mood_label, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

    return len(train_data), train_dataloader, valid_dataloader, test_dataloader









