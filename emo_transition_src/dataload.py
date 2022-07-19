import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import BertTokenizer, BertConfig
from sklearn.model_selection import train_test_split


from utils import Emotion_dict

def load_data(args, DATA_PATH):
    '''
    Load data...
    '''
    df = pd.read_csv(DATA_PATH, sep='\t').fillna('Nan')
    
    Utterance_1   = df['Utterance_1'].values 
    Utterance_2   = df['Utterance_2'].values
    Utterance_3   = df['Utterance_3'].values
    personalities = [eval(i) for i in df['Personality']]

    if args.Senti_or_Emo == 'Emotion': # Emotion
        init_emo     = df['Emotion_1']
        response_emo = df['Emotion_3']
        init_emo     = [Emotion_dict[i] for i in init_emo]
        response_emo = [Emotion_dict[i] for i in response_emo]
        labels       = df['Emotion_3']
    else: # Sentiment (have not modified)
        init_emo     = df['Emotion_1']
        response_emo = df['Emotion_3']
        init_emo     = [Emotion_dict[i] for i in init_emo]
        response_emo = [Emotion_dict[i] for i in response_emo]
        labels       = df['Emotion_3']
    
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    label_enc    = labelencoder.fit_transform(labels)
    labels       = label_enc

    if args.base == 'RoBERTa':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    elif args.base == 'BERT':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    input_ids_1 = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN,pad_to_max_length=True) for sent in Utterance_1]
    input_ids_2 = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN,pad_to_max_length=True) for sent in Utterance_2]
    input_ids_3 = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN,pad_to_max_length=True) for sent in Utterance_3]

    attention_masks_1 = [[float(i > 0) for i in seq] for seq in input_ids_1]
    attention_masks_2 = [[float(i > 0) for i in seq] for seq in input_ids_2]
    attention_masks_3 = [[float(i > 0) for i in seq] for seq in input_ids_3]

    ## Train Test Split

    train_inputs_1,test_inputs_1,train_labels,test_labels = \
    train_test_split(input_ids_1, labels, random_state=args.SEED, test_size=0.1, stratify=labels)

    train_inputs_2,test_inputs_2,train_labels,test_labels = \
        train_test_split(input_ids_2, labels, random_state=args.SEED, test_size=0.1, stratify=labels)

    train_inputs_3,test_inputs_3,train_labels,test_labels = \
        train_test_split(input_ids_3, labels, random_state=args.SEED, test_size=0.1, stratify=labels)

    train_attn_masks_1, test_attn_masks_1, train_labels, test_labels = \
        train_test_split(attention_masks_1, labels, random_state=args.SEED, test_size=0.1, stratify=labels)

    train_attn_masks_2, test_attn_masks_2, train_labels, test_labels = \
        train_test_split(attention_masks_2, labels, random_state=args.SEED, test_size=0.1, stratify=labels)

    train_attn_masks_3, test_attn_masks_3, train_labels, test_labels = \
        train_test_split(attention_masks_3, labels, random_state=args.SEED, test_size=0.1, stratify=labels)


    train_personalities,test_personalities,_,_ = \
        train_test_split(personalities, labels, random_state=args.SEED, test_size=0.1, stratify=labels)

    train_init_emo,test_init_emo,_,_ = \
        train_test_split(init_emo, labels, random_state=args.SEED, test_size=0.1, stratify=labels)

    train_response_emo,test_response_emo,_,_ = \
        train_test_split(response_emo, labels, random_state=args.SEED, test_size=0.1, stratify=labels)


    train_set_labels = train_labels

    train_inputs_1,valid_inputs_1,train_labels,valid_labels = \
        train_test_split(train_inputs_1, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)

    train_inputs_2,valid_inputs_2,train_labels,valid_labels = \
        train_test_split(train_inputs_2, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)

    train_inputs_3,valid_inputs_3,train_labels,valid_labels = \
        train_test_split(train_inputs_3, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)

    train_attn_masks_1, valid_attn_masks_1, train_labels, valid_labels = \
        train_test_split(train_attn_masks_1, train_set_labels, random_state=args.SEED, test_size=0.1,
                         stratify=train_set_labels)

    train_attn_masks_2, valid_attn_masks_2, train_labels, valid_labels = \
        train_test_split(train_attn_masks_2, train_set_labels, random_state=args.SEED, test_size=0.1,
                         stratify=train_set_labels)

    train_attn_masks_3, valid_attn_masks_3, train_labels, valid_labels = \
        train_test_split(train_attn_masks_3, train_set_labels, random_state=args.SEED, test_size=0.1,
                         stratify=train_set_labels)


    train_personalities,valid_personalities,_,_ = \
        train_test_split(train_personalities, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)

    train_init_emo,valid_init_emo,_,_ = \
        train_test_split(train_init_emo, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)

    train_response_emo,valid_response_emo,_,_ = \
        train_test_split(train_response_emo, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)


    ## Tensor Wrapper
    train_inputs_1      = torch.tensor(train_inputs_1)
    valid_inputs_1      = torch.tensor(valid_inputs_1)
    test_inputs_1       = torch.tensor(test_inputs_1)
    
    train_inputs_2      = torch.tensor(train_inputs_2)
    valid_inputs_2      = torch.tensor(valid_inputs_2)
    test_inputs_2       = torch.tensor(test_inputs_2)
    
    train_inputs_3      = torch.tensor(train_inputs_3)
    valid_inputs_3      = torch.tensor(valid_inputs_3)
    test_inputs_3       = torch.tensor(test_inputs_3)

    train_attn_masks_1 = torch.tensor(train_attn_masks_1)
    valid_attn_masks_1 = torch.tensor(valid_attn_masks_1)
    test_attn_masks_1 = torch.tensor(test_attn_masks_1)

    train_attn_masks_2 = torch.tensor(train_attn_masks_2)
    valid_attn_masks_2 = torch.tensor(valid_attn_masks_2)
    test_attn_masks_2 = torch.tensor(test_attn_masks_2)

    train_attn_masks_3 = torch.tensor(train_attn_masks_3)
    valid_attn_masks_3 = torch.tensor(valid_attn_masks_3)
    test_attn_masks_3 = torch.tensor(test_attn_masks_3)
    
    train_labels        = torch.tensor(train_labels)
    valid_labels        = torch.tensor(valid_labels)
    test_labels         = torch.tensor(test_labels)
    
    train_personalities = torch.tensor(train_personalities)
    valid_personalities = torch.tensor(valid_personalities)
    test_personalities  = torch.tensor(test_personalities)
    
    train_init_emo      = torch.tensor(train_init_emo)
    valid_init_emo      = torch.tensor(valid_init_emo)
    test_init_emo       = torch.tensor(test_init_emo)
    
    train_response_emo  = torch.tensor(train_response_emo)
    valid_response_emo  = torch.tensor(valid_response_emo)
    test_response_emo   = torch.tensor(test_response_emo)


    train_data = TensorDataset(train_inputs_1, train_inputs_2, train_inputs_3, train_attn_masks_1, train_attn_masks_2, train_attn_masks_3,
                               train_personalities, train_init_emo, train_response_emo, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    valid_data = TensorDataset(valid_inputs_1, valid_inputs_2, valid_inputs_3, valid_attn_masks_1, valid_attn_masks_2, valid_attn_masks_3,
                                   valid_personalities, valid_init_emo, valid_response_emo, valid_labels)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)

    test_data = TensorDataset(test_inputs_1, test_inputs_2, test_inputs_3, test_attn_masks_1, test_attn_masks_2, test_attn_masks_3,
                                  test_personalities, test_init_emo, test_response_emo, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

    return len(train_data), train_dataloader, valid_dataloader, test_dataloader









