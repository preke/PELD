import pandas as pd
import numpy as np
import re
import time



meld_data_path  = 'source_data/MELD/'
emory_data_path  = 'source_data/emorynlp/'

Personality_dict = {
    'Chandler' : '[0.648, 0.375, 0.386, 0.58, 0.477]',
    'Joey' : '[0.574, 0.614, 0.297, 0.545, 0.455]',
    'Ross' : '[0.722, 0.489, 0.6, 0.533, 0.356]',    
    'Monica' : '[0.713, 0.457, 0.457, 0.66, 0.511]',
    'Phoebe' : '[0.6, 0.48, 0.31, 0.46, 0.56]',
    'Rachel' : '[0.635, 0.354, 0.521, 0.552, 0.469]'

}


Emotion_Senti = {
    'anger' : 'negative',
    'sadness' : 'negative',
    'neutral' :    'neutral',
    'joy' : 'positive',
    'surprise' : 'positive',
    'fear' : 'negative',
    'disgust': 'negative'
}



#  ********************************** MELD **********************************


def extract_meld_row(df, keep_same_speaker=True):
    ans_list = []
    column_list = ['Speaker_1', 'Speaker_2', 'Personality',\
                   'Utterance_1', 'Utterance_2', 'Utterance_3',\
                   'Emotion_1', 'Emotion_2', 'Emotion_3',
                   'Sentiment_1', 'Sentiment_2', 'Sentiment_3',]
    
    for group in df.groupby('Dialogue_ID'):
        sub_df = group[1]
        for i in range(len(sub_df) - 2):
            if sub_df.iloc[i]['Speaker'] != sub_df.iloc[i+2]['Speaker']:
                continue
            else:
                tmp_list = []

                tmp_list.append(sub_df.iloc[i]['Speaker'])
                tmp_list.append(sub_df.iloc[i+1]['Speaker'])

                try:
                    tmp_list.append(Personality_dict[sub_df.iloc[i]['Speaker']])
                except:
                    continue

                tmp_list.append(re.sub(r'[^\x00-\x7f]',r' ', sub_df.iloc[i]['Utterance'].strip()))
                tmp_list.append(re.sub(r'[^\x00-\x7f]',r' ', sub_df.iloc[i+1]['Utterance'].strip()))
                tmp_list.append(re.sub(r'[^\x00-\x7f]',r' ', sub_df.iloc[i+2]['Utterance'].strip()))

                tmp_list.append(sub_df.iloc[i]['Emotion'])
                tmp_list.append(sub_df.iloc[i+1]['Emotion'])
                tmp_list.append(sub_df.iloc[i+2]['Emotion'])

                tmp_list.append(Emotion_Senti[sub_df.iloc[i]['Emotion']])
                tmp_list.append(Emotion_Senti[sub_df.iloc[i+1]['Emotion']])
                tmp_list.append(Emotion_Senti[sub_df.iloc[i+2]['Emotion']])
                
                if keep_same_speaker == False and sub_df.iloc[i]['Speaker'] == sub_df.iloc[i+1]['Speaker']:
                    pass
                else:
                    ans_list.append(tmp_list)
    ans_df = pd.DataFrame(ans_list, columns = column_list)
    return ans_df


meld_train_df = pd.read_csv(meld_data_path + 'train_sent_emo.csv')
meld_valid_df = pd.read_csv(meld_data_path + 'dev_sent_emo.csv')
meld_test_df = pd.read_csv(meld_data_path + 'test_sent_emo.csv')

meld_train = extract_meld_row(meld_train_df, keep_same_speaker = False)
meld_valid = extract_meld_row(meld_valid_df, keep_same_speaker = False)
meld_test  = extract_meld_row(meld_test_df, keep_same_speaker = False)

print('MELD: ')
print(meld_train.shape)
print(meld_valid.shape)
print(meld_test.shape)


#  ********************************** EmoryNLP **********************************

Emo_rename_dict = {
    'Joyful' : 'joy',
    'Mad'    : 'anger',
    'Neutral': 'neutral',
    'Sad'    : 'sadness',
    'Scared' : 'fear'
}

Senti_dict = {
    'Joyful' : 'positive',
    'Mad'    : 'negative',
    'Neutral': 'neutral',
    'Sad'    : 'negative',
    'Scared' : 'negative'
}

def extract_emory_row(df, keep_same_speaker=True):
    
    df['Speaker'] = df['Speaker'].apply(lambda x: eval(x)[0].split()[0])
    ans_list = []
    column_list = ['Speaker_1', 'Speaker_2', 'Personality',\
                   'Utterance_1', 'Utterance_2', 'Utterance_3',\
                   'Emotion_1', 'Emotion_2', 'Emotion_3',
                   'Sentiment_1', 'Sentiment_2', 'Sentiment_3',]
    
    for group in df.groupby(['Season', 'Episode', 'Scene_ID']):
        sub_df = group[1]
        for i in range(len(sub_df) - 2):
            if sub_df.iloc[i]['Speaker'] != sub_df.iloc[i+2]['Speaker']:
                continue
            else:
                tmp_list = []

                tmp_list.append(sub_df.iloc[i]['Speaker'])
                tmp_list.append(sub_df.iloc[i+1]['Speaker'])

                try:
                    tmp_list.append(Personality_dict[sub_df.iloc[i]['Speaker']])
                except:
                    continue

                tmp_list.append(re.sub(r'[^\x00-\x7f]',r' ', sub_df.iloc[i]['Utterance'].strip()))
                tmp_list.append(re.sub(r'[^\x00-\x7f]',r' ', sub_df.iloc[i+1]['Utterance'].strip()))
                tmp_list.append(re.sub(r'[^\x00-\x7f]',r' ', sub_df.iloc[i+2]['Utterance'].strip()))
                try:
                    tmp_list.append(Emo_rename_dict[sub_df.iloc[i]['Emotion']])
                    
                    tmp_list.append(Emo_rename_dict[sub_df.iloc[i+1]['Emotion']])
                    
                    tmp_list.append(Emo_rename_dict[sub_df.iloc[i+2]['Emotion']])
                    
                    # ----
                    
                    tmp_list.append(Senti_dict[sub_df.iloc[i]['Emotion']])
                    
                    tmp_list.append(Senti_dict[sub_df.iloc[i+1]['Emotion']])
                    
                    tmp_list.append(Senti_dict[sub_df.iloc[i+2]['Emotion']])
                except:
                    continue
                
                if keep_same_speaker == False and sub_df.iloc[i]['Speaker'] == sub_df.iloc[i+1]['Speaker']:
                    pass
                else:
                    ans_list.append(tmp_list)
    ans_df = pd.DataFrame(ans_list, columns = column_list)
    return ans_df



emory_train_df = pd.read_csv(emory_data_path + 'emorynlp_train_final.csv')
emory_valid_df = pd.read_csv(emory_data_path + 'emorynlp_dev_final.csv')
emory_test_df = pd.read_csv(emory_data_path + 'emorynlp_test_final.csv')


emory_train = extract_emory_row(emory_train_df, keep_same_speaker = False)
emory_valid = extract_emory_row(emory_valid_df, keep_same_speaker = False)
emory_test  = extract_emory_row(emory_test_df, keep_same_speaker = False)

print('EmoryNLP')
print(emory_train.shape)
print(emory_valid.shape)
print(emory_test.shape)


df = pd.concat([meld_train,meld_valid,meld_test,emory_train,emory_valid,emory_test]).drop_duplicates()
df['Personality'] = df['Personality'].apply(eval) 

labels = df['Emotion_3']
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
label_enc = labelencoder.fit_transform(labels)
labels = label_enc


SEED = 42
from sklearn.model_selection import train_test_split
train, test, train_labels, test_labels = \
    train_test_split(df, labels, random_state=SEED, test_size=0.1, stratify=labels)

train, valid, train_labels, valid_labels = \
    train_test_split(train, train_labels, random_state=SEED, test_size=0.1, stratify=train_labels)



df.to_csv('Dyadic_PELD.tsv', sep='\t', index=False)