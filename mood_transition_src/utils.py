import re
import pandas as pd

Emotion_dict = {
    'anger': [-0.51, 0.59, 0.25],
    'disgust': [-0.60, 0.35, 0.11],
    'fear': [-0.62, 0.82, -0.43],
    'joy': [0.81, 0.51, 0.46],
    'neutral': [0.0, 0.0, 0.0],
    'sadness': [-0.63, -0.27, -0.33],
    'surprise': [0.40, 0.67, -0.13]
}



# Mood_dict = {
#     'neutral': [0.0, 0.0, 0.0],
#     'M1' : [0.605, 0.59 , 0.165],
#     'M2': [-0.57666667,  0.58666667, -0.02333333],
#     'M3': [-0.63, -0.27, -0.33]
# }

Mood_dict = {
    'neutral': [0.0, 0.0, 0.0],
    'M1'     : [1.0, 1.0 , 0.0],
    'M2'     : [-1.0,  1.0, 0.0],
    'M3'     : [-1.0, -1.0, 0.0]
}

# Mood_dict = {
#     'neutral': [1.0, 0.0, 0.0, 0.0],
#     'M1' : [0.0, 1.0, 0.0, 0.0],
#     'M2': [0.0, 0.0, 1.0, 0.0],
#     'M3': [0.0, 0.0, 0.0, 1.0]
# }


Personality_dict = {
    'Chandler' : '[0.648, 0.375, 0.386, 0.58, 0.477]',
    'Joey' : '[0.574, 0.614, 0.297, 0.545, 0.455]',
    'Ross' : '[0.722, 0.489, 0.6, 0.533, 0.356]',    
    'Monica' : '[0.713, 0.457, 0.457, 0.66, 0.511]',
    'Phoebe' : '[0.6, 0.48, 0.31, 0.46, 0.56]',
    'Rachel' : '[0.635, 0.354, 0.521, 0.552, 0.469]'
}

## -0.5
# Personality_dict = {
#     'Chandler' : '[0.148, -0.125, -0.114, 0.08, -0.023]',
#     'Joey' : '[0.074, 0.114, -0.203, 0.045, -0.045]',
#     'Ross' : '[0.222, -0.011, 0.1, 0.033, -0.144]',    
#     'Monica' : '[0.213, -0.043, -0.043, 0.16, 0.011]',
#     'Phoebe' : '[0.1, -0.02, -0.19, -0.04, 0.06]',
#     'Rachel' : '[0.135, -0.146, 0.021, 0.052, -0.031]'

# }


Emotion_Senti = {
    'anger' : 'negative',
    'sadness' : 'negative',
    'neutral' :    'neutral',
    'joy' : 'positive',
    'surprise' : 'positive',
    'fear' : 'negative',
    'disgust': 'negative'
}


## get vad dict
def get_vad_dict():
    VAD_Lexicons = pd.read_csv('../source_data/NRC-VAD-Lexicon.txt', sep='\t')
    VAD_dict = {}
    for r in VAD_Lexicons.iterrows():
        VAD_dict[r[1]['Word']] = [r[1]['Valence'], r[1]['Arousal'], r[1]['Dominance']]
    return VAD_dict



def get_vad(VAD_dict, sents):
    VAD_scores = []
    for sent in sents:
        w_list = re.sub(r'[^\w\s\[\]]','',sent).split()
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
        VAD_scores.append([v_score, a_score, d_score])
    return VAD_scores

