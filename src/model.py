from transformers import RobertaConfig, RobertaModel, PreTrainedModel
import torch.nn as nn
import torch.nn.functional as F
import torch



class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RobertaConfig
    base_model_prefix = "roberta-base"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class RobertaClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks.
    """

    def __init__(self, config, num_labels):
        super().__init__()
        self.reduce = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels
        self.out_proj = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, features):
        features = self.reduce(features)
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
    
class RobertaClassificationHead_v3(nn.Module):
    """
    This is specially for mode 3
    """

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features #[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Emo_Generation(RobertaPreTrainedModel):
    def __init__(self, config, mode):
        super().__init__(config)
        self.num_labels = 7
        self.mid_size = 100 
        self.roberta = RobertaModel(config)
        self.mode = mode
        self.utter_classifier = RobertaClassificationHead(config, num_labels=3)
        if mode == 1: # mode 1: directlly classify with bert embedding;
            pass
        elif mode == 2: # mode 2: concat bert embedding and personality;
            self.personality_trans = nn.Linear(5, self.mid_size) # 5-d personality vec     
        elif mode == 3: # mode 3: personality-based emotion transition;
            self.init_trans = nn.Linear(3, 3)
            self.vad_para_trans = nn.Linear(3, 3) 
            self.vad_to_hidden = nn.Linear(3, config.hidden_size)
            self.classifier = RobertaClassificationHead_v3(config, num_labels=self.num_labels)

    def personality_to_vad(self, personality):
        O, C, E, A, N = personality[:, 0], personality[:, 1], personality[:, 2], personality[:, 3], personality[:, 4]
        valence = 0.21 * E + 0.59 * A + 0.19 * N
        arousal = 0.15 * O + 0.30 * A - 0.57 * N
        dominance = 0.25 * O + 0.17 * C + 0.60 * E - 0.32 * A
        # print(valence.unsqueeze(-1).shape)
        return torch.cat((valence.unsqueeze(-1), arousal.unsqueeze(-1), dominance.unsqueeze(-1)), 1)
    
    def forward(self, input_ids_1=None, input_ids_2=None, input_ids_3=None, attn_mask_1, attn_mask_2, attn_mask_3,
                      personality=None, init_emo=None):

        ## utterance robert embedding
        roberta_outputs_1 = self.roberta(input_ids_1, attention_mask = attn_mask_1)
        roberta_hidden_1 = roberta_outputs_1[0]

        roberta_outputs_2 = self.roberta(input_ids_2,  attention_mask = attn_mask_2)
        roberta_hidden_2 = roberta_outputs_2[0]

        roberta_hidden = torch.cat((roberta_hidden_1, roberta_hidden_2), 2)
        
        if self.mode == 1:
            logits = self.utter_classifier(roberta_hidden)
        elif self.mode == 2:
            personality = self.personality_trans(personality.cuda(device))
            logits = self.utter_classifier(roberta_hidden, personality)        
        elif self.mode == 3:
            utter_emo = self.utter_classifier(roberta_hidden) # delta of v, a, d
            personality_paras = self.personality_to_vad(personality) # p_v, p_a, p_d (\prime)
            personality_influence = self.vad_para_trans(personality_paras) # P_v, P_a, P_d 
            # init_emo = self.init_trans(init_emo) # v_i, a_i, d_i
            target_emo = init_emo + utter_emo * personality_influence
            hidden = self.vad_to_hidden(target_emo)
            logits = self.classifier(hidden)
        
        return logits