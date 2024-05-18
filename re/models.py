import torch
import numpy as np
from transformers import BertModel, BertPreTrainedModel

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        # set emb_name to the name of the embedding layer in your model
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings'):
        # set emb_name to the name of the embedding layer in your model
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            # special 'bert.pooler.dense.weight, bert.pooler.dense.bias'
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            # special 'bert.pooler.dense.weight, bert.pooler.dense.bias'
            if param.requires_grad and name in self.grad_backup:
                param.grad = self.grad_backup[name]

class BertForRelationMultiMention(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelationMultiMention, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size * 3)
        self.classifier = torch.nn.Linear(config.hidden_size * 3, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None, input_position=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]
        
        sub_output = torch.cat([a.index_select(0, idxs[idxs!=-1]).logsumexp(0).unsqueeze(0) for a, idxs in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a.index_select(0, idxs[idxs!=-1]).logsumexp(0).unsqueeze(0) for a, idxs in zip(sequence_output, obj_idx)])
        
        prod = torch.mul(sub_output, obj_output)
        
        rep1 = torch.cat((sub_output, obj_output, prod), dim=1)
        rep2 = torch.cat((obj_output, sub_output, prod), dim=1)
        
        rep1 = self.layer_norm(rep1)
        rep1 = self.dropout(rep1)
        logits1 = self.classifier(rep1)
        
        rep2 = self.layer_norm(rep2)
        rep2 = self.dropout(rep2)
        logits2 = self.classifier(rep2)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss1 = loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
            loss2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
            return loss1 + loss2, logits1 + logits2
        else:
            return logits1 + logits2
            
class BertForRelationMultiMentionAttention(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelationMultiMentionAttention, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size * 3)
        self.classifier = torch.nn.Linear(config.hidden_size * 3, self.num_labels)
        self.softmax = torch.nn.Softmax(dim = 0)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None, input_position=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]

        subs = torch.cat([torch.nn.functional.pad(a.index_select(0, idxs[idxs!=-1]), pad=(0, 0, 0, 32 - len(idxs[idxs!=-1]))).unsqueeze(0) for a, idxs in zip(sequence_output, sub_idx)])
        objs = torch.cat([torch.nn.functional.pad(a.index_select(0, idxs[idxs!=-1]), pad=(0, 0, 0, 32 - len(idxs[idxs!=-1]))).unsqueeze(0) for a, idxs in zip(sequence_output, obj_idx)])
        
        compatibility = torch.cat([torch.matmul(s, torch.transpose(o, 0, 1)).unsqueeze(0) for s, o in zip(subs, objs)])
        most_compatible = [np.unravel_index(torch.argmax(comp).detach().cpu().numpy(), comp.shape) for comp in compatibility]

        sub_output = torch.cat([a[c[0]].unsqueeze(0) for a, c in zip(subs, most_compatible)])
        obj_output = torch.cat([a[c[1]].unsqueeze(0) for a, c in zip(objs, most_compatible)])
        
        prod = torch.mul(sub_output, obj_output)
        
        rep1 = torch.cat((sub_output, obj_output, prod), dim=1)
        rep2 = torch.cat((obj_output, sub_output, prod), dim=1)
        
        rep1 = self.layer_norm(rep1)
        rep1 = self.dropout(rep1)
        logits1 = self.classifier(rep1)
        
        rep2 = self.layer_norm(rep2)
        rep2 = self.dropout(rep2)
        logits2 = self.classifier(rep2)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss1 = loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
            loss2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
            return loss1 + loss2, logits1 + logits2
        else:
            return logits1 + logits2
            
class BertForRelationMultiMentionFocalLoss(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelationMultiMentionFocalLoss, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size * 3)
        self.classifier = torch.nn.Linear(config.hidden_size * 3, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None, input_position=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]
        
        sub_output = torch.cat([a.index_select(0, idxs[idxs!=-1]).logsumexp(0).unsqueeze(0) for a, idxs in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a.index_select(0, idxs[idxs!=-1]).logsumexp(0).unsqueeze(0) for a, idxs in zip(sequence_output, obj_idx)])
        
        prod = torch.mul(sub_output, obj_output)
        
        rep1 = torch.cat((sub_output, obj_output, prod), dim=1)
        rep2 = torch.cat((obj_output, sub_output, prod), dim=1)
        
        rep1 = self.layer_norm(rep1)
        rep1 = self.dropout(rep1)
        logits1 = self.classifier(rep1)
        
        rep2 = self.layer_norm(rep2)
        rep2 = self.dropout(rep2)
        logits2 = self.classifier(rep2)

        # if labels is not None:
        #     loss_fct = torch.nn.CrossEntropyLoss()
        #     loss1 = loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
        #     loss2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
        #     return loss1 + loss2, logits1 + logits2
        # else:
        return logits1, logits2
            
class BertForRelationMultiMentionAttentionFocalLoss(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelationMultiMentionAttentionFocalLoss, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size * 3)
        self.classifier = torch.nn.Linear(config.hidden_size * 3, self.num_labels)
        self.softmax = torch.nn.Softmax(dim = 0)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None, input_position=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]

        subs = torch.cat([torch.nn.functional.pad(a.index_select(0, idxs[idxs!=-1]), pad=(0, 0, 0, 32 - len(idxs[idxs!=-1]))).unsqueeze(0) for a, idxs in zip(sequence_output, sub_idx)])
        objs = torch.cat([torch.nn.functional.pad(a.index_select(0, idxs[idxs!=-1]), pad=(0, 0, 0, 32 - len(idxs[idxs!=-1]))).unsqueeze(0) for a, idxs in zip(sequence_output, obj_idx)])
        
        compatibility = torch.cat([torch.matmul(s, torch.transpose(o, 0, 1)).unsqueeze(0) for s, o in zip(subs, objs)])
        most_compatible = [np.unravel_index(torch.argmax(comp).detach().cpu().numpy(), comp.shape) for comp in compatibility]

        sub_output = torch.cat([a[c[0]].unsqueeze(0) for a, c in zip(subs, most_compatible)])
        obj_output = torch.cat([a[c[1]].unsqueeze(0) for a, c in zip(objs, most_compatible)])
        
        prod = torch.mul(sub_output, obj_output)
        
        rep1 = torch.cat((sub_output, obj_output, prod), dim=1)
        rep2 = torch.cat((obj_output, sub_output, prod), dim=1)
        
        rep1 = self.layer_norm(rep1)
        rep1 = self.dropout(rep1)
        logits1 = self.classifier(rep1)
        
        rep2 = self.layer_norm(rep2)
        rep2 = self.dropout(rep2)
        logits2 = self.classifier(rep2)

        # if labels is not None:
        #     loss_fct = torch.nn.CrossEntropyLoss()
        #     loss1 = loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
        #     loss2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
        #     return loss1 + loss2, logits1 + logits2
        # else:
        return logits1, logits2