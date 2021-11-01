import os
import sys
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import transformers

from tqdm import tqdm


class ContextualRank(nn.Module):

    def __init__(
            self,
            max_length=200,
            context_type='bert-base-uncased',
            loss_type='pair'
    ):

        super(ContextualRank, self).__init__()
        self.max_length = max_length
        self.loss_type = loss_type
        self.context_type = context_type

        self.tokenizer = transformers.BertTokenizer.from_pretrained(context_type)
        self.bert = transformers.BertModel.from_pretrained(context_type)

        self.enc_config = transformers.BertConfig(num_hidden_layers=1)
        self.dropout = nn.Dropout(self.enc_config.hidden_dropout_prob)
        self.cls = nn.Linear(self.enc_config.hidden_size, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.cls.weight, -initrange, initrange)

    def forward(
            self,
            data,
            labels=None
    ):
        batch_data = []
        for x in data:
            doc_list = []
            for d in x[1]:
                text_left = x[0]
                text_right = d
                inputs = self.tokenizer(text=text_left,
                                        text_pair=text_right,
                                        return_tensors="pt",
                                        truncation='longest_first',
                                        padding='max_length',
                                        max_length=self.max_length,
                                        )
                inputs = inputs.to(self.bert.device)
                outputs = self.bert(**inputs, return_dict=True)
                doc_list.append(outputs.pooler_output)
            doc_rep = torch.cat(doc_list)
            batch_data.append(doc_rep)
        batch_data = torch.stack(batch_data)
        labels = labels.to(self.bert.device)
        # print(batch_data.shape)
        # print(labels.shape)

        enc_outputs = self.dropout(batch_data)
        logits = self.cls(enc_outputs).squeeze(dim=-1)

        loss = None
        if labels is not None:
            # get mask
            mask = (labels == -1).to(torch.float32)

            # label smooth
            if self.loss_type == 'attn':
                mask_label = (labels == 0).to(torch.float32) + (labels == -1).to(torch.float32)
                labels = labels + mask_label * (-1e6)
                labels_smooth = nn.functional.softmax(labels, dim=1)
                logits_fixed = logits + mask * (-1e6)
                loss_each = -nn.functional.log_softmax(logits_fixed, dim=1) * labels_smooth
                loss = loss_each.sum(dim=1).mean()
                # input()
            elif self.loss_type == 'pair':
                labels_matrix = torch.unsqueeze(labels, 1) - torch.unsqueeze(labels, 2)
                labels_matrix = (labels_matrix != 0).float() * labels_matrix
                scores_matrix = torch.unsqueeze(logits, 1) - torch.unsqueeze(logits, 2)
                mask_r = 1.0 - mask
                mask_matrix = torch.unsqueeze(mask_r, 1) * torch.unsqueeze(mask_r, 2)
                filter_matrix = labels_matrix * mask_matrix
                diff_matrix = scores_matrix * filter_matrix
                filter_abs_matrix = filter_matrix.abs()
                loss_matrix = filter_abs_matrix * torch.max(torch.Tensor([0.0]).to(labels.device), 1.0 - diff_matrix)
                loss = loss_matrix.sum()

        return logits, loss


class ContextualSetRank(nn.Module):

    def __init__(
            self,
            max_length=200,
            context_type='bert-base-uncased',
            loss_type='pair',
            num_hidden_layers=1
    ):

        super(ContextualSetRank, self).__init__()
        self.max_length = max_length
        self.loss_type = loss_type
        self.context_type = context_type
        self.num_hidden_layers = num_hidden_layers

        # self.tokenizer = transformers.BertTokenizer.from_pretrained(context_type)
        # self.bert = transformers.BertModel.from_pretrained(context_type)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(context_type)
        self.bert = transformers.AutoModel.from_pretrained(context_type)

        self.enc_config = transformers.BertConfig(num_hidden_layers=self.num_hidden_layers)
        self.enc = transformers.modeling_bert.BertEncoder(config=self.enc_config)
        self.dropout = nn.Dropout(self.enc_config.hidden_dropout_prob)
        self.cls = nn.Linear(self.enc_config.hidden_size, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.cls.weight, -initrange, initrange)

    def forward(
            self,
            data,
            labels=None
    ):
        batch_data = []
        for x in data:
            doc_list = []
            # print(x[1])
            for d in x[1]:
                # if len(d) == 0:
                # print("kong")
                # continue
                text_left = x[0]
                # print(text_left)
                text_right = d
                inputs = self.tokenizer(text=text_left,
                                        text_pair=text_right,
                                        return_tensors="pt",
                                        truncation='longest_first',
                                        padding='max_length',
                                        max_length=self.max_length,
                                        )
                inputs = inputs.to(self.bert.device)
                # print(self.bert.device)
                outputs = self.bert(**inputs, return_dict=True)
                doc_list.append(outputs.pooler_output)
            doc_rep = torch.cat(doc_list)
            batch_data.append(doc_rep)
        batch_data = torch.stack(batch_data)
        labels = labels.to(self.bert.device)
        # print(batch_data.shape)
        # print(labels.shape)
        # print(batch_data.shape)
        enc_outputs = self.enc(hidden_states=batch_data)[0]
        enc_outputs = self.dropout(enc_outputs)
        logits = self.cls(enc_outputs).squeeze(dim=-1)

        loss = None
        if labels is not None:
            # get mask
            mask = (labels == -1).to(torch.float32)

            # label smooth
            if self.loss_type == 'attn':
                mask_label = (labels == 0).to(torch.float32) + (labels == -1).to(torch.float32)
                labels = labels + mask_label * (-1e6)
                labels_smooth = nn.functional.softmax(labels, dim=1)
                print(labels_smooth)
                logits_fixed = logits + mask * (-1e6)
                print(logits_fixed)
                loss_each = -nn.functional.log_softmax(logits_fixed, dim=1) * labels_smooth
                loss = loss_each.sum(dim=1).mean()
                # input()
            elif self.loss_type == 'pair':
                labels_matrix = torch.unsqueeze(labels, 1) - torch.unsqueeze(labels, 2)
                # print(labels_matrix.dtype)
                # print((labels_matrix != 0).float().dtype)
                labels_matrix = (labels_matrix != 0).float() * labels_matrix
                # print(labels_matrix)
                scores_matrix = torch.unsqueeze(logits, 1) - torch.unsqueeze(logits, 2)
                # print(scores_matrix)
                mask_r = 1.0 - mask
                mask_matrix = torch.unsqueeze(mask_r, 1) * torch.unsqueeze(mask_r, 2)
                # print(mask_matrix)
                filter_matrix = labels_matrix * mask_matrix
                diff_matrix = scores_matrix * filter_matrix
                # print(diff_matrix)
                filter_abs_matrix = filter_matrix.abs()
                loss_matrix = filter_abs_matrix * torch.max(torch.Tensor([0.0]).to(labels.device), 1.0 - diff_matrix)
                # print(loss_matrix)
                loss = loss_matrix.sum()
                # print(loss)
                # input()

        return logits, loss


class ContextualRankWithLTR(nn.Module):

    def __init__(
            self,
            max_length=200,
            ltr_dim=46,
            context_type='bert-base-uncased',
            loss_type='pair'
    ):

        super(ContextualRankWithLTR, self).__init__()
        self.max_length = max_length
        self.loss_type = loss_type
        self.context_type = context_type

        self.tokenizer = transformers.BertTokenizer.from_pretrained(context_type)
        self.bert = transformers.BertModel.from_pretrained(context_type)

        self.enc_config = transformers.BertConfig(num_hidden_layers=1)
        self.dropout = nn.Dropout(self.enc_config.hidden_dropout_prob)
        self.dim = self.enc_config.hidden_size + ltr_dim
        self.cls = nn.Linear(self.dim, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.cls.weight, -initrange, initrange)

    def forward(
            self,
            data,
            ltr,
            labels=None
    ):
        batch_data = []
        for x in data:
            doc_list = []
            for d in x[1]:
                text_left = x[0]
                text_right = d
                inputs = self.tokenizer(text=text_left,
                                        text_pair=text_right,
                                        return_tensors="pt",
                                        truncation='longest_first',
                                        padding='max_length',
                                        max_length=self.max_length,
                                        )
                inputs = inputs.to(self.bert.device)
                outputs = self.bert(**inputs, return_dict=True)
                doc_list.append(outputs.pooler_output)
            doc_rep = torch.cat(doc_list)
            batch_data.append(doc_rep)
        batch_data = torch.stack(batch_data)
        labels = labels.to(self.bert.device)
        ltr = ltr.to(self.bert.device)
        # print(batch_data.shape)
        # print(labels.shape)

        enc_outputs = self.dropout(batch_data)
        # 加上ltr特征
        # print(enc_outputs.shape, ltr.shape)
        inputs_with_LTR = torch.cat((enc_outputs, ltr), 2)
        logits = self.cls(inputs_with_LTR).squeeze(dim=-1)

        loss = None
        if labels is not None:
            # get mask
            mask = (labels == -1).to(torch.float32)

            # label smooth
            if self.loss_type == 'attn':
                mask_label = (labels == 0).to(torch.float32) + (labels == -1).to(torch.float32)
                labels = labels + mask_label * (-1e6)
                labels_smooth = nn.functional.softmax(labels, dim=1)
                print(labels_smooth)
                logits_fixed = logits + mask * (-1e6)
                print(logits_fixed)
                loss_each = -nn.functional.log_softmax(logits_fixed, dim=1) * labels_smooth
                loss = loss_each.sum(dim=1).mean()
                # input()
            elif self.loss_type == 'pair':
                labels_matrix = torch.unsqueeze(labels, 1) - torch.unsqueeze(labels, 2)
                labels_matrix = (labels_matrix != 0).float() * labels_matrix
                # print(labels_matrix)
                scores_matrix = torch.unsqueeze(logits, 1) - torch.unsqueeze(logits, 2)
                # print(scores_matrix)
                mask_r = 1.0 - mask
                mask_matrix = torch.unsqueeze(mask_r, 1) * torch.unsqueeze(mask_r, 2)
                # print(mask_matrix)
                filter_matrix = labels_matrix * mask_matrix
                diff_matrix = scores_matrix * filter_matrix
                # print(diff_matrix)
                filter_abs_matrix = filter_matrix.abs()
                loss_matrix = filter_abs_matrix * torch.max(torch.Tensor([0.0]).to(labels.device), 1.0 - diff_matrix)
                # print(loss_matrix)
                loss = loss_matrix.sum()
                # print(loss)
                # input()

        return logits, loss


class ContextualSetRankWithLTR(nn.Module):

    def __init__(
            self,
            max_length=200,
            ltr_dim=46,
            context_type='bert-base-uncased',
            loss_type='pair',
            num_hidden_layers=1
    ):

        super(ContextualSetRankWithLTR, self).__init__()
        self.max_length = max_length
        self.loss_type = loss_type
        self.context_type = context_type
        self.num_hidden_layers = num_hidden_layers

        # self.tokenizer = transformers.BertTokenizer.from_pretrained(context_type)
        # self.bert = transformers.BertModel.from_pretrained(context_type)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(context_type)
        self.bert = transformers.AutoModel.from_pretrained(context_type)

        self.enc_config = transformers.BertConfig(num_hidden_layers=self.num_hidden_layers)
        # self.encoder = transformers.modeling_bert.BertEncoder(config=self.enc_config)
        self.enc = transformers.modeling_bert.BertEncoder(config=self.enc_config)
        self.dropout = nn.Dropout(self.enc_config.hidden_dropout_prob)
        self.dim = self.enc_config.hidden_size + ltr_dim
        self.cls_1 = nn.Linear(self.dim, 1)
        # self.cls = nn.Linear(self.enc_config.hidden_size, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.cls_1.weight, -initrange, initrange)

    def forward(
            self,
            data,
            ltr,
            labels=None
    ):
        batch_data = []
        for x in data:
            doc_list = []
            # print(x[1])
            for d in x[1]:
                # if len(d) == 0:
                # print("kong")
                # continue
                text_left = x[0]
                # print(text_left)
                text_right = d
                inputs = self.tokenizer(text=text_left,
                                        text_pair=text_right,
                                        return_tensors="pt",
                                        truncation='longest_first',
                                        padding='max_length',
                                        max_length=self.max_length,
                                        )
                inputs = inputs.to(self.bert.device)
                # print(self.bert.device)
                outputs = self.bert(**inputs, return_dict=True)
                doc_list.append(outputs.pooler_output)
            doc_rep = torch.cat(doc_list)
            batch_data.append(doc_rep)
        batch_data = torch.stack(batch_data)
        labels = labels.to(self.bert.device)
        ltr = ltr.to(self.bert.device)
        # print(batch_data.shape)
        # print(labels.shape)
        # print(batch_data.shape)
        # enc_outputs = self.encoder(hidden_states=batch_data)[0]
        enc_outputs = self.enc(hidden_states=batch_data)[0]
        enc_outputs = self.dropout(enc_outputs)
        inputs_with_LTR = torch.cat((enc_outputs, ltr), 2)
        logits = self.cls_1(inputs_with_LTR).squeeze(dim=-1)

        # logits = self.cls(enc_outputs).squeeze(dim=-1)

        loss = None
        if labels is not None:
            # get mask
            mask = (labels == -1).to(torch.float32)

            # label smooth
            if self.loss_type == 'attn':
                mask_label = (labels == 0).to(torch.float32) + (labels == -1).to(torch.float32)
                labels = labels + mask_label * (-1e6)
                labels_smooth = nn.functional.softmax(labels, dim=1)
                print(labels_smooth)
                logits_fixed = logits + mask * (-1e6)
                print(logits_fixed)
                loss_each = -nn.functional.log_softmax(logits_fixed, dim=1) * labels_smooth
                loss = loss_each.sum(dim=1).mean()
                # input()
            elif self.loss_type == 'pair':
                labels_matrix = torch.unsqueeze(labels, 1) - torch.unsqueeze(labels, 2)
                # print(labels_matrix.dtype)
                # print((labels_matrix != 0).float().dtype)
                labels_matrix = (labels_matrix != 0).float() * labels_matrix
                # print(labels_matrix)
                scores_matrix = torch.unsqueeze(logits, 1) - torch.unsqueeze(logits, 2)
                # print(scores_matrix)
                mask_r = 1.0 - mask
                mask_matrix = torch.unsqueeze(mask_r, 1) * torch.unsqueeze(mask_r, 2)
                # print(mask_matrix)
                filter_matrix = labels_matrix * mask_matrix
                diff_matrix = scores_matrix * filter_matrix
                # print(diff_matrix)
                filter_abs_matrix = filter_matrix.abs()
                loss_matrix = filter_abs_matrix * torch.max(torch.Tensor([0.0]).to(labels.device), 1.0 - diff_matrix)
                # print(loss_matrix)
                loss = loss_matrix.sum()
                # print(loss)
                # input()

        return logits, loss


class ContextualSetRankWithLTR_V2(nn.Module):

    def __init__(
            self,
            max_length=200,
            ltr_dim=46,
            context_type='bert-base-uncased',
            loss_type='pair',
            num_hidden_layers=1
    ):

        super(ContextualSetRankWithLTR_V2, self).__init__()
        self.max_length = max_length
        self.loss_type = loss_type
        self.context_type = context_type
        self.num_hidden_layers = num_hidden_layers
        self.dim = 768 + ltr_dim + 2

        # self.tokenizer = transformers.BertTokenizer.from_pretrained(context_type)
        # self.bert = transformers.BertModel.from_pretrained(context_type)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(context_type)
        self.bert = transformers.AutoModel.from_pretrained(context_type)

        self.enc_config = transformers.BertConfig(num_hidden_layers=self.num_hidden_layers, hidden_size=self.dim)
        self.encoder = transformers.modeling_bert.BertEncoder(config=self.enc_config)
        self.dropout = nn.Dropout(self.enc_config.hidden_dropout_prob)
        self.cls_1 = nn.Linear(self.enc_config.hidden_size, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.cls_1.weight, -initrange, initrange)

    def forward(
            self,
            data,
            ltr,
            labels=None
    ):
        # print("data:", len(data), "ltr:", len(ltr), "labels:", len(labels))
        batch_data = []
        for x in (data):
            doc_list = []
            # print(x[1])
            for d in x[1]:
                # if len(d) == 0:
                # print("kong")
                # continue
                text_left = x[0]
                # print(text_left)
                text_right = d
                inputs = self.tokenizer(text=text_left,
                                        text_pair=text_right,
                                        return_tensors="pt",
                                        truncation='longest_first',
                                        padding='max_length',
                                        max_length=self.max_length,
                                        )
                inputs = inputs.to(self.bert.device)
                # print(self.bert.device)
                outputs = self.bert(**inputs, return_dict=True)
                # print(outputs.pooler_output.shape) #测试长度 [1, 768]
                doc_list.append(outputs.pooler_output)
            doc_rep = torch.cat(doc_list)  # 测试长度 doc_rep torch.Size([20, 768])
            batch_data.append(doc_rep)
        batch_data = torch.stack(batch_data)
        # print("batch_data", batch_data.shape) #batch_data torch.Size([1, 20, 768])
        ltr = ltr.to(self.bert.device)
        # print("ltr：",ltr.shape[1])
        zeros = torch.zeros((1, ltr.shape[1], 2))  # 好不优雅（也没考虑batch的问题
        zeros = zeros.to(self.bert.device)
        # print("zeros", zeros.shape)
        inputs_with_LTR = torch.cat((batch_data, ltr), 2)  # [20, 46]
        inputs_with_LTR = torch.cat((inputs_with_LTR, zeros), 2)
        # print("inputs_with_LTR", inputs_with_LTR.shape)
        labels = labels.to(self.bert.device)
        # print(batch_data.shape)
        # print(labels.shape)
        # print(batch_data.shape)
        # enc_outputs = self.encoder(hidden_states=batch_data)[0]
        enc_outputs = self.encoder(hidden_states=inputs_with_LTR)[0]
        enc_outputs = self.dropout(enc_outputs)
        logits = self.cls_1(enc_outputs).squeeze(dim=-1)

        loss = None
        if labels is not None:
            # get mask
            mask = (labels == -1).to(torch.float32)

            # label smooth
            if self.loss_type == 'attn':
                mask_label = (labels == 0).to(torch.float32) + (labels == -1).to(torch.float32)
                labels = labels + mask_label * (-1e6)
                labels_smooth = nn.functional.softmax(labels, dim=1)
                # print(labels_smooth)
                logits_fixed = logits + mask * (-1e6)
                # print(logits_fixed)
                loss_each = -nn.functional.log_softmax(logits_fixed, dim=1) * labels_smooth
                loss = loss_each.sum(dim=1).mean()
                # input()
            elif self.loss_type == 'pair':
                labels_matrix = torch.unsqueeze(labels, 1) - torch.unsqueeze(labels, 2)
                # print(labels_matrix.dtype)
                # print((labels_matrix != 0).float().dtype)
                labels_matrix = (labels_matrix != 0).float() * labels_matrix
                # print(labels_matrix)
                scores_matrix = torch.unsqueeze(logits, 1) - torch.unsqueeze(logits, 2)
                # print(scores_matrix)
                mask_r = 1.0 - mask
                mask_matrix = torch.unsqueeze(mask_r, 1) * torch.unsqueeze(mask_r, 2)
                # print(mask_matrix)
                filter_matrix = labels_matrix * mask_matrix
                diff_matrix = scores_matrix * filter_matrix
                # print(diff_matrix)
                filter_abs_matrix = filter_matrix.abs()
                loss_matrix = filter_abs_matrix * torch.max(torch.Tensor([0.0]).to(labels.device), 1.0 - diff_matrix)
                # print(loss_matrix)
                loss = loss_matrix.sum()
                # print(loss)
                # input()

        return logits, loss


class OnlyLTR(nn.Module):

    def __init__(
            self,
            max_length=200,
            ltr_dim=46,
            context_type='bert-base-uncased',
            loss_type='pair',
            num_hidden_layers=1
    ):

        super(OnlyLTR, self).__init__()
        self.max_length = max_length
        self.loss_type = loss_type
        self.context_type = context_type
        self.num_hidden_layers = num_hidden_layers
        # self.dim = 768 + ltr_dim + 2
        # self.dim = ltr_dim + 2
        self.dim = ltr_dim + 3

        # self.tokenizer = transformers.BertTokenizer.from_pretrained(context_type)
        # self.bert = transformers.BertModel.from_pretrained(context_type)
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(context_type)
        self.bert = transformers.AutoModel.from_pretrained(context_type)

        self.enc_config = transformers.BertConfig(num_hidden_layers=self.num_hidden_layers, hidden_size=self.dim)
        self.encoder = transformers.modeling_bert.BertEncoder(config=self.enc_config)
        self.dropout = nn.Dropout(self.enc_config.hidden_dropout_prob)
        self.cls_1 = nn.Linear(self.enc_config.hidden_size, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.cls_1.weight, -initrange, initrange)

    def forward(
            self,
            data,
            ltr,
            labels=None
    ):
        # print("data:", len(data), "ltr:", len(ltr), "labels:", len(labels))
        # batch_data = []
        # for x in (data):
        #     doc_list = []
        #     # print(x[1])
        #     for d in x[1]:
        #         # if len(d) == 0:
        #         # print("kong")
        #         # continue
        #         text_left = x[0]
        #         # print(text_left)
        #         text_right = d
        #         inputs = self.tokenizer(text=text_left,
        #                                 text_pair=text_right,
        #                                 return_tensors="pt",
        #                                 truncation='longest_first',
        #                                 padding='max_length',
        #                                 max_length=self.max_length,
        #                                 )
        #         inputs = inputs.to(self.bert.device)
        #         # print(self.bert.device)
        #         outputs = self.bert(**inputs, return_dict=True)
        #         # print(outputs.pooler_output.shape) #测试长度 [1, 768]
        #         doc_list.append(outputs.pooler_output)
        #     doc_rep = torch.cat(doc_list)  # 测试长度 doc_rep torch.Size([20, 768])
        #     batch_data.append(doc_rep)
        # batch_data = torch.stack(batch_data)
        # print("batch_data", batch_data.shape) #batch_data torch.Size([1, 20, 768])
        ltr = ltr.to(self.bert.device)
        # print("ltr：",ltr.shape[1])
        # zeros = torch.zeros((1, ltr.shape[1], 2))  # 好不优雅（也没考虑batch的问题
        zeros = torch.zeros((1, ltr.shape[1], 3))
        zeros = zeros.to(self.bert.device)
        # print("zeros", zeros.shape)
        # inputs_with_LTR = torch.cat((batch_data, ltr), 2)  # [20, 46]
        # inputs_with_LTR = torch.cat((inputs_with_LTR, zeros), 2)
        inputs_with_LTR = torch.cat((ltr, zeros), 2)
        # print("inputs_with_LTR", inputs_with_LTR.shape)
        labels = labels.to(self.bert.device)
        # print(batch_data.shape)
        # print(labels.shape)
        # print(batch_data.shape)
        # enc_outputs = self.encoder(hidden_states=batch_data)[0]
        enc_outputs = self.encoder(hidden_states=inputs_with_LTR)[0]
        enc_outputs = self.dropout(enc_outputs)
        logits = self.cls_1(enc_outputs).squeeze(dim=-1)

        loss = None
        if labels is not None:
            # get mask
            mask = (labels == -1).to(torch.float32)

            # label smooth
            if self.loss_type == 'attn':
                mask_label = (labels == 0).to(torch.float32) + (labels == -1).to(torch.float32)
                labels = labels + mask_label * (-1e6)
                labels_smooth = nn.functional.softmax(labels, dim=1)
                print(labels_smooth)
                logits_fixed = logits + mask * (-1e6)
                print(logits_fixed)
                loss_each = -nn.functional.log_softmax(logits_fixed, dim=1) * labels_smooth
                loss = loss_each.sum(dim=1).mean()
                # input()
            elif self.loss_type == 'pair':
                labels_matrix = torch.unsqueeze(labels, 1) - torch.unsqueeze(labels, 2)
                # print(labels_matrix.dtype)
                # print((labels_matrix != 0).float().dtype)
                labels_matrix = (labels_matrix != 0).float() * labels_matrix
                # print(labels_matrix)
                scores_matrix = torch.unsqueeze(logits, 1) - torch.unsqueeze(logits, 2)
                # print(scores_matrix)
                mask_r = 1.0 - mask
                mask_matrix = torch.unsqueeze(mask_r, 1) * torch.unsqueeze(mask_r, 2)
                # print(mask_matrix)
                filter_matrix = labels_matrix * mask_matrix
                diff_matrix = scores_matrix * filter_matrix
                # print(diff_matrix)
                filter_abs_matrix = filter_matrix.abs()
                loss_matrix = filter_abs_matrix * torch.max(torch.Tensor([0.0]).to(labels.device), 1.0 - diff_matrix)
                # print(loss_matrix)
                loss = loss_matrix.sum()
                # print(loss)
                # input()

        return logits, loss
