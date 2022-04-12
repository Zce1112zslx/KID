import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sparse


from utils import generate_perm_inv, batched_index_select, get_mask_from_index, masked_softmax

class Mlp(nn.Module):
    def __init__(self, input_size, output_size):
        super(Mlp, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        out = self.linear(x)
        return out


class Relu_Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Relu_Linear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.linear(x)
        return out

class PointerNet(nn.Module):
    def __init__(self, query_vec_size, src_encoding_size, attention_type='affine'):
        super(PointerNet, self).__init__()

        assert attention_type in ('affine', 'dot_prod')
        if attention_type == 'affine':
            self.src_encoding_linear = Mlp(src_encoding_size, query_vec_size)

        self.src_linear = Mlp(src_encoding_size,src_encoding_size)
        self.activate = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.fc =nn.Linear(src_encoding_size*2,src_encoding_size, bias=True)

        self.attention_type = attention_type

    def forward(self, src_encodings, src_token_mask,query_vec,head_vec=None):

        # (batch_size, 1, src_sent_len, query_vec_size)
        if self.attention_type == 'affine':
            src_encod = self.src_encoding_linear(src_encodings).unsqueeze(1)
            head_weights = self.src_linear(src_encodings).unsqueeze(1)

        # (batch_size, tgt_action_num, query_vec_size, 1)
        if head_vec is not None:
            src_encod = torch.cat([src_encod,head_weights],dim = -1)
            q = torch.cat([head_vec, query_vec], dim=-1).permute(1, 0, 2).unsqueeze(3)


        else:
            q = query_vec.permute(1, 0, 2).unsqueeze(3)

        weights = torch.matmul(src_encod, q).squeeze(3)
        ptr_weights = weights.permute(1, 0, 2)

        # if head_vec is not None:
        #     src_weights = torch.matmul(head_weights, q_h).squeeze(3)
        #     src_weights = src_weights.permute(1, 0, 2)
        #     ptr_weights = weights+src_weights
        #
        # else:
        #    ptr_weights = weights

        ptr_weights_masked = ptr_weights.clone().detach()
            # (tgt_action_num, batch_size, src_sent_len)
        src_token_mask= 1 - src_token_mask
        src_token_mask = src_token_mask.unsqueeze(0).expand_as(ptr_weights)
        src_token_mask = src_token_mask > 0
        # ptr_weights.data.masked_fill_(src_token_mask, -float('inf'))
        ptr_weights_masked.data.masked_fill_(src_token_mask, -float('inf'))


        return ptr_weights,ptr_weights_masked

class GCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.relu = nn.ReLU()


    def forward(self, x: torch.Tensor, adjacency_hat: torch.sparse_coo_tensor):
        x_l = self.linear(x)
        x_d = torch.sparse.mm(adjacency_hat, x_l)
        x = self.relu(x_d)
        return x

class RelGCNConv(nn.Module):
    def __init__(self, in_features, out_features, rel_num):
        super(RelGCNConv, self).__init__()

        self.linears = nn.ModuleList([nn.Linear(in_features, out_features, bias=False) for i in range(rel_num)])
        # self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.tanh = nn.Tanh()


    def forward(self, x: torch.Tensor, adjacency_list):
        x_list = []
        for i, adjacency_hat in enumerate(adjacency_list):
            x_l = self.linears[i](x)
            x_d = torch.sparse.mm(adjacency_hat, x_l)
            x_list.append(x_d.unsqueeze(0))
        x = self.tanh(torch.sum(torch.cat(x_list, dim=0), dim=0))
        return x

class RoleClassification(nn.Module):
    def __init__(self, q_features, k_features):
        super(RoleClassification, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(q_features, q_features),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(q_features, k_features),
            nn.Tanh()
        )
    
    def forward(self, q:torch.Tensor, k:torch.Tensor):
        Q = self.linear(q).unsqueeze(-1) #(B, k_features, 1)
        K = k.repeat([Q.shape[0], 1, 1]) #(B, 9634, k_features)
        weight = torch.bmm(K, Q).squeeze() # (B, 9634)
        return weight


# input: sentence
# output: hidden states of each sentence and target representation
class Encoder(nn.Module):
    def __init__(self, opt, config, word_embedding:nn.modules.sparse.Embedding,
                 lemma_embedding: nn.modules.sparse.Embedding):
        super(Encoder, self).__init__()
        self.opt =opt
        self.hidden_size = opt.rnn_hidden_size
        self.emb_size = opt.encoder_emb_size
        self.rnn_input_size = self.emb_size*2+opt.pos_emb_size+opt.token_type_emb_size
        self.word_number = config.word_number
        self.lemma_number = config.lemma_number
        self.maxlen = opt.maxlen


        self.dropout = 0.2
        self.word_embedding = word_embedding
        self.lemma_embedding = lemma_embedding
        self.pos_embedding = nn.Embedding(config.pos_number, opt.pos_emb_size)
        self.token_type_embedding = nn.Embedding(2, opt.token_type_emb_size)
        self.cell_name = opt.cell_name
        self.embed_dropout = nn.Dropout(0.5)


        self.target_linear = nn.Linear(2*self.hidden_size, 2*self.hidden_size)

        self.relu_linear = Relu_Linear(4*self.hidden_size+self.rnn_input_size, opt.node_emb_size)

        self.dep_gcn0 = GCNConv(2*self.hidden_size, 2*self.hidden_size)
        self.dep_gcn1 = GCNConv(2*self.hidden_size, 2*self.hidden_size)

        if self.cell_name == 'gru':
            self.rnn = nn.GRU(self.rnn_input_size, self.hidden_size,num_layers=self.opt.num_layers,
                              dropout=self.dropout,bidirectional=True, batch_first=True) 
        elif self.cell_name == 'lstm':
            self.rnn = nn.LSTM(self.rnn_input_size, self.hidden_size,num_layers=self.opt.num_layers,
                               dropout=self.dropout,bidirectional=True, batch_first=True)
        else:
            print('cell_name error')

    def forward(self, word_input: torch.Tensor, lemma_input: torch.Tensor, pos_input: torch.Tensor,
                lengths:torch.Tensor, frame_idx, dep_A, token_type_ids=None, attention_mask=None):
        # print(word_input)
        word_embedded = self.embed_dropout(self.word_embedding(word_input))
        lemma_embedded = self.embed_dropout(self.lemma_embedding(lemma_input))
        pos_embedded = self.embed_dropout(self.pos_embedding(pos_input))

        token_type_embedded = self.embed_dropout(self.token_type_embedding(token_type_ids))

        embedded = torch.cat([word_embedded, lemma_embedded, pos_embedded,token_type_embedded], dim=-1)

        lengths=lengths.squeeze()
        # sorted before pack
        l = lengths.cpu().numpy()
        perm_idx = np.argsort(-l)
        perm_idx_inv = generate_perm_inv(perm_idx)

        embedded = embedded[perm_idx]

        if lengths is not None:
            rnn_input = nn.utils.rnn.pack_padded_sequence(embedded, lengths=l[perm_idx],
                                                          batch_first=True)

        output, hidden = self.rnn(rnn_input)

        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=self.maxlen, batch_first=True)


        output = output[perm_idx_inv] # (B, T, H)

        if self.cell_name == 'gru':
            hidden = hidden[:, perm_idx_inv]
            hidden = (lambda a: sum(a)/(2*self.opt.num_layers))(torch.split(hidden, 1, dim=0))


        elif self.cell_name == 'lstm':
            hn0 = hidden[0][:, perm_idx_inv]
            hn1 = hidden[1][:, perm_idx_inv]
            hn  = tuple([hn0,hn1])
            hidden = tuple(map(lambda state: sum(torch.split(state, 1, dim=0))/(2*self.opt.num_layers), hn))

        batch_size = output.shape[0]
        dep_gcn_output = self.dep_gcn1(self.dep_gcn0(output.reshape(batch_size*self.maxlen, -1), dep_A), dep_A).reshape(batch_size, self.maxlen, -1)
        output = output + dep_gcn_output
        target_state_head = batched_index_select(target=output, indices=frame_idx[0])
        target_state_tail = batched_index_select(target=output, indices=frame_idx[1])
        target_state = (target_state_head + target_state_tail) / 2
        target_state = self.target_linear(target_state)

        target_emb_head = batched_index_select(target=embedded, indices=frame_idx[0])
        target_emb_tail = batched_index_select(target=embedded, indices=frame_idx[1])
        target_emb = (target_emb_head + target_emb_tail) / 2

        # Q: target_state K/V: hidden_state
        attentional_target_state = attention_layer(attention_mask=attention_mask, hidden_state=output,
                                                   target_state=target_state)


        target_state =torch.cat([target_state.squeeze(),attentional_target_state.squeeze(), target_emb.squeeze()], dim=-1)
        target = self.relu_linear(target_state)

        target_span = torch.cat([target_state_head + target_state_tail, target_state_head - target_state_tail], dim=-1)
        return output, hidden, target, target_span

class Decoder(nn.Module):
    def __init__(self, opt, embedding_frozen=False):
        super(Decoder, self).__init__()
        # rnn _init_
        self.opt = opt
        self.device = opt.cuda
        self.graph_name_list = ['self_loop', 'frame_fe', 'frame_frame', 'inter_fe', 'intra_fe']
        self.emb_size = opt.decoder_emb_size
        self.node_emb_size = opt.node_emb_size
        self.hidden_size = opt.decoder_hidden_size
        self.encoder_hidden_size = opt.rnn_hidden_size
        self.fe_dict = np.load(opt.fe_dict_path, allow_pickle=True).item()
        self.frame_embedding = nn.Embedding(opt.frame_number+1, opt.node_emb_size)
        self.role_embedding = nn.Embedding(opt.role_number+1, opt.node_emb_size)
        self.frame_number = opt.frame_number
        self.role_number = opt.role_number + 1
        self.embed_dropout = nn.Dropout(0.5)
        self.role_pred = RoleClassification(3*self.node_emb_size, self.node_emb_size)
        self.gcn_0 = GCNConv(2*self.node_emb_size, 2*self.node_emb_size)
        self.gcn_1 = GCNConv(2*self.node_emb_size, 2*self.node_emb_size)

        # decoder _init_
        self.decodelen = opt.fe_padding_num+1
        self.frame_pred = RoleClassification(self.node_emb_size, self.node_emb_size)
        self.role_fc_layer = nn.Linear(3*self.node_emb_size, opt.role_number+1)


        self.head_fc_layer = Mlp(2*self.node_emb_size, self.encoder_hidden_size)
        self.tail_fc_layer = Mlp(2*self.node_emb_size, self.encoder_hidden_size)

        self.span_fc_layer = Mlp(4 * self.encoder_hidden_size, self.node_emb_size)

        
        if embedding_frozen is True:
            for param in self.frame_embedding.parameters():
                param.requires_grad = False
            for param in self.role_embedding.parameters():
                param.requires_grad = False


        # pointer _init_
        self.ent_pointer = PointerNet(query_vec_size=self.hidden_size, src_encoding_size=2*self.encoder_hidden_size)
        self.head_pointer = PointerNet(query_vec_size=self.hidden_size, src_encoding_size=2*self.encoder_hidden_size)
        self.tail_pointer = PointerNet(query_vec_size=self.hidden_size, src_encoding_size=2*self.encoder_hidden_size)

        self.A_frame_list = []
        for graph_name in self.graph_name_list:
        # framenet graph adj matrix load...
            A_frame_coo = sparse.load_npz(opt.save_graph_path + graph_name+'.npz')
            if graph_name == 'self_loop':
                adj_size = torch.Size(A_frame_coo.shape)
            indices = torch.from_numpy(np.vstack((A_frame_coo.row, A_frame_coo.col)).astype(np.int64))
            values = torch.from_numpy(A_frame_coo.data)
            self.A_frame_list.append(torch.sparse_coo_tensor(indices, values, adj_size).to(self.device))
        self.frame_gcn_0 = RelGCNConv(self.node_emb_size, self.node_emb_size, len(self.graph_name_list))
        self.frame_gcn_1 = RelGCNConv(self.node_emb_size, self.node_emb_size, len(self.graph_name_list))

    def forward(self, encoder_output: torch.Tensor, target_span: torch.Tensor, target_state: torch.Tensor,
                attention_mask: torch.Tensor, gold_frame_id, frame_list, fe_list, adj_list, **kwargs):
        lu_mask = None
        if 'lu_mask' in kwargs:
            lu_mask = kwargs['lu_mask']
            if lu_mask is not None:
                lu_mask = lu_mask[:, :-1]
        teacher_enforcing = False
        if 'teacher_enforcing' in kwargs:
            teacher_enforcing = kwargs['teacher_enforcing']
        gold_fe_type = None
        gold_fe_head = None
        gold_fe_tail = None
        if 'gold_fe' in kwargs:
            gold_fe = kwargs['gold_fe']
            gold_fe_type = gold_fe[0]
            gold_fe_head = gold_fe[1]
            gold_fe_tail = gold_fe[2]
        pred_frame_list = []
        pred_head_list = []
        pred_tail_list = []
        pred_role_list = []

        pred_frame_action = []

        pred_head_action = []
        pred_tail_action = []
        pred_role_action = []


        all_frame_embeddings = self.frame_embedding(torch.arange(self.frame_number).to(self.device))
        all_fe_embeddings = self.role_embedding(torch.LongTensor([self.fe_dict[i] for i in range(self.role_number)]).to(self.device))
        all_embeddings = torch.cat([all_frame_embeddings, all_fe_embeddings], dim=0)
        frame_gcn_out = self.frame_gcn_1(self.frame_gcn_0(all_embeddings, self.A_frame_list), self.A_frame_list)
        all_embeddings = frame_gcn_out.unsqueeze(0) 
        batch_size = encoder_output.shape[0]
        target_node = self.span_fc_layer(target_span).squeeze()
        pred_frame_weight = self.frame_pred(target_node, all_embeddings)[:,:self.frame_number]
        pred_frame_list.append(pred_frame_weight)
        
        # train
        if lu_mask is None:
            gold_frame_one_hot = F.one_hot(gold_frame_id, num_classes=frame_list.shape[-1]).squeeze() # (B, 10)
            gold_frame_label = torch.sum(gold_frame_one_hot * frame_list, dim=-1) # (B, 1)
            pred_frame_action.append(gold_frame_label.squeeze())
            gold_frame_fe_list = torch.sum(gold_frame_one_hot.unsqueeze(-1) * fe_list, dim=1) # (B, 33)
            gold_frame_fe_one_hot = F.one_hot(gold_frame_fe_list, num_classes=self.role_number).squeeze() # (B, 33, 9635)
            frame_fe_mask = torch.sum(gold_frame_fe_one_hot, dim=1).squeeze() != 0 # (B, 9635) boolean
            gold_frame_label_one_hot = F.one_hot(gold_frame_label.squeeze(), all_embeddings.shape[1]).unsqueeze(-1)
            frame_node = self.embed_dropout(torch.sum(gold_frame_label_one_hot * all_embeddings, dim=1)) # (B, node_emb_size)
            
        # test
        else:
            pred_frame_weight_masked = pred_frame_weight.clone().detach()
            LU_mask = (lu_mask == 0) 
            pred_frame_weight_masked.data.masked_fill_(LU_mask, -float('inf'))
            pred_frame_indices = torch.argmax(pred_frame_weight_masked.squeeze(), dim=-1).squeeze()
            pred_frame_action.append(pred_frame_indices)
            pred_lu_one_hot = F.one_hot(pred_frame_indices, num_classes=lu_mask.shape[-1]).squeeze() # (B, 1019)
            pred_frame_id = torch.sum(pred_lu_one_hot * lu_mask, dim=-1) - 1
            pred_frame_one_hot = F.one_hot(pred_frame_id, num_classes=frame_list.shape[-1]).squeeze() # (B, 10)
            pred_frame_label = torch.sum(pred_frame_one_hot * frame_list, dim=-1) # (B, 1)
            pred_frame_fe_list = torch.sum(pred_frame_one_hot.unsqueeze(-1) * fe_list, dim=1) # (B, 33)
            pred_frame_fe_one_hot = F.one_hot(pred_frame_fe_list, num_classes=self.role_number).squeeze() # (B, 33, 9635)
            frame_fe_mask = torch.sum(pred_frame_fe_one_hot, dim=1).squeeze() != 0 # (B, 9635) boolean
            pred_frame_label_one_hot = F.one_hot(pred_frame_label.squeeze(), all_embeddings.shape[1]).unsqueeze(-1)
            frame_node = torch.sum(pred_frame_label_one_hot * all_embeddings, dim=1) # (B, node_emb_size)
        node_vec = torch.cat([frame_node, target_node], dim=-1).unsqueeze(1)


        span_mask = attention_mask.clone()
        for t in range(self.decodelen):

            adj_hat = adj_list[t]
            node_hidden = self.gcn_1(self.gcn_0(node_vec.reshape(batch_size*(t+1), -1), adj_hat), adj_hat).reshape(batch_size, t+1, -1)

            graph_vec = F.max_pool1d(node_hidden.permute(0, 2, 1), t+1).squeeze()


            # arg pred
            head_input = self.head_fc_layer(graph_vec)
            tail_input = self.tail_fc_layer(graph_vec)

            head_pointer_weight, head_pointer_weight_masked = self.head_pointer(src_encodings=encoder_output,
                                                                                src_token_mask=span_mask,
                                                                                query_vec=head_input.view(1, self.opt.batch_size, -1))

            head_indices = torch.argmax(head_pointer_weight_masked.squeeze(), dim=-1).squeeze()
            head_mask = head_mask_update(span_mask, head_indices=head_indices, max_len=self.opt.maxlen)

            tail_pointer_weight, tail_pointer_weight_masked = self.tail_pointer(src_encodings=encoder_output,
                                                                                src_token_mask=head_mask,
                                                                                query_vec=tail_input.view(1, self.opt.batch_size,-1))

            tail_indices = torch.argmax(tail_pointer_weight_masked.squeeze(), dim=-1).squeeze()
            if teacher_enforcing:
                span_mask = span_mask_update(attention_mask=span_mask, head_indices=gold_fe_head[:, t].unsqueeze(-1),
                                        tail_indices=gold_fe_tail[:, t].unsqueeze(-1), max_len=self.opt.maxlen)
            else:
                span_mask = span_mask_update(attention_mask=span_mask, head_indices=head_indices,
                                       tail_indices=tail_indices, max_len=self.opt.maxlen)

            pred_head_list.append(head_pointer_weight)
            pred_tail_list.append(tail_pointer_weight)
            pred_head_action.append(head_indices)
            pred_tail_action.append(tail_indices)
            if teacher_enforcing:
                head_target = batched_index_select(target=encoder_output, indices=gold_fe_head[:, t])
                tail_target = batched_index_select(target=encoder_output, indices=gold_fe_tail[:, t])
            else:
                head_target = batched_index_select(target=encoder_output, indices=head_indices.squeeze())
                tail_target = batched_index_select(target=encoder_output, indices=tail_indices.squeeze())
            span_input = self.span_fc_layer(torch.cat([head_target+tail_target, head_target-tail_target], dim=-1))

            # role pred
            
            role_weight = self.role_pred(torch.cat([graph_vec, span_input.squeeze()], dim=-1), all_embeddings)[:,self.frame_number:]
            role_weight_masked = role_weight.squeeze().clone().detach()
            role_weight_masked.data.masked_fill_(~frame_fe_mask, -float('inf'))
            role_indices = torch.argmax(role_weight_masked.squeeze(), dim=-1).squeeze() # (B, )
            if teacher_enforcing:
                role_emb = self.embed_dropout(self.role_embedding(gold_fe_type[:, t]))
            else:
                role_one_hot = F.one_hot(role_indices+self.frame_number, all_embeddings.shape[1]).unsqueeze(-1)
                role_emb = self.embed_dropout(torch.sum(role_one_hot * all_embeddings, dim=1))
            pred_role_action.append(role_indices)
            pred_role_list.append(role_weight)

            new_node = torch.cat([role_emb, span_input.squeeze()], dim=-1) # (B, 2*node_emb_size)
            node_vec = torch.cat([node_vec, new_node.unsqueeze(1)], dim=1)
        return pred_frame_list, pred_head_list, pred_tail_list, pred_role_list, pred_frame_action,\
                        pred_head_action, pred_tail_action, pred_role_action

 
    def decode_step(self, rnn_cell: nn.modules, input: torch.Tensor, decoder_state: torch.Tensor):

        output, state = rnn_cell(input.view(-1, 1, self.emb_size), decoder_state)

        return output, state




class Model(nn.Module):
    def __init__(self, opt, config, load_emb=True):
        super(Model, self).__init__()
        self.word_vectors = config.word_vectors
        self.lemma_vectors = config.lemma_vectors
        self.word_embedding = nn.Embedding(config.word_number, opt.encoder_emb_size, padding_idx=0)
        self.lemma_embedding = nn.Embedding(config.lemma_number, opt.encoder_emb_size, padding_idx=0)
        self.batch_size = opt.batch_size
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device(opt.cuda)

        if load_emb:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.word_vectors))
            self.lemma_embedding.weight.data.copy_(torch.from_numpy(self.lemma_vectors))
        
        self.encoder = Encoder(opt, config, self.word_embedding, self.lemma_embedding)
        self.decoder = Decoder(opt)


    def forward(self, word_ids, lemma_ids, pos_ids, lengths, frame_idx, token_type_ids, attention_mask, gold_frame_id, frame_list, fe_list, adj_list, dep_A, lu_mask=None):
        encoder_output, encoder_state, target_state, target_span = self.encoder(word_input=word_ids, lemma_input=lemma_ids,
                                                                   pos_input=pos_ids, lengths=lengths,
                                                                   frame_idx=frame_idx,
                                                                   dep_A = dep_A,
                                                                   token_type_ids=token_type_ids,
                                                                   attention_mask=attention_mask)
        pred_frame_list, pred_head_list, pred_tail_list, pred_role_list, pred_frame_action, \
        pred_head_action, pred_tail_action, pred_role_action = self.decoder(encoder_output=encoder_output,
                                                                            target_span=target_span,
                                                                            target_state=target_state,
                                                                            attention_mask=attention_mask,
                                                                            gold_frame_id = gold_frame_id,
                                                                            frame_list = frame_list,
                                                                            fe_list = fe_list,
                                                                            adj_list= adj_list,
                                                                            lu_mask=lu_mask)
        return {
            'pred_frame_list' : pred_frame_list,
            'pred_head_list' : pred_head_list,
            'pred_tail_list' : pred_tail_list,
            'pred_role_list' :  pred_role_list,
            'pred_frame_action' : pred_frame_action,
            'pred_head_action' : pred_head_action,
            'pred_tail_action' : pred_tail_action,
            'pred_role_action' : pred_role_action
        }         



def head_mask_update(attention_mask: torch.Tensor, head_indices: torch.Tensor, max_len):
    indices=head_indices
    indices_mask = get_mask_from_index(indices, max_len)
    mask = attention_mask & (~indices_mask)

    return mask


def span_mask_update(attention_mask: torch.Tensor, head_indices: torch.Tensor, tail_indices: torch.Tensor, max_len):
    tail = tail_indices + 1
    head_indices_mask = get_mask_from_index(head_indices, max_len)
    tail_indices_mask = get_mask_from_index(tail, max_len)
    span_indices_mask = tail_indices_mask ^ head_indices_mask
    mask = attention_mask & (~span_indices_mask)

    return mask
        
def attention_layer(attention_mask: torch.Tensor, hidden_state: torch.Tensor, target_state: torch.Tensor):
    q = target_state.squeeze().unsqueeze(2)
    context_att = torch.bmm(hidden_state, q).squeeze()

    mask = 1-attention_mask
    mask = mask > 0
    context_att = context_att.masked_fill_(mask, -float('inf'))
    context_att = F.softmax(context_att, dim=-1)
    attentional_hidden_state = torch.bmm(hidden_state.permute(0, 2, 1), context_att.unsqueeze(2)).squeeze()

    return attentional_hidden_state
