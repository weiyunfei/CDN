import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class CDN(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers_hopd=3, num_dec_layers_interaction=3, 
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        vanilla_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        hoi_encoder_layer = EncoderLayerBottleneck(d_model, nhead, dim_feedforward, 
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = EncoderHOI(vanilla_encoder_layer, 4, hoi_encoder_layer, 2, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_dec_layers_hopd, decoder_norm,
                                          return_intermediate=return_intermediate_dec)


        interaction_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        interaction_decoder_norm = nn.LayerNorm(d_model)
        self.interaction_decoder = TransformerDecoder(interaction_decoder_layer, num_dec_layers_interaction, interaction_decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory, memory_inst, memory_verb = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hopd_out = self.decoder(tgt, memory_inst, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        hopd_out = hopd_out.transpose(1, 2)


        interaction_query_embed = hopd_out[-1]
        interaction_query_embed = interaction_query_embed.permute(1, 0, 2)

        interaction_tgt = torch.zeros_like(interaction_query_embed)
        interaction_decoder_out = self.interaction_decoder(interaction_tgt, memory_verb, memory_key_padding_mask=mask,
                                  pos=pos_embed, query_pos=interaction_query_embed)
        interaction_decoder_out = interaction_decoder_out.transpose(1, 2)

        return hopd_out, interaction_decoder_out, memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class EncoderHOI(nn.Module):

    def __init__(self, vanilla_encoder_layer=TransformerEncoderLayer, num_vanilla_layers=4,
                 hoi_encoder_layer=EncoderLayerBottleneck, num_hoi_layers=2, normalize_before=False):
        super().__init__()
        self.layers = get_clones(vanilla_encoder_layer, num_vanilla_layers)
        self.hoi_layers = get_clones(hoi_encoder_layer, num_hoi_layers)

        self.norm = self.norm_sub = self.norm_obj = self.norm_verb = None
        if normalize_before:
            self.norm = nn.LayerNorm(d_model)
            self.norm_inst = nn.LayerNorm(d_model)
            self.norm_verb = nn.LayerNorm(d_model)

    def eye_init(self):
        for layer in self.hoi_layers:
            layer.eye_init()

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, pos=pos,
                           src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        embed_inst = embed_verb = output
        for layer in self.hoi_layers:
            embed_inst, embed_verb = layer(
                embed_inst, embed_verb, src_mask=mask, pos=pos,
                src_key_padding_mask=src_key_padding_mask)
        if self.norm_inst is not None:
            embed_inst = self.norm_inst(embed_inst)
            embed_verb = self.norm_verb(embed_verb)

        return output, embed_inst, embed_verb


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class EncoderLayerBottleneck(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=256,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False,
                 **kwargs):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        self.inst_proj = nn.Linear(d_model, d_model)
        self.verb_proj = nn.Linear(d_model, d_model)

        self.cross_attn_verb = nn.MultiheadAttention(d_model, nhead, dropout)
        self.norm_verb = nn.LayerNorm(d_model)
        self.dropout_verb = nn.Dropout(dropout)
        # self.reduce_dim = nn.Linear(2 * d_model, d_model)

        self.linear1 = self.dropout = self.linear2 = None
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def eye_init(self):
        nn.init.eye_(self.inst_proj.weight)
        nn.init.constant_(self.inst_proj.bias, 0)
        nn.init.eye_(self.verb_proj.weight)
        nn.init.constant_(self.verb_proj.bias, 0)

    def forward_cross_attn(self, src_inst, src_verb, pos, src_key_padding_mask):
        embed_inst = src_inst
        embed_obj = src_obj

        # dst_verb = torch.cat([embed_sub, embed_obj], dim=-1)
        # dst_verb = self.reduce_dim(dst_verb)
        if self.normalize_before:
            embed_verb = self.cross_attn_pre(src=src_verb, dst=embed_inst, tgt='verb', pos=pos,
                                             dst_key_padding_mask=src_key_padding_mask)
        else:
            embed_verb = self.cross_attn_post(src=src_verb, dst=embed_inst, tgt='verb', pos=pos,
                                              dst_key_padding_mask=src_key_padding_mask)
        return embed_inst, embed_verb

    def cross_attn_post(self, src, dst, tgt,
                        dst_mask: Optional[Tensor] = None,
                        dst_key_padding_mask: Optional[Tensor] = None,
                        pos: Optional[Tensor] = None):
        src2 = getattr(self, f'cross_attn_{tgt}')(
            query=self.with_pos_embed(src, pos),
            key=self.with_pos_embed(dst, pos),
            value=dst, attn_mask=dst_mask,
            key_padding_mask=dst_key_padding_mask)[0]
        src = getattr(self, f'norm_{tgt}')(src + getattr(self, f'dropout_{tgt}')(src2))

        return src

    def cross_attn_pre(self, src, dst, tgt,
                       dst_mask: Optional[Tensor] = None,
                       dst_key_padding_mask: Optional[Tensor] = None,
                       pos: Optional[Tensor] = None):
        src2 = getattr(self, f'cross_attn_{tgt}')(
            query=self.with_pos_embed(getattr(self, f'norm_{tgt}')(src), pos),
            key=self.with_pos_embed(dst, pos),
            value=dst, attn_mask=dst_mask,
            key_padding_mask=dst_key_padding_mask)[0]
        src = src + getattr(self, f'dropout_{tgt}')(src2)

        return src

    def forward_post(self, src_inst, src_verb,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        src_inst = self.inst_proj(src_inst)
        q_s = k_s = self.with_pos_embed(src_inst, pos)
        src_inst2 = self.self_attn(q_s, k_s, value=src_inst, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src_inst = self.norm1(src_inst + self.dropout1(src_inst2))

        # src_obj = self.obj_proj(src_obj)
        # q_o = k_o = self.with_pos_embed(src_obj, pos)
        # src_obj2 = self.self_attn(q_o, k_o, value=src_obj, attn_mask=src_mask,
        #                           key_padding_mask=src_key_padding_mask)[0]
        # src_obj = self.norm1(src_obj + self.dropout1(src_obj2))

        src_verb = self.verb_proj(src_verb)
        q_v = k_v = self.with_pos_embed(src_verb, pos)
        src_verb2 = self.self_attn(q_v, k_v, value=src_verb, attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)[0]
        src_verb = self.norm1(src_verb + self.dropout1(src_verb2))

        src_inst, src_verb = self.forward_cross_attn(src_inst, src_verb, pos,
                                                             src_key_padding_mask)
        src_inst2 = self.linear2(self.dropout(self.activation(self.linear1(src_inst), inplace=True)))
        src_inst = self.norm2(src_inst + self.dropout2(src_inst2))

        # src_obj2 = self.linear2(self.dropout(self.activation(self.linear1(src_obj), inplace=True)))
        # src_obj = self.norm2(src_obj + self.dropout2(src_obj2))

        src_verb2 = self.linear2(self.dropout(self.activation(self.linear1(src_verb), inplace=True)))
        src_verb = self.norm2(src_verb + self.dropout2(src_verb2))

        return src_sub, src_obj, src_verb

    def forward_pre(self, src_sub, src_obj, src_verb,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        src_sub = self.sub_proj(src_sub)
        src_sub2 = self.norm_sub(src_sub)
        q_s = k_s = self.with_pos_embed(src_sub2, pos)
        src_sub2 = self.self_attn(q_s, k_s, value=src_sub2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src_sub = src_sub + self.dropout_sub(src_sub2)

        src_obj = self.obj_proj(src_obj)
        src_obj2 = self.norm_obj(src_obj)
        q_o = k_o = self.with_pos_embed(src_obj2, pos)
        src_obj2 = self.self_attn(q_o, k_o, value=src_obj2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src_obj = src_obj + self.dropout_sub(src_obj2)

        src_verb = self.verb_proj(src_verb)
        src_verb2 = self.norm_verb(src_verb)
        q_v = k_v = self.with_pos_embed(src_verb2, pos)
        src_verb2 = self.self_attn(q_v, k_v, value=src_verb2, attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)[0]
        src_verb = src_verb + self.dropout_verb(src_verb2)

        src_sub, src_obj, src_verb = self.forward_cross_attn(src_sub, src_obj, src_verb, pos,
                                                             src_key_padding_mask)
        src_sub2 = self.norm2(src_sub)
        src_sub2 = self.linear2(self.dropout(self.activation(self.linear1(src_sub2), inplace=True)))
        src_sub = src_sub + self.dropout2(src_sub2)

        src_obj2 = self.norm2(src_obj)
        src_obj2 = self.linear2(self.dropout(self.activation(self.linear1(src_obj2), inplace=True)))
        src_obj = src_obj + self.dropout2(src_obj2)

        src_verb2 = self.norm2(src_verb)
        src_verb2 = self.linear2(self.dropout(self.activation(self.linear1(src_verb2), inplace=True)))
        src_verb = src_verb + self.dropout2(src_verb2)

        return src_sub, src_obj, src_verb

    def forward(self, src_inst, src_verb,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            raise NotImplementedError, "Use post mode"
            return self.forward_pre(src_sub, src_obj, src_verb, src_mask,
                                    src_key_padding_mask, pos)
        return self.forward_post(src_inst, src_verb, src_mask,
                                 src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_cdn(args):
    return CDN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers_hopd=args.dec_layers_hopd,
        num_dec_layers_interaction=args.dec_layers_interaction,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
