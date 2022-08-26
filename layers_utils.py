from transformers.models.t5.modeling_t5 import *
import torch
import torch.nn as nn

# class Attention(nn.Module):
#     def __init__(self):

class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        # TODO: replace relu with gelu
        # self.non_linearity = nn.relu

    def forward(self, inputs, attention_mask=None):
        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################
        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = torch.relu(inputs.matmul(self.attention_weights))
        ##################################################################
        # Step 2 - Masking
        ##################################################################
        if attention_mask is not None:
            scores = scores + attention_mask
        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################
        scores = self.softmax(scores)
        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        # sum the hidden states
        representations = weighted.sum(1).squeeze(1)

        return representations, scores

class Pseudi_Attention(T5Attention):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__(config)
        # TODO: z or z_k and z_v?
        self.z = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # self.z_v = nn.Linear(self.d_model, self.inner_dim, bias=False)

    def forward(
            self,
            hidden_states,
            z,
            mask=None,
            key_value_states=None,
            position_bias=None,
            past_key_value=None,
            layer_head_mask=None,
            query_length=None,
            use_cache=False,
            output_attentions=False,
    ):
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                    len(past_key_value) == 2
            ), "past_key_value should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_value)
            )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """  projection """
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """  reshape """
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """ projects hidden states correctly to key/query states """
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )   #(64, 12, 32, 64)
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        key_z = shape(self.z(z))  # (batch_size, n_heads, seq_length, dim_per_head)
        value_z = shape(self.z(z))  #(64, 12, 1, 64)

        # concat
        key_states = torch.cat((key_z, key_states), dim=-2) #(64, 12, 33, 64)
        value_states = torch.cat((value_z, value_states), dim=-2) #(64, 12, 33, 64)
        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        #(64, 12, 32, 33)
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length+1), device=scores.device, dtype=scores.dtype
                )   #(1, 12, 32, 32)
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -seq_length:, :]
            # mask (64, 1, 1, 32)
            if mask is not None:
                mask_add = torch.zeros(
                    (batch_size, 1, mask.shape[2], 1), device=mask.device, dtype=mask.dtype
                )
                mask_tuple = (mask, mask_add)
                mask = torch.cat(mask_tuple, dim=-1)
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
            # (64, 12, 32, 32)
        scores += position_bias
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)  (64, 12, 32, 33)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

# class T5LayerAttention(T5LayerSelfAttention):
#     def __init__(self, config, has_relative_attention_bias=False):
#         super().__init__(config)
#         self.SelfAttention = Pseudi_Attention(config, has_relative_attention_bias=False)
#
#     def forward(
#         self,
#         hidden_states,
#         z,
#         attention_mask=None,
#         position_bias=None,
#         layer_head_mask=None,
#         past_key_value=None,
#         use_cache=False,
#         output_attentions=False,
#     ):
#         normed_hidden_states = self.layer_norm(hidden_states)
#         normed_z = self.layer_norm(z)
#         attention_output = self.SelfAttention(
#             normed_hidden_states,
#             normed_z,
#             mask=attention_mask,
#             position_bias=position_bias,
#             layer_head_mask=layer_head_mask,
#             past_key_value=past_key_value,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#         )
#         hidden_states = hidden_states + self.dropout(attention_output[0])
#         outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
#         return outputs

class Pseudi_T5LayerAttention(T5LayerCrossAttention):
    def __init__(self, config):
        super().__init__(config)
        self.EncDecAttention = Pseudi_Attention(config, has_relative_attention_bias=False)

    def forward(
        self,
        hidden_states,
        z,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        normed_z = self.layer_norm(z)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            normed_z,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs

class Decoder_Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config)
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(Pseudi_T5LayerAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        emotion_representations,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        encoder_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
                expected_num_past_key_values,
                "2 (past / key) for cross attention" if expected_num_past_key_values == 4 else "",
                len(past_key_value),
            )
            assert len(past_key_value) == expected_num_past_key_values, error_message

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                emotion_representations,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)

