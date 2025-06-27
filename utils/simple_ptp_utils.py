import torch, einops


class SimpleAttentionProcessor():
    """
    code from https://github.com/yuval-alaluf/Attend-and-Excite/blob/main/utils/ptp_utils.py
    """

    def __init__(self, attention_store, place_in_unet):
        self.attention_store = attention_store
        self.place_in_unet = place_in_unet

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        attention_probs_cpu = attention_probs.half().cpu()
        
        if is_cross:
            attention_probs_cpu = einops.rearrange(
                attention_probs_cpu, 
                "(b hd) (h1 w1) l -> b hd h1 w1 l", 
                b=batch_size,
                h1=int(attention_probs_cpu.size(-2) ** 0.5),
            )
            self.attention_store(attention_probs_cpu, is_cross, self.place_in_unet)
        else:
            attention_probs_cpu = einops.rearrange(
                attention_probs_cpu, 
                "(b hd) (h1 w1) (h2 w2) -> b hd h1 w1 h2 w2", 
                b=batch_size,
                h1=int(attention_probs_cpu.size(-2) ** 0.5),
                h2=int(attention_probs_cpu.size(-1) ** 0.5)
            )
            self.attention_store(attention_probs_cpu, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class SimpleAttentionStore():
    """
    code from https://github.com/yuval-alaluf/Attend-and-Excite/blob/main/utils/ptp_utils.py
    """
    
    def __init__(self):
        self.store = SimpleAttentionStore.get_empty_store()
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attention_score, 
                       is_cross: bool,
                       place_in_unet: str):

        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.store[key].append(attention_score)
        return attention_score
    
    def get_self_attention_maps(self):
        self_attention_maps = []
        for key in self.store.keys():
            if "self" in key:
                self_attention_maps.extend(self.store[key])
        return self_attention_maps
    
    def get_cross_attention_maps(self):
        cross_attention_maps = []
        for key in self.store.keys():
            if "cross" in key:
                cross_attention_maps.extend(self.store[key])
        return cross_attention_maps


def set_attention_processors(pipe, attention_store):
    attn_processors = pipe.unet.attn_processors

    for key in attn_processors.keys():
        if "up_blocks" in key:
            attn_processors[key] = SimpleAttentionProcessor(attention_store, "up")

    pipe.unet.set_attn_processor(attn_processors)



    
        
