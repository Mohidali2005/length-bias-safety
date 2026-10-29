def build_ids(tokenizer,message,prefill,bos):
    """Turn one user message into prompt token ids with the chat template of the model"""
    text = tokenizer.apply_chat_template([{"role":"user","content":message}],tokenize=False,add_generation_prompt=True)
    if prefill:
        text += prefill
    ids = tokenizer(text,add_special_tokens=False).input_ids
    starts_with_bos = ids[0]==tokenizer.bos_token_id
    assert bos or not starts_with_bos,"the template adds a bos token but this model was trained without one"
    # some templates already write the bos token and adding it again would double it
    if bos and not starts_with_bos:
        ids = [tokenizer.bos_token_id]+ids
    return ids
