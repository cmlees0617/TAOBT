# FILE: embedding_utilities.py
# AUTHOR: Caleb Lees
# LAST MODIFIED: 10 February 2026

import torch
import numpy as np
from transformers import BertTokenizerFast, RobertaForMaskedLM


def get_bert_verse_vector(text, model_path):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = RobertaForMaskedLM.from_pretrained(model_path)
    model.eval() # Set to evaluation mode (disables dropout, etc.)

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.roberta(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        mean_pooled = torch.mean(last_hidden_state, dim=1)
        return mean_pooled.squeeze().numpy()