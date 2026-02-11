# FILE: train_ot_model.py
# AUTHOR: Caleb M. Lees
# LAST MODIFIED: 9 February 2026

from pathlib import Path
import json
from tokenizers import BertWordPieceTokenizer
from transformers import (
    RobertaConfig, 
    RobertaForMaskedLM, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset


def ot_to_txt(input_dir:str, output_dir:str):
    """
    Convert the old testament from a directory of JSONs to a .txt file, where each verse
    lives on its own line. This is ideal for training.
    """
    src = Path(input_dir)
    out = Path(output_dir)

    all_hebrew_verses = []
    for json_file in src.glob("*.json"):
        with open(json_file, 'r') as f:
            scroll_dict = json.load(f)
            for chapter in scroll_dict.keys():
                all_hebrew_verses.extend([scroll_dict[chapter][verse] for verse in scroll_dict[chapter].keys()])

    with open(out / "tanakh_strongs.txt", "w") as f:
        for verse in all_hebrew_verses:
            f.write(verse + "\n")

    print(f"Wrote {len(all_hebrew_verses)} verses to training file '{out / "tanakh_strongs.txt"}'")


def train_ot_transformer(train_file_path:str, output_dir:str):
    """
    Trains an Old Testament sentence embedder using a Small RoBERTa architecture with
    - 6 hidden layers
    - 8 attention heads
    - embedding dimension of 512
    The objective function uses Masked Language Modeling with probability 0.15.
    """

    # 1. Train the tokenizer: Strong's ID -> token
    tokenizer_obj = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False
    )
    tokenizer_obj.train(
        files=train_file_path,
        vocab_size=10000,
        min_frequency=2
    )
    tokenizer_obj.save_model(output_dir)
    
    # Reload as a Transformers-compatible tokenizer
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(output_dir, lowercase=False)

    # Configure the transformer
    config = RobertaConfig(
        vocab_size=10000,
        max_position_embeddings=512,
        num_attention_heads=8,
        num_hidden_layers=6,
        type_vocab_size=1
    )
    model = RobertaForMaskedLM(config=config)

    # Load tokenized data
    raw_dataset = load_dataset("text", data_files={"train": train_file_path})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )

    tokenized_dataset = raw_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )

    # Setup Data Collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Train the transformer
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/ot_bert",
        num_train_epochs=50,
        per_device_train_batch_size=32,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=500
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"]
    )
    
    trainer.train()
    trainer.save_model(f"{output_dir}/ot_bert_final")




# ot_to_txt('./texts/lemmatized_manuscripts/ot', './texts/training/ot')
train_ot_transformer('./texts/training/ot/tanakh_strongs.txt', './models')




