import argparse
import os
import logging
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin

from models import T5CVAE
from dataset import STCDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)





def main():
    args_dict = dict(
        data_dir="./data",  # path for data files
        output_dir="M1model",  # path to save the checkpoints
        model_name_or_path='mengzi-t5-base',
        tokenizer_name_or_path='mengzi-t5-base',
        max_seq_length=32,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=5000,
        train_batch_size=64,
        eval_batch_size=64,
        num_train_epochs=10,
        n_gpu=1,
        gradient_accumulation_steps=8,
        early_stop_callback=True,
        fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
        opt_level='apex',
        # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        # seed=42,
    )

    if not os.path.exists('model'):
        os.makedirs('model')

    args_dict.update({'output_dir': 'model'})
    args = argparse.Namespace(**args_dict)
    print(args_dict)


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir, filename='{epoch}-{val_loss:.2f}', monitor="val_loss", mode="min", save_top_k=2
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")


    train_params = dict(
        # accumulate_grad_batches=args.gradient_accumulation_steps,
        accumulate_grad_batches=1,
        max_epochs=10,
        precision=16 if args.fp_16 else 32,
        amp_backend=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        enable_checkpointing=checkpoint_callback,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=100,
        gpus=-1,
        strategy=DDPPlugin(find_unused_parameters=False),
        # val_check_interval=0.5,
    )


    # train_data = open('train.pkl', 'rb')
    # val_data = open('valid.pkl', 'rb')

    train_data = open('valid.pkl', 'rb')
    val_data = open('test.pkl', 'rb')

    print ("Initialize model")
    model = T5CVAE(args, train_data, val_data)

    trainer = pl.Trainer(**train_params)

    print (" Training model")
    trainer.fit(model)

    print ("training finished")

    print ("Saving model")
    model.model.save_pretrained('model')

    print ("Saved model")

if __name__ == "__main__":
    set_seed(42)
    main()
