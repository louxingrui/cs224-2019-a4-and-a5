import random
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import copy
import re
import math
from transformers import (
    AdamW,
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5Config,
    get_linear_schedule_with_warmup
)

from CVAE_model import CVAE

from utils.utils import init_para_frompretrained

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


class T5CVAE(pl.LightningModule):
    def __init__(self, hparams, train_data, val_data):
        super(T5CVAE, self).__init__()

        self.hparams.update(vars(hparams))
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        # add new tokens
        characters = ["<Other>", "<Like>", "<Sadness>", "<Disgust>", "<Anger>", "<Happiness>"]
        self.tokenizer.add_tokens(characters)
        assert self.tokenizer.encode("<Other>") == [32128, 1], "the embedding has changed!"

        self.endoftext = 1

        self.model = CVAE.from_pretrained('mengzi-t5-base')
        # resize Embedding #(32128, 768) > (32134, 768)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.rep_encoder.resize_token_embeddings(len(self.tokenizer))

        init_para_frompretrained(self.model.rep_encoder, self.model.encoder, share_para=True)

        self.train_dataset = pickle.load(train_data)
        self.val_dataset = pickle.load(val_data)


    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(
            self, input_ids, target_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids=input_ids,
            target_ids=target_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        # should use deepcopy
        labels = copy.deepcopy(batch["target_ids"])
        # when cal loss, ignore <pad>
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        # print(batch["target_ids"])
        # print(batch["target_mask"])
        # print(labels)
        outputs = self(
            input_ids=batch["source_ids"],
            target_ids=batch["target_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask'],
        )

        total_loss, kl_loss, logits = outputs

        return total_loss, kl_loss, logits

    def training_step(self, batch, batch_idx):
        total_loss, kl_loss, _ = self._step(batch)
        logging_dict = {"loss": total_loss}
        self.log_dict(logging_dict, on_step=False, on_epoch=True)
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        # target_ids = batch["target_ids"]
        total_loss, nll_loss, _ = self._step(batch)
        # tokens = target_ids.tolist()
        # tokens = [t[:t.index(self.endoftext) + 1] if self.endoftext in t else t for t in tokens]
        # ntokens = sum([len(t) for t in tokens])
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None
                       ):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


class M2Model(pl.LightningModule):
    def __init__(self, hparams, train_data, val_data):
        super(M2Model, self).__init__()

        self.hparams.update(vars(hparams))
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        # add new tokens
        characters = ["<Other>", "<Like>", "<Sadness>", "<Disgust>", "<Anger>", "<Happiness>"]
        self.tokenizer.add_tokens(characters)
        assert self.tokenizer.encode("<Other>") == [32128, 1], "the embedding has changed!"

        self.endoftext = 1

        self.model = CVAE.from_pretrained('mengzi-t5-base')
        # resize Embedding #(32128, 768) > (32134, 768)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.rep_encoder.resize_token_embeddings(len(self.tokenizer))

        init_para_frompretrained(self.model.rep_encoder, self.model.encoder, share_para=True)

        self.train_dataset = pickle.load(train_data)
        self.val_dataset = pickle.load(val_data)


    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(
            self, input_ids, target_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids=input_ids,
            target_ids=target_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        # should use deepcopy
        labels = copy.deepcopy(batch["target_ids"])
        # when cal loss, ignore <pad>
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        # print(batch["target_ids"])
        # print(batch["target_mask"])
        # print(labels)
        outputs = self(
            input_ids=batch["source_ids"],
            target_ids=batch["target_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask'],
        )

        total_loss, kl_loss, logits = outputs

        return total_loss, kl_loss, logits

    def training_step(self, batch, batch_idx):
        total_loss, kl_loss, _ = self._step(batch)
        logging_dict = {"loss": total_loss}
        self.log_dict(logging_dict, on_step=False, on_epoch=True)
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        # target_ids = batch["target_ids"]
        total_loss, nll_loss, _ = self._step(batch)
        # tokens = target_ids.tolist()
        # tokens = [t[:t.index(self.endoftext) + 1] if self.endoftext in t else t for t in tokens]
        # ntokens = sum([len(t) for t in tokens])
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None
                       ):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)

