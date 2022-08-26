import argparse
import os
import logging
import random
import copy
import math
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
# from torch.nn.parallel import
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
from transformers import (
    AdamW,
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5Config,
    get_linear_schedule_with_warmup
)

from dataset import STCDataset
from CVAE_model import CVAE
from utils.utils import init_para_frompretrained, get_datasets

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

def random_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--start_rank', type=int, default=0)
    parser.add_argument('--adress', type=str, default='tcp://127.0.0.1:80')

    parser.add_argument('--load_model_dir', type=str, default='/common-data/new_build/xingrui.lou/lxr/mengzi-t5-base')
    parser.add_argument('--data_base', type=str, default='/common-data/new_build/xingrui.lou/lxr/data')
    parser.add_argument('--output_dir', type=str, default='/common-data/new_build/xingrui.lou/lxr/M2model')
    parser.add_argument('--model_type', type=str, default='CVAE', choices=['Seq2Seq', 'CVAE'])



    parser.add_argument("--numStyleEmb",
                        default=30,
                        type=int,
                        help="not used here")

    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.5,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--cross_folds",
                        default=10,
                        type=int,
                        help="not used here")

    parser.add_argument("--kl_weight",
                        default=0,
                        type=float,
                        help="the weight of KLD loss term")
    parser.add_argument("--update_kl",
                        default=False,
                        action='store_true',
                        help="whether to optimize the KLD loss")
    parser.add_argument("--kl_interval",
                        default=15,
                        type=int,
                        help="optimize KLD every K steps")
    parser.add_argument("--patience",
                        default=3,
                        type=int,
                        help="used for early_stopping")
    parser.add_argument("--acc_thresh",
                        default=0.30,
                        type=float,
                        help="start to optimize KLD when achive the accucary thersh.")
    parser.add_argument("--beam_size",
                        default=5,
                        type=int,
                        help="")
    parser.add_argument("--sample_num",
                        default=5,
                        type=int,
                        help="")
    parser.add_argument("--fixed_layers",
                        default=9,
                        type=int,
                        help="how many transformer layers in BERT are fixed")
    parser.add_argument("--num_hidden_layers",
                        default=-1,
                        type=int,
                        help="how many transformer layers in our model， default settings is load when it's -1")
    parser.add_argument("--noise",
                        default=0.,
                        type=float,
                        help="how many noise added to the latent z variable")
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        '--opt_level',
        type=str,
        default='O0',
        choices=[
            'O0',
            'O1',
            'O2'])

    args = parser.parse_args()

    return args

def main(gpu, nprocs, args):


    def warmup_linear(x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def train(data_loader, epoch, global_step):
        logger.info("\ntraining epoch %d, Num examples = %d, Batch size = %d, Num steps = %d" \
                    % (epoch, len(data_loader), args.train_batch_size, num_train_steps))

        model.train()
        tr_loss = 0
        tr_accuracy = 0
        tr_kld = 0
        nb_tokens = 0

        nb_tr_steps = 0

        print_steps = 200 * args.gradient_accumulation_steps
        print_accuracy = 0
        print_tokens = 0

        print_kld = 0
        print_loss = 0
        print_loss_emo = 0

        best_ppl_kld = 1e9
        patience = 0
        for step, batch in enumerate(data_loader):
            # should use deepcopy
            labels = copy.deepcopy(batch["target_ids"])
            # when cal loss, ignore <pad>
            labels[labels[:, :] == tokenizer.pad_token_id] = -100

            # print(batch["target_ids"])
            # print(batch["target_mask"])
            # print(labels)
            outputs = model(
                input_ids=batch["source_ids"].to(args.device),
                target_ids=batch["target_ids"].to(args.device),
                attention_mask=batch["source_mask"].to(args.device),
                labels=labels.to(args.device),
                decoder_attention_mask=batch['target_mask'].to(args.device),
            )

            total_loss, kld, nll_loss, logits = outputs

            # kl_interval: 每多少步优化一次KLD
            if args.update_kl:
                if args.kl_weight < 1:
                    args.kl_weight += 1e-05
                loss = nll_loss + int(step % args.kl_interval == 0) * args.kl_weight * kld
            # loss = loss_nll + int(step%args.kl_interval==0)*args.kl_weight*kld

            else:
                loss = nll_loss + 1e-05 * int(step % 5 == 0) * kld
            # loss = loss_nll + 1e-05*int(step%5==0)*kld

            temp_accuracy, temp_tokens = accuracy(logits, batch)

            if args.world_size > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.\
                kld = kld.mean()
            if args.fp16 and args.loss_scale != 1.0:
                # rescale loss for fp16 training
                # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                loss = loss * args.loss_scale
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # print(nll_loss)
            # print(kld)
            # print(kld.item())
            tr_loss += nll_loss.item()
            tr_kld += kld.item()

            nb_tokens += temp_tokens
            nb_tr_steps += 1

            tr_accuracy += temp_accuracy

            print_tokens += temp_tokens
            print_accuracy += temp_accuracy
            print_loss += nll_loss.item()
            print_kld += kld.item()

            if nb_tr_steps % print_steps == 0:
                print_accuracy = print_accuracy / print_tokens

                print_loss /= print_steps
                print_loss_emo /= print_steps
                print_kld /= print_steps

                '''
                ######
                用于模型early-stopping
                （目前针对VAEs模型没有通用的标准的early stopping规则，通常KLD和NLL二者互相竞争平衡，而对文本生成模型来说，训练指标最好时在测试推理时效果不一定最好。。。难以评价）
                 我在这个实验中一般未使用early stopping
                ######

                if math.exp(print_loss) + print_kld < best_ppl_kld:
                    best_ppl_kld = math.exp(print_loss) + print_kld
                    patience = 0
                else:
                    patience += 1

                if patience == args.patience:
                    stop_training = True
                    break
                '''
                # 预训练NLL项时，当token预测准确率达到阈值时开始优化KLD项
                if print_accuracy > args.acc_thresh:
                    args.update_kl = True

                logger.info(
                    'steps:%d/total_steps:%d, loss:%.3f, ppl:%.2f,  kld:%4.2f , kl_weight:%.5f, acc:%3f' % (
                        global_step, num_train_steps, print_loss, math.exp(print_loss), print_kld,
                        args.kl_weight, print_accuracy))

                print_accuracy = 0
                print_tokens = 0

                print_loss = 0
                print_loss_emo = 0
                print_kld = 0

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        # break

        tr_accuracy /= nb_tokens
        tr_loss /= nb_tr_steps
        tr_kld /= nb_tr_steps

        return tr_loss, tr_accuracy, tr_kld, global_step

    def evaluate(data_loader):
        model.eval()
        eval_loss = 0
        eval_accuracy = 0
        nb_tokens = 0
        eval_kld = 0

        nb_tr_steps = 0

        total = math.ceil(len(data_loader) / args.train_batch_size)
        n = 0

        for sample_idx, batch in enumerate(data_loader):
            # should use deepcopy
            labels = copy.deepcopy(batch["target_ids"])
            # when cal loss, ignore <pad>
            labels[labels[:, :] == tokenizer.pad_token_id] = -100

            # print(batch["target_ids"])
            # print(batch["target_mask"])
            # print(labels)
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["source_ids"].to(args.device),
                    target_ids=batch["target_ids"].to(args.device),
                    attention_mask=batch["source_mask"].to(args.device),
                    labels=labels.to(args.device),
                    decoder_attention_mask=batch['target_mask'].to(args.device),
                )

                total_loss, kld, nll_loss, logits = outputs
                temp_accuracy, temp_tokens = accuracy(logits, batch)

            eval_loss += nll_loss.mean().item()
            nb_tokens += temp_tokens

            eval_kld += kld.item()
            nb_tr_steps += 1

            eval_accuracy += temp_accuracy

        # break

        eval_accuracy /= nb_tokens
        eval_loss /= nb_tr_steps
        eval_kld /= nb_tr_steps

        return eval_loss, eval_accuracy, eval_kld

    def accuracy(logits, batch):
        _, preds = logits.max(2)
        input_ids = batch["target_ids"].to(args.device)
        resp_mask = batch["target_mask"].to(args.device)
        temp_accuracy = ((preds[:, :-1] == input_ids[:, :-1]).float() * resp_mask[:, :-1]).sum().item()
        temp_examples = resp_mask[:, :].sum().item()

        return temp_accuracy, temp_examples

    random_seed(args.seed)

    print(f'GPU: {gpu}')

    args.gpu = gpu
    args.rank = args.start_rank + args.gpu
    print(f'Rank: {args.rank}')

    group = dist.group.WORLD
    args.device = gpu
    n_gpu = torch.cuda.device_count()
    logger.info("gpu {}, n_gpu {}".format(gpu, n_gpu))
    torch.cuda.set_device(gpu)
    dist.init_process_group("nccl",
                            rank=args.rank,
                            init_method=args.adress,
                            world_size=args.world_size)



    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = T5Tokenizer.from_pretrained(args.load_model_dir)
    # add new tokens
    characters = ["<Other>", "<Like>", "<Sadness>", "<Disgust>", "<Anger>", "<Happiness>"]
    tokenizer.add_tokens(characters)
    assert tokenizer.encode("<Other>") == [32128, 1], "the embedding has changed!"

    model = CVAE.from_pretrained(args.load_model_dir)
    # resize Embedding #(32128, 768) > (32134, 768)
    model.resize_token_embeddings(len(tokenizer))
    model.rep_encoder.resize_token_embeddings(len(tokenizer))
    logger.info('Begin init Rep_encoder')
    init_para_frompretrained(model.rep_encoder, model.encoder, share_para=True)
    logger.info('Init Rep_encoder Done')
    model.cuda(gpu)
    dataset_paths = {
        'train': os.path.join(args.data_base, 'train.pkl'),
        'valid': os.path.join(args.data_base, 'valid.pkl'),
        'test': os.path.join(args.data_base, 'test.pkl'),
    }
    logger.info('Begin load data')
    datasets = get_datasets(dataset_paths)
    logger.info('Load data Done')
    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            datasets['train'],
            num_replicas=args.world_size,
            rank=args.rank
        )
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            datasets['valid'],
            num_replicas=args.world_size,
            rank=args.rank
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            datasets['test'],
            num_replicas=args.world_size,
            rank=args.rank
        )
    else:
        train_sampler = RandomSampler(datasets['train'])
        valid_sampler = RandomSampler(datasets['valid'])
        test_sampler = RandomSampler(datasets['test'])
    train_seq_dataloader = DataLoader(datasets['train'], sampler=train_sampler, batch_size=args.train_batch_size)
    valid_seq_dataloader = DataLoader(datasets['valid'], sampler=valid_sampler, batch_size=args.train_batch_size)
    test_seq_dataloader = DataLoader(datasets['test'], sampler=test_sampler, batch_size=args.train_batch_size)

    num_train_steps = int(len(datasets["train"]) / args.train_batch_size / \
                          args.gradient_accumulation_steps * args.num_train_epochs)

    t_total = num_train_steps
    global_step = 0
    best_ppl_kld = 1e09
    stop_training = False

    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    # -------------------------------------------------------------------------------

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon)
    model, optimizer = amp.initialize(
        model, optimizer, opt_level=args.opt_level)
    model = DDP(model)
    # model.config = config
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=args.warmup_steps,
    #     num_training_steps=args.train_n_steps)

    stop_training = False

    for epoch in range(int(args.num_train_epochs)):
        if stop_training:
            break
        if args.do_train:
            tr_loss, tr_accuracy, tr_kld, global_step = train(train_seq_dataloader, epoch, global_step)
        else:
            tr_loss, tr_accuracy, tr_acc_emo_post, tr_acc_emo_prior, tr_kld = 0., 0., 0., 0., 999
        dev_loss, dev_accuracy, dev_kld = evaluate(valid_seq_dataloader)
        test_loss, test_accuracy, test_kld = evaluate(test_seq_dataloader)
        logger.info('\n'+'#'*40)
        logger.info('epoch%d, train_acc:%.3f, train_ppl:%.2f, train_kld:%.3f'
                     % (epoch, tr_accuracy, math.exp(tr_loss), tr_kld))
        logger.info('epoch%d, dev_acc:%.3f, dev_ppl:%.2f, dev_kld:%.3f'
                     % (epoch, dev_accuracy, math.exp(dev_loss), dev_kld))
        logger.info('epoch%d, test_acc:%.3f, test_ppl:%.2f, test_kld:%.3f'
                     % (epoch, test_accuracy, math.exp(test_loss), test_kld))
        logger.info('#'*40)

        if args.do_train:
            now = datetime.now()
            strnow = datetime.strftime(now, '%Y-%m-%d_%H_%M_%S_')
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, strnow + "pytorch_model.bin")
            logging.info('Saved model file:%s', output_model_file)
            if args.do_train:
                torch.save(model_to_save.state_dict(), output_model_file)

    now = datetime.now()
    strnow = datetime.strftime(now, '%Y-%m-%d_%H_%M_%S_')

if __name__ == "__main__":

    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        main, nprocs=ngpus_per_node, args=(ngpus_per_node, args)
    )



