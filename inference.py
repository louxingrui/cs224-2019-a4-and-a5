import time
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from dataset import STCDataset
from torch.utils.data import DataLoader
import pickle
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def top_k_top_p_filtering(logits, top_k=40, top_p=0.90, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits

def sample_sequence(model, tokenizer, length, x_mask=None, x_tokens=None, y_mask=None, y_tokens=None,
                    temperature=0.9, top_k=40, top_p=0.90, device='cuda', sample=True, eos_token=None, model_type='cvae_model'):
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    batch_size, seq_len = x_tokens.shape
    # decoder_start_token_id = self.config.decoder_start_token_id
    prev = torch.tensor([[0]] * batch_size, dtype=torch.long, device=device)

    output = prev
    probability = torch.tensor([], dtype=torch.float, device=device)
    if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)
    pooled_feats = None
    model.eval()
    with torch.no_grad():
        # get latent Z
        c_mean, c_logvar, e_mean, e_logvar, encoder_outputs = model.encoder(
            input_ids=x_tokens,
            attention_mask=x_mask,)
        latent_c_mean, latent_c_logvar = c_mean, c_logvar
        latent_e_mean, latent_e_logvar = e_mean, e_logvar
        cz = model.reparameterize(latent_c_mean, latent_c_logvar)
        ez = model.reparameterize(latent_e_mean, latent_e_logvar)
        assert not torch.isnan(cz).any(), 'training get nan cz'
        assert not torch.isnan(ez).any(), 'training get nan ez'
        # latent_mean, latent_logvar = prior_mean, prior_logvar
        # z = model.reparameterize(latent_mean, latent_logvar)
        # assert not torch.isnan(z).any(), 'training get nan z'

        pooled_feats_e = latent_e_mean.cpu().numpy()
        pooled_feats_c = latent_c_mean.cpu().numpy()
        # print("ez:", ez)


    # generation
    #     for i in range(length):
    #
    #         decoder_outputs = model.decoder(
    #                 input_ids=prev,
    #                 past_key_values=encoder_outputs.past_key_values,
    #                 encoder_hidden_states=encoder_outputs.last_hidden_state,
    #                 encoder_attention_mask=x_mask,
    #                 content_representations=None,
    #                 emotion_representations=ez,
    #         )
    #         sequence_output = decoder_outputs[0]
    #         logits = model.lm_head(sequence_output)
    #         logits = logits[:, -1, :] / temperature
    #         logits = top_k_top_p_filtering(logits, top_k, top_p)
    #         probs = torch.softmax(logits, dim=-1)
    #         if sample:
    #             next_token = torch.multinomial(probs, num_samples=1)
    #         else:
    #             _, next_token = torch.topk(probs, k=1, dim=-1)
    #
    #         probability = torch.cat((probability, probs.gather(1, next_token)), dim=1)
    #         output = torch.cat((output, next_token), dim=1)
    #         prev = output
    #
    #         # early stopping if all sents have ended once
    #         if_end[next_token.view(-1).eq(eos_token)] = True
    #         if if_end.all(): break

    # return output, probability
    return pooled_feats_c, pooled_feats_e

def inference(passage, emotion_tag, tokenizer, model):
    # "<Other>", "<Like>", "<Sadness>", "<Disgust>", "<Anger>", "<Happiness>"
    text = "%s %s" % (emotion_tag, passage)

    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    print("Context:{}, Emotion:{} ".format(passage, emotion_tag))

    output, _ = sample_sequence(model, tokenizer, length=32,
                             x_mask=attention_masks, x_tokens=input_ids, eos_token=1
                             )
    output = tokenizer.batch_decode(output)
    print ("\nBeam decoding [Most accurate questions] ::\n")
    for out in output:
        print(out)

    print ("\n")
    return 0

def eavl_test(dataset, file_path1, file_path2, tokenizer, model):
    from tqdm import tqdm
    # from utils import evaluate_ppl
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # f_beam = open(file_path1, 'w', encoding='utf-8')

    pooled_feats_c = None
    pooled_feats_e = None
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids = batch["source_ids"].to(device)
        attention_masks = batch["source_mask"].to(device)

        outputs_c, outputs_e = sample_sequence(model, tokenizer, length=32, x_mask=attention_masks, x_tokens=input_ids, y_mask=None, y_tokens=None,
                    eos_token=1, model_type='cvae_model')
        if pooled_feats_c is None:
            pooled_feats_c = outputs_c
            pooled_feats_e = outputs_e
        else:
            pooled_feats_c = np.concatenate((pooled_feats_c, outputs_c), axis=0)
            pooled_feats_e = np.concatenate((pooled_feats_e, outputs_e), axis=0)
        # print("feats:", pooled_feats.shape)
        np.save(file_path1, pooled_feats_c)
        np.save(file_path2, pooled_feats_e)
        # outputs = tokenizer.batch_decode(outputs)
    #     for output in outputs:
    #         f_beam.write(''.join(output)+'\n')
    # f_beam.close()

#
def main():

    from CVAE_model import CVAE
    from utils.utils import init_para_frompretrained
    set_seed(42)

    tokenizer = T5Tokenizer.from_pretrained('mengzi-t5-base')
    # add new tokens
    characters = ["<Other>", "<Like>", "<Sadness>", "<Disgust>", "<Anger>", "<Happiness>"]
    tokenizer.add_tokens(characters)
    assert tokenizer.encode("<Other>") == [32128, 1], "the embedding has changed!"

    model = CVAE.from_pretrained('mengzi-t5-base')
    # resize Embedding #(32128, 768) > (32134, 768)
    model.resize_token_embeddings(len(tokenizer))
    model.rep_encoder.resize_token_embeddings(len(tokenizer))
    model_state_dict = torch.load(
        '/common-data/new_build/xingrui.lou/Dul_attention_cvae/Dul_LVG_model/pytorch_model.bin')
    model.load_state_dict(model_state_dict, strict=False)
    init_para_frompretrained(model.rep_encoder, model.encoder, share_para=True)
    del model_state_dict


    model = model.to(device)
    f = open("./data/test.pkl", "rb")
    dataset = pickle.load(f)
    eavl_test(dataset, "Dual-LVG-c.npy", "Dual-LVG-e.npy", tokenizer, model)
    return 0

if __name__ == "__main__":
    main()
    # from CVAE_model import CVAE
    # from utils.utils import init_para_frompretrained
    # set_seed(42)
    # model = CVAE.from_pretrained('cvae_model')
    # tokenizer = T5Tokenizer.from_pretrained('mengzi-t5-base')
    # characters = ["<Other>", "<Like>", "<Sadness>", "<Disgust>", "<Anger>", "<Happiness>"]
    # tokenizer.add_tokens(characters)
    # assert tokenizer.encode("<Other>") == [32128, 1]
    # init_para_frompretrained(model.rep_encoder, model.encoder, share_para=True)
    #
    # model = model.to(device)


    # gen
    # passage = '我真是太喜欢吃火锅了！'
    # emotion = '<Happiness>'
    # inference(passage, emotion, tokenizer, model)
