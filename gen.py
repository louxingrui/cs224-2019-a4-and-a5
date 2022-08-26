import time
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from transformers.generation_utils import top_k_top_p_filtering
from dataset import STCDataset
from torch.utils.data import DataLoader
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def sample_sequence(model, tokenizer, length, x_mask=None, x_tokens=None, y_mask=None, y_tokens=None,
                    temperature=0.9, top_k=40, top_p=0.90, device='cuda', sample=True, eos_token=None, model_type='cvae_model'):
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    batch_size, seq_len = x_tokens.shape
    prev = torch.tensor([[0]] * batch_size, dtype=torch.long, device=device)

    output = prev
    probability = torch.tensor([], dtype=torch.float, device=device)
    if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)
    model.eval()
    with torch.no_grad():
        # get latent Z
        prior_c_mean, prior_c_logvar, prior_e_mean, prior_e_logvar, encoder_outputs = model.encoder(
            input_ids=x_tokens,
            attention_mask=x_mask,)
        latent_c_mean, latent_c_logvar = prior_c_mean, prior_c_logvar
        latent_e_mean, latent_e_logvar = prior_e_mean, prior_e_logvar

        cz = model.reparameterize(latent_c_mean, latent_c_logvar)
        ez = model.reparameterize(latent_e_mean, latent_e_logvar)

        assert not torch.isnan(cz).any(), 'training get nan cz'
        assert not torch.isnan(ez).any(), 'training get nan ez'

        # generation
        for i in range(length):
            decoder_outputs = model.decoder(
                    input_ids=prev,
                    past_key_values=encoder_outputs.past_key_values,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    encoder_attention_mask=x_mask,
                    content_representations=cz,
                    emotion_representations=ez,
            )
            sequence_output = decoder_outputs[0]
            logits = model.lm_head(sequence_output)
            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k, top_p)
            probs = torch.softmax(logits, dim=-1)
            if sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            probability = torch.cat((probability, probs.gather(1, next_token)), dim=1)
            output = torch.cat((output, next_token), dim=1)
            prev = output

            # early stopping if all sents have ended once
            if_end[next_token.view(-1).eq(eos_token)] = True
            if if_end.all(): break

    return output, probability

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

def eavl_test(dataset, file_path1, tokenizer, model):
    from tqdm import tqdm
    # from utils import evaluate_ppl
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size)
    f_beam = open(file_path1, 'w', encoding='utf-8')


    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids = batch["source_ids"].to(device)
        attention_masks = batch["source_mask"].to(device)

        outputs, _ = sample_sequence(model, tokenizer, length=32, x_mask=attention_masks, x_tokens=input_ids,
                                    eos_token=1, model_type='cvae_model')
        outputs = tokenizer.batch_decode(outputs)
        # print(''.join(output))
        for output in outputs:
            f_beam.write(''.join(output)+'\n')
    f_beam.close()

#
def main(seed, i):

    from CVAE_model import CVAE
    from utils.utils import init_para_frompretrained
    set_seed(seed)
    print("Loading Model")
    model = CVAE.from_pretrained('dul_model')
    tokenizer = T5Tokenizer.from_pretrained('mengzi-t5-base')
    characters = ["<Other>", "<Like>", "<Sadness>", "<Disgust>", "<Anger>", "<Happiness>"]
    tokenizer.add_tokens(characters)
    assert tokenizer.encode("<Other>") == [32128, 1]
    init_para_frompretrained(model.rep_encoder, model.encoder, share_para=True)
    model = model.to(device)
    f = open("data/test.pkl", "rb")
    dataset = pickle.load(f)
    print("Begining inference")
    eavl_test(dataset, "results/results{}.txt".format(i), tokenizer, model)
    return 0

if __name__ == "__main__":

    seed = 42
    # main(seed, i)
    from CVAE_model import CVAE
    from utils.utils import init_para_frompretrained
    set_seed(42)

    tokenizer = T5Tokenizer.from_pretrained('mengzi-t5-base')
    characters = ["<Other>", "<Like>", "<Sadness>", "<Disgust>", "<Anger>", "<Happiness>"]
    tokenizer.add_tokens(characters)
    assert tokenizer.encode("<Other>") == [32128, 1]

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
    passage = '我真是太喜欢吃火锅了！'

    emotion = '<Other>'
    inference(passage, emotion, tokenizer, model)
