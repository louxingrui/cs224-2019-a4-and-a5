from torch.utils.data import Dataset
from transformers import T5Tokenizer
from tqdm import tqdm

class STCDataset(Dataset):
    def __init__(self, tokenizer, data, max_len=32):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        # self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))
        self.data = data
        self._build()

    def __getitem__(self, idx):
        source_ids = self.inputs[idx]["input_ids"].squeeze()
        target_ids = self.targets[idx]["input_ids"].squeeze()
        src_mask = self.inputs[idx]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[idx]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.data)

    def _build(self):
        for idx in tqdm(range(len(self.data))):
            emotin_tag, input, target = self.data[idx].strip().split('\t')

            # input_ = "<emotion>" + input_
            input_ = "%s %s" % (emotin_tag, input)
            target = target

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len, padding='max_length', return_tensors="pt", truncation=True)
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, padding='max_length', return_tensors="pt", truncation=True)

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)





if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained("./mengzi-t5-base")

    # add new tokens
    characters = ["<Other>", "<Like>", "<Sadness>", "<Disgust>", "<Anger>", "<Happiness>"]
    tokenizer.add_tokens(characters)
    assert tokenizer.encode("<Other>") == [32128, 1]

    data = open('./data/train.txt', 'r', encoding='utf-8').readlines()
    dataset = STCDataset(tokenizer, data)
    import pickle
    file = open('./train.pkl', 'wb')
    pickle.dump(dataset, file)
    print("Val dataset: ", len(dataset))

    data = dataset[10]
    print(tokenizer.decode(data['source_ids']))
    print(tokenizer.decode(data['target_ids']))


