import torch
from datasets import load_dataset

class PubMedQA(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        # Tải dataset PubMedQA (bản labeled)
        dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")

        # Tự chia train/test (80% train, 20% test) vì dataset này không chia sẵn
        split_idx = int(0.8 * len(dataset))
        if split == 'train':
            self.data = dataset.select(range(split_idx))
        else:
            self.data = dataset.select(range(split_idx, len(dataset)))

        self.max_len = max_len
        # Map nhãn: yes->0, no->1, maybe->2
        self.label_map = {'yes': 0, 'no': 1, 'maybe': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # Gom ngữ cảnh (abstract) và câu hỏi
        context = "".join(item['context']['contexts'])
        text = f"Context: {context} Question: {item['question']}"

        label_str = item['final_decision']
        label_id = self.label_map.get(label_str, 2)

        inputs = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return inputs['input_ids'].squeeze(), torch.tensor(label_id, dtype=torch.long)

def fetch_pubmedqa(args, root):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    return PubMedQA('train', tokenizer), PubMedQA('test', tokenizer)