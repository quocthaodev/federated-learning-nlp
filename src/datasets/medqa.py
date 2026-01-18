import torch
import json
from datasets import load_dataset

class MedQA(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        
        # Tải dataset GBaker chuẩn
        print(f"[MedQA] Downloading dataset for {split} from GBaker/MedQA-USMLE-4-options...")
        try:
            dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="train" if split == 'train' else "test")
        except Exception as e:
            print("[MedQA] Thử server dự phòng...")
            dataset = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="train" if split == 'train' else "test")
            
        self.data = dataset
        self.max_len = max_len
        self.label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # --- SỬA LỖI ÉP KIỂU SỐ NGUYÊN ---
        if hasattr(index, 'item'):
            index = index.item()
        else:
            index = int(index)
        # ---------------------------------

        item = self.data[index]
        
        # Xử lý options
        opts = item['options'] 
        if isinstance(opts, str):
            try:
                opts = eval(opts)
            except: pass
        
        opt_str = ""
        for key in ['A', 'B', 'C', 'D']:
            val = opts.get(key, "")
            opt_str += f"{key}: {val} "
        
        text = f"Question: {item['question']} Options: {opt_str}"
        label_char = item['answer_idx']
        label_id = self.label_map.get(label_char, 0)

        inputs = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return inputs['input_ids'].squeeze(), torch.tensor(label_id, dtype=torch.long)

def fetch_medqa(args, root):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    return MedQA('train', tokenizer), MedQA('test', tokenizer)
