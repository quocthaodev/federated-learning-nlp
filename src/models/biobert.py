import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

class BioBERT(nn.Module):
    def __init__(self, num_classes=4, pretrained_model_name='dmis-lab/biobert-v1.1', dropout=0.1, **kwargs):
        super().__init__()
        # Tải cấu hình và model từ HuggingFace
        self.config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=num_classes)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, config=self.config)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Trả về kết quả dự đoán (logits)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits