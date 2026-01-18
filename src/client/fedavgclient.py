import copy
import torch
from torch.utils.data import DataLoader

class FedavgClient:
    def __init__(self, args, train_dataset, model):
        self.args = args
        # Tạo bộ nạp dữ liệu (Batch size = 4)
        self.train_loader = DataLoader(train_dataset, batch_size=args.B, shuffle=True)
        # Copy model về để không ảnh hưởng model gốc
        self.model = copy.deepcopy(model)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def update(self):
        # Hàm train cục bộ (Local Update)
        self.model.train()
        # Dùng AdamW optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        
        for epoch in range(self.args.E):
            for batch in self.train_loader:
                input_ids, labels = batch
                input_ids = input_ids.to(self.args.device)
                labels = labels.to(self.args.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids) # Forward
                loss = self.criterion(outputs, labels)    # Tính lỗi
                loss.backward()                           # Tính gradient
                optimizer.step()                          # Cập nhật trọng số
                
        # Trả về bộ trọng số mới sau khi học xong
        return self.model.state_dict()