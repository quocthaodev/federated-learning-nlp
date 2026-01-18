import copy
import torch
import os
import numpy as np
from src.client.fedavgclient import FedavgClient
from src.algorithm.fedavg import FedAvgOptimizer
from torch.utils.data import DataLoader

class FedavgServer:
    def __init__(self, args, writer, server_testset, client_datasets, model):
        self.args = args
        self.writer = writer
        self.test_loader = DataLoader(server_testset, batch_size=args.B, shuffle=False)
        self.global_model = model
        self.optimizer = FedAvgOptimizer()
        
        # Khởi tạo danh sách Client
        self.clients = []
        for client_train, _ in client_datasets:
            c = FedavgClient(args, client_train, model)
            self.clients.append(c)
            
    def update(self):
        print(f"[Server] Start Training for {self.args.R} rounds...")
        
        # Tạo thư mục lưu nếu chưa có
        if not os.path.exists(self.args.result_path):
            os.makedirs(self.args.result_path)

        for round in range(self.args.R):
            print(f"\n--- GLOBAL ROUND {round+1}/{self.args.R} ---")
            
            client_weights_list = []
            dataset_sizes = []
            global_w = self.global_model.state_dict()
            
            # 1. Client Training
            for i, client in enumerate(self.clients):
                client.model.load_state_dict(global_w)
                print(f"\t[Client {i+1}] Training...")
                w_local = client.update()
                client_weights_list.append(w_local)
                dataset_sizes.append(len(client.train_loader.dataset))
                
            # 2. Aggregation
            print("\t[Server] Aggregating updates...")
            self.global_model = self.optimizer.step(self.global_model, client_weights_list, dataset_sizes)
            
            # 3. Evaluation
            acc = self.evaluate()
            print(f"\t>>> Round {round+1} Result: Test Accuracy = {acc*100:.2f}%")

            # --- QUAN TRỌNG: LƯU CHECKPOINT SAU MỖI VÒNG ---
            # Lưu vào thư mục result (đã nằm trong Drive của bạn)
            save_path = os.path.join(self.args.result_path, f"{self.args.exp_name}_round{round+1}.pth")
            torch.save(self.global_model.state_dict(), save_path)
            print(f"\t[Checkpoint] Model saved to {save_path}")
            # ------------------------------------------------

    def evaluate(self):
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids, labels = batch
                input_ids = input_ids.to(self.args.device)
                labels = labels.to(self.args.device)
                
                outputs = self.global_model(input_ids=input_ids)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0
        
    def finalize(self):
        print("[Server] Training Finished.")
