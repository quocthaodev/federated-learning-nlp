import os
import argparse
import torch
from src.utils import set_seed
from src.loaders import load_dataset
from src.models import BioBERT
from src.server import FedavgServer

def main(args):
    # 1. Cấu hình cơ bản
    set_seed(args.seed)
    
    # Tự động chọn thiết bị
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        print("[WARNING] Không tìm thấy GPU, chuyển sang chạy CPU.")
    
    device = torch.device(args.device)
    args.data_path = './data'
    args.result_path = './result'
    if not os.path.exists(args.result_path): os.makedirs(args.result_path)

    # 2. Load Dữ liệu
    print(f"[Main] Loading dataset: {args.dataset}...")
    client_datasets, server_testset = load_dataset(args)
    
    # 3. Load Model
    print(f"[Main] Loading model: {args.model_name} ({args.pretrained_model_name})...")
    model = BioBERT(pretrained_model_name=args.pretrained_model_name)
    model.to(device)

    # 4. Khởi tạo Server
    print(f"[Main] Initializing Server with algorithm: {args.algorithm}...")
    
    # Dummy writer để tránh lỗi tensorboard (nếu code server yêu cầu)
    class DummyWriter:
        def add_scalar(self, *args): pass
        def close(self): pass
    
    server = FedavgServer(args, DummyWriter(), server_testset, client_datasets, model)

    # 5. Bắt đầu Train
    print("[Main] Start Training...")
    server.update() 
    
    # 6. Kết thúc
    print("[Main] Training Completed.")
    server.finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Các tham số quan trọng
    parser.add_argument('--exp_name', type=str, required=True, help='Tên thí nghiệm')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='MedQA')
    parser.add_argument('--model_name', type=str, default='BioBERT')
    parser.add_argument('--pretrained_model_name', type=str, default='dmis-lab/biobert-v1.1')
    
    # Tham số Federated Learning
    parser.add_argument('--algorithm', type=str, default='fedavg')
    parser.add_argument('--split_type', type=str, default='iid')
    parser.add_argument('--K', type=int, default=5, help='Number of clients')
    parser.add_argument('--R', type=int, default=20, help='Total rounds') 
    parser.add_argument('--E', type=int, default=1, help='Local epochs')
    parser.add_argument('--B', type=int, default=4, help='Batch size')
    
    # Tham số Optimizer
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    args = parser.parse_args()
    main(args)
