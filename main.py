import os
import argparse
import torch
import logging
from src.utils import set_seed, init_weights
from src.loaders import load_dataset
from src.models import BioBERT
from src.server import FedavgServer

def main(args):
    # 1. Cấu hình cơ bản
    set_seed(args.seed)
    device = torch.device(args.device)
    # Đảm bảo đường dẫn result đúng
    args.data_path = './data'
    args.result_path = './result'
    if not os.path.exists(args.result_path): os.makedirs(args.result_path)

    # 2. Load Dữ liệu
    print(f"[Main] Loading dataset: {args.dataset}...")
    client_datasets, server_testset = load_dataset(args)
    
    # 3. Load Model & CHECKPOINT (ĐÃ SỬA ĐOẠN NÀY)
    print(f"[Main] Loading model: {args.model_name} ({args.pretrained_model_name})...")
    model = BioBERT(pretrained_model_name=args.pretrained_model_name)
    
    # --- ĐOẠN CODE THÊM VÀO ĐỂ RESUME ---
    # Tên file checkpoint bạn muốn load (Exp1_BioBERT_Real_round4.pth)
    checkpoint_path = os.path.join(args.result_path, "Exp1_BioBERT_Real_round4.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"\n[RESUME] >>> Tìm thấy Checkpoint: {checkpoint_path}")
        print(f"[RESUME] >>> Đang nạp trọng số từ vòng 4 để chạy tiếp...")
        # Load trọng số vào model (map_location để tránh lỗi lệch device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"\n[WARNING] Không tìm thấy file {checkpoint_path}, sẽ chạy từ đầu (Start from scratch)!")
    # ------------------------------------

    model.to(device)

    # 4. Khởi tạo Server
    print(f"[Main] Initializing Server with algorithm: {args.algorithm}...")
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
    # Các tham số điều khiển
    parser.add_argument('--exp_name', type=str, default='test_run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset', type=str, default='MedQA')
    parser.add_argument('--model_name', type=str, default='BioBERT')
    parser.add_argument('--pretrained_model_name', type=str, default='dmis-lab/biobert-v1.1')
    parser.add_argument('--algorithm', type=str, default='fedavg')
    parser.add_argument('--split_type', type=str, default='iid')
    parser.add_argument('--K', type=int, default=2, help='Number of clients')
    parser.add_argument('--R', type=int, default=1, help='Number of rounds')
    parser.add_argument('--E', type=int, default=1, help='Local epochs')
    parser.add_argument('--B', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--init_type', type=str, default='xavier')
    parser.add_argument('--init_gain', type=float, default=1.0)
    
    args = parser.parse_args()
    main(args)