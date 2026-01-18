import torch
from src.loaders.split import simulate_split

def load_dataset(args):
    # Tải dữ liệu MedQA hoặc PubMedQA
    if args.dataset == 'MedQA':
        from src.datasets.medqa import fetch_medqa
        raw_train, raw_test = fetch_medqa(args, root=args.data_path)
    elif args.dataset == 'PubMedQA':
        from src.datasets.pubmedqa import fetch_pubmedqa
        raw_train, raw_test = fetch_pubmedqa(args, root=args.data_path)
    else:
        raise ValueError('Dataset not supported')

    # Chia dữ liệu cho K khách hàng
    split_map = simulate_split(args, raw_train)

    client_datasets = []
    for client_id in range(args.K):
        indices = split_map[client_id]
        # Tạo tập con local cho từng client
        local_train = torch.utils.data.Subset(raw_train, indices)
        # Mỗi client giữ 1 bản copy local train và toàn bộ test set (để server test)
        client_datasets.append((local_train, raw_test))

    return client_datasets, raw_test