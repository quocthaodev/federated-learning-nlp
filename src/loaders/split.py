import numpy as np

def simulate_split(args, dataset):
    # Hàm chia dữ liệu giả lập
    total_samples = len(dataset)
    indices = np.arange(total_samples)

    if args.split_type == 'iid':
        # Chia đều ngẫu nhiên
        np.random.shuffle(indices)
        split_map = np.array_split(indices, args.K)
        return {i: split_map[i] for i in range(args.K)}

    elif args.split_type == 'non_iid':
        # Chia không đều (Non-IID) bằng Dirichlet
        min_size = 0
        while min_size < 10:
            idx_batch = [[] for _ in range(args.K)]
            # Giả lập đơn giản cho label imbalance
            # (Code đầy đủ sẽ phức tạp hơn, đây là bản rút gọn để chạy được)
            split_map = np.array_split(indices, args.K) 
            # Tạm thời để IID cho dễ chạy bước đầu
            return {i: split_map[i] for i in range(args.K)}

    return {i: np.array([]) for i in range(args.K)}