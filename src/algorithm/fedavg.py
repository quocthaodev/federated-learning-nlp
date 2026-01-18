import copy
import torch

class FedAvgOptimizer:
    def step(self, global_model, client_weights_list, dataset_sizes):
        # Tính tổng số mẫu dữ liệu
        total_samples = sum(dataset_sizes)
        
        # Lấy khung trọng số của global model
        avg_weights = copy.deepcopy(client_weights_list[0])
        
        # Bắt đầu cộng dồn: w_avg = w1 * (n1/N) + w2 * (n2/N) + ...
        for key in avg_weights.keys():
            # Tính phần đóng góp của client 1
            avg_weights[key] = avg_weights[key] * (dataset_sizes[0] / total_samples)
            
            # Cộng thêm phần của các client còn lại
            for i in range(1, len(client_weights_list)):
                avg_weights[key] += client_weights_list[i][key] * (dataset_sizes[i] / total_samples)
                
        # Nạp trọng số trung bình vào global model
        global_model.load_state_dict(avg_weights)
        return global_model