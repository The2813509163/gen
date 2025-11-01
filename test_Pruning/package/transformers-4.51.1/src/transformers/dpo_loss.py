import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from collections import defaultdict
import math

class AdvancedParameterDistributionDPOLoss(nn.Module):
    def __init__(self, beta=0.1, reward_type='kl_divergence', layer_weights=None, 
                 histogram_bins=50, epsilon=1e-8):
        """
        改进的基于参数分布的DPO损失函数
        
        Args:
            beta: DPO loss中的温度参数
            reward_type: reward函数类型
                - 'kl_divergence': KL散度 (推荐)
                - 'js_divergence': Jensen-Shannon散度
                - 'wasserstein_distance': Wasserstein距离
                - 'histogram_intersection': 直方图交集
                - 'bhattacharyya_distance': Bhattacharyya距离
                - 'cosine_similarity': 余弦相似度
                - 'l2_distance': L2距离
                - 'parameter_correlation': 参数相关性
                - 'distribution_similarity': 参数值分布相似性
                - 'combined': 组合多种度量
            layer_weights: 不同层的权重字典
            histogram_bins: 直方图分箱数量
            epsilon: 数值稳定性参数
        """
        super().__init__()
        self.beta = beta
        self.reward_type = reward_type
        self.layer_weights = layer_weights or {}
        self.histogram_bins = histogram_bins
        self.epsilon = epsilon
        self.initial_params = {}
        self.initial_param_distributions = {}
        
    def store_initial_parameters(self, model):
        """存储初始模型参数和分布信息"""
        self.initial_params = {}
        self.initial_param_distributions = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_data = param.data.clone().detach()
                self.initial_params[name] = param_data
                
                # 存储参数分布信息
                param_flat = param_data.flatten()
                self.initial_param_distributions[name] = {
                    'values': param_flat,
                    'mean': param_flat.mean().item(),
                    'std': param_flat.std().item(),
                    'min': param_flat.min().item(),
                    'max': param_flat.max().item(),
                    'histogram': self._compute_histogram(param_flat),
                    'shape': param_data.shape
                }
    
    def _compute_histogram(self, tensor, bins=None):
        """计算参数的直方图"""
        if bins is None:
            bins = self.histogram_bins
        
        tensor_np = tensor.detach().cpu().numpy()
        hist, bin_edges = np.histogram(tensor_np, bins=bins, density=True)
        
        # 转换为概率分布
        hist = hist / (hist.sum() + self.epsilon)
        return {
            'hist': torch.tensor(hist, dtype=torch.float32),
            'bin_edges': torch.tensor(bin_edges, dtype=torch.float32)
        }
    
    def _tensor_to_distribution(self, tensor):
        """将tensor转换为概率分布"""
        tensor_flat = tensor.flatten()
        
        # 使用softmax将参数值转换为概率分布
        # 先进行归一化避免数值问题
        normalized = (tensor_flat - tensor_flat.mean()) / (tensor_flat.std() + self.epsilon)
        prob_dist = F.softmax(normalized, dim=0)
        
        return prob_dist
    
    
    
    def js_divergence_reward(self, current_params):
        """基于Jensen-Shannon散度的reward函数"""
        total_js = 0
        total_weight = 0
        
        for name, current_param in current_params.items():
            if name in self.initial_params:
                initial_hist = self.initial_param_distributions[name]['histogram']['hist']
                current_hist = self._compute_histogram(current_param.flatten())['hist']
                
                # 归一化
                initial_hist = initial_hist + self.epsilon
                current_hist = current_hist + self.epsilon
                initial_hist = initial_hist / initial_hist.sum()
                current_hist = current_hist / current_hist.sum()
                
                # 计算JS散度
                m = 0.5 * (initial_hist + current_hist)
                js_div = 0.5 * F.kl_div(torch.log(initial_hist), m, reduction='sum') + \
                        0.5 * F.kl_div(torch.log(current_hist), m, reduction='sum')
                
                weight = self.layer_weights.get(name, 1.0)
                total_js += js_div.item() * weight
                total_weight += weight
        
        avg_js = total_js / total_weight if total_weight > 0 else 0
        return -avg_js  # JS散度越小，reward越高
    
    def wasserstein_distance_reward(self, current_params):
        """基于Wasserstein距离的reward函数"""
        total_wasserstein = 0
        total_weight = 0
        
        for name, current_param in current_params.items():
            if name in self.initial_params:
                initial_values = self.initial_param_distributions[name]['values']
                current_values = current_param.flatten()
                
                # 计算1-Wasserstein距离 (Earth Mover's Distance)
                initial_sorted, _ = torch.sort(initial_values)
                current_sorted, _ = torch.sort(current_values)
                
                # 如果长度不同，需要插值对齐
                if len(initial_sorted) != len(current_sorted):
                    min_len = min(len(initial_sorted), len(current_sorted))
                    initial_sorted = initial_sorted[:min_len]
                    current_sorted = current_sorted[:min_len]
                
                wasserstein_dist = torch.mean(torch.abs(initial_sorted - current_sorted)).item()
                
                weight = self.layer_weights.get(name, 1.0)
                total_wasserstein += wasserstein_dist * weight
                total_weight += weight
        
        avg_wasserstein = total_wasserstein / total_weight if total_weight > 0 else 0
        return -avg_wasserstein  # 距离越小，reward越高
    
    def histogram_intersection_reward(self, current_params):
        """基于直方图交集的reward函数"""
        total_intersection = 0
        total_weight = 0
        
        for name, current_param in current_params.items():
            if name in self.initial_params:
                initial_hist = self.initial_param_distributions[name]['histogram']['hist']
                current_hist = self._compute_histogram(current_param.flatten())['hist']
                
                # 计算直方图交集
                intersection = torch.sum(torch.min(initial_hist, current_hist)).item()
                
                weight = self.layer_weights.get(name, 1.0)
                total_intersection += intersection * weight
                total_weight += weight
        
        return total_intersection / total_weight if total_weight > 0 else 0
    
    def bhattacharyya_distance_reward(self, current_params):
        """基于Bhattacharyya距离的reward函数"""
        total_bhattacharyya = 0
        total_weight = 0
        
        for name, current_param in current_params.items():
            if name in self.initial_params:
                initial_hist = self.initial_param_distributions[name]['histogram']['hist']
                current_hist = self._compute_histogram(current_param.flatten())['hist']
                
                # 归一化
                initial_hist = initial_hist + self.epsilon
                current_hist = current_hist + self.epsilon
                initial_hist = initial_hist / initial_hist.sum()
                current_hist = current_hist / current_hist.sum()
                
                # 计算Bhattacharyya系数
                bc = torch.sum(torch.sqrt(initial_hist * current_hist))
                # Bhattacharyya距离
                bd = -torch.log(bc + self.epsilon)
                
                weight = self.layer_weights.get(name, 1.0)
                total_bhattacharyya += bd.item() * weight
                total_weight += weight
        
        avg_bhattacharyya = total_bhattacharyya / total_weight if total_weight > 0 else 0
        return -avg_bhattacharyya  # 距离越小，reward越高
    
    def combined_reward(self, current_params):
        """组合多种度量的reward函数"""
        rewards = {}
        
        # 计算各种度量
        rewards['kl'] = self.kl_divergence_reward(current_params)
        rewards['js'] = self.js_divergence_reward(current_params)
        rewards['wasserstein'] = self.wasserstein_distance_reward(current_params)
        rewards['intersection'] = self.histogram_intersection_reward(current_params)
        rewards['bhattacharyya'] = self.bhattacharyya_distance_reward(current_params)
        
        # 权重组合
        weights = {
            'kl': 0.3,
            'js': 0.2,
            'wasserstein': 0.2,
            'intersection': 0.15,
            'bhattacharyya': 0.15
        }
        
        combined_reward = sum(weights[key] * rewards[key] for key in weights)
        return combined_reward
    
    # 保留原有的reward函数
    def cosine_similarity_reward(self, current_params):
        """基于余弦相似度的reward函数"""
        total_similarity = 0
        total_weight = 0
        
        for name, current_param in current_params.items():
            if name in self.initial_params:
                initial_param = self.initial_params[name]
                
                initial_flat = initial_param.flatten()
                current_flat = current_param.flatten()
                
                similarity = F.cosine_similarity(
                    initial_flat.unsqueeze(0), 
                    current_flat.unsqueeze(0)
                ).item()
                
                weight = self.layer_weights.get(name, 1.0)
                total_similarity += similarity * weight
                total_weight += weight
        
        return total_similarity / total_weight if total_weight > 0 else 0
    
    def l2_distance_reward(self, current_params):
        """基于L2距离的reward函数"""
        total_distance = 0
        total_weight = 0
        
        for name, current_param in current_params.items():
            if name in self.initial_params:
                initial_param = self.initial_params[name]
                
                distance = torch.norm(current_param - initial_param).item()
                initial_norm = torch.norm(initial_param).item()
                normalized_distance = distance / (initial_norm + self.epsilon)
                
                weight = self.layer_weights.get(name, 1.0)
                total_distance += normalized_distance * weight
                total_weight += weight
        
        avg_distance = total_distance / total_weight if total_weight > 0 else 0
        return -avg_distance
    
    def kl_divergence_reward(self, current_params):
        """基于KL散度的reward函数"""
        device = next(iter(current_params.values())).device
        total_kl = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_weight = 0.0
        
        for name, current_param in current_params.items():
            if name in self.initial_params:
                #基于参数值直接转换的概率分布
                initial_param_tensor = self.initial_params[name].to(device)
                initial_prob = self._tensor_to_distribution(initial_param_tensor)
                current_prob = self._tensor_to_distribution(current_param)
                
                kl_div_direct = F.kl_div(
                    torch.log(current_prob + self.epsilon),
                    initial_prob,
                    reduction='sum'
                )

                weight = self.layer_weights.get(name, 1.0)
                total_kl = total_kl + kl_div_direct * weight
                total_weight = total_weight + weight
        
        # 6. 计算平均KL散度，结果仍为张量
        if total_weight > 0:
            # 确保除数也是张量，或者是一个标量，PyTorch会自动处理
            avg_kl = total_kl / total_weight
            # avg_kl = total_kl
        else:
            avg_kl = torch.tensor(0.0, device=device)
        return -avg_kl  # KL散度越小，reward越高
    def compute_parameter_reward(self, model):
        """计算参数分布reward"""
        current_params = {}
        #print("AAAA:",model.named_parameters())
        #print("AAA:",self.initial_params.keys())
        for name, param in model.named_parameters():
            #print("name:",name)
            #print("param:",param.requires_grad)
            if param.requires_grad and name in self.initial_params:
                #print("OKOKOKOKOK")
                current_params[name] = param
        
        reward_functions = {
            'kl_divergence': self.kl_divergence_reward,
            # 'js_divergence': self.js_divergence_reward,
            # 'wasserstein_distance': self.wasserstein_distance_reward,
            # 'histogram_intersection': self.histogram_intersection_reward,
            # 'bhattacharyya_distance': self.bhattacharyya_distance_reward,
            # 'combined': self.combined_reward,
            # 'cosine_similarity': self.cosine_similarity_reward,
            # 'l2_distance': self.l2_distance_reward,
        }
        
        if self.reward_type not in reward_functions:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        
        return reward_functions[self.reward_type](current_params)
    
    
    def forward(self, model, task_loss=None, return_details=False):
        """
        计算参数分布DPO损失
        
        Args:
            model: 当前模型
            task_loss: 原始任务损失（可选）
            return_details: 是否返回详细信息
        """
        if not self.initial_params:
            raise ValueError("Initial parameters not stored. Call store_initial_parameters() first.")
        
        # 计算当前模型参数的reward
        current_reward = self.compute_parameter_reward(model)
        # current_reward = torch.tensor(0.0)
        # preferred是初始参数分布，reward设为0作为基准
        preferred_reward = torch.tensor(0.0, device=current_reward.device)
        
        # 计算DPO损失
        # 我们希望current_reward尽可能高（接近初始分布）
        reward_diff = current_reward - preferred_reward  # current_reward应该接近0（初始状态）
        # reward_diff = torch.tensor(0.0)
        dpo_loss = -F.logsigmoid(self.beta * reward_diff)
        #dpo_loss = 0
       # print("dpo_loss:",dpo_loss)
        if task_loss is not None:
            return task_loss + dpo_loss
        else:
            return dpo_loss
# 自适应权重版本
class AdaptiveParameterDistributionDPOLoss(AdvancedParameterDistributionDPOLoss):
    def __init__(self, beta=0.1, reward_type='kl_divergence', 
                 adaptive_weights=True, importance_threshold=0.1, 
                 weight_update_frequency=10):
        super().__init__(beta, reward_type)
        self.adaptive_weights = adaptive_weights
        self.importance_threshold = importance_threshold
        self.weight_update_frequency = weight_update_frequency
        self.step_count = 0
        self.layer_importance = {}
        self.gradient_history = defaultdict(list)
    
    def update_layer_importance(self, model):
        """更新层重要性权重"""
        if not self.adaptive_weights:
            return
        
        self.step_count += 1
        
        # 收集梯度信息
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and name in self.initial_params:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                importance = grad_norm / (param_norm + self.epsilon)
                self.gradient_history[name].append(importance)
                
                # 保持历史记录长度
                if len(self.gradient_history[name]) > 100:
                    self.gradient_history[name] = self.gradient_history[name][-100:]
        
        # 定期更新权重
        if self.step_count % self.weight_update_frequency == 0:
            self.layer_importance = {}
            for name, history in self.gradient_history.items():
                if history:
                    avg_importance = np.mean(history[-10:])  # 使用最近10步的平均值
                    self.layer_importance[name] = max(avg_importance, self.importance_threshold)
                else:
                    self.layer_importance[name] = 1.0
            
            self.layer_weights = self.layer_importance
    
    def forward(self, model, task_loss=None, return_details=False):
        """重写forward方法，加入自适应权重更新"""
        self.update_layer_importance(model)
        return super().forward(model, task_loss, return_details)

# 使用示例和测试
def comprehensive_example():
    """综合使用示例"""
    from transformers import GPT2LMHeadModel
    
    # 创建模型
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # 测试不同的reward类型
    reward_types = ['kl_divergence', 'js_divergence', 'wasserstein_distance', 
                   'histogram_intersection', 'bhattacharyya_distance', 'combined']
    
    results = {}
    
    for reward_type in reward_types:
        print(f"\n测试 {reward_type} reward函数:")
        
        # 创建损失函数
        param_dpo_loss = AdvancedParameterDistributionDPOLoss(
            beta=0.1, 
            reward_type=reward_type,
            histogram_bins=30
        )
        
        # 存储初始参数
        param_dpo_loss.store_initial_parameters(model)
        
        # 模拟参数更新
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 添加小的随机扰动
                    param.data += torch.randn_like(param) * 0.01
        
        # 计算损失
        dpo_loss, details = param_dpo_loss(model, return_details=True)
        results[reward_type] = details
        
        print(f"  Current Reward: {details['current_reward']:.6f}")
        print(f"  DPO Loss: {details['dpo_loss']:.6f}")
    
    # 比较结果
    print("\n=== 结果比较 ===")
    for reward_type, details in results.items():
        print(f"{reward_type:20s}: Reward={details['current_reward']:8.6f}, Loss={details['dpo_loss']:8.6f}")

def training_example():
    """训练示例"""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 创建参数分布DPO损失
    param_dpo_loss = AdaptiveParameterDistributionDPOLoss(
        beta=0.1, 
        reward_type='kl_divergence',
        adaptive_weights=True
    )
    
    # 存储初始参数
    param_dpo_loss.store_initial_parameters(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # 模拟训练数据
    texts = ["Hello world", "How are you doing today?", "This is a test sentence."]
    
    model.train()
    for epoch in range(3):
        total_loss = 0
        
        for text in texts:
            # 准备数据
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            optimizer.zero_grad()
            
            # 计算任务损失
            outputs = model(**inputs, labels=inputs['input_ids'])
            task_loss = outputs.loss
            
            # 计算参数分布DPO损失
            total_loss_value, details = param_dpo_loss(
                model, task_loss, return_details=True
            )
            
            total_loss_value.backward()
            optimizer.step()
            
            total_loss += total_loss_value.item()
        
        print(f"Epoch {epoch+1}: Average Loss = {total_loss/len(texts):.4f}")
        print(f"  Task Loss: {details['task_loss']:.4f}")
        print(f"  DPO Loss: {details['dpo_loss']:.4f}")
        print(f"  Current Reward: {details['current_reward']:.6f}")

if __name__ == "__main__":
    print("=== 综合测试 ===")
    comprehensive_example()
    
    print("\n=== 训练示例 ===")
    training_example()