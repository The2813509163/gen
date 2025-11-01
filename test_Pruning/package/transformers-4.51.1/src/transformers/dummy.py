import numpy as np
import torch
from scipy.stats import norm

class GaussianStratifiedSampler:
    """
    一个从模型词典中进行分层高斯采样的采样器。
    
    它首先根据高斯分布为整个词典计算一个全局的概率权重，
    然后在采样时，根据给定的分层权重选择一个分层，
    并在此分层内部根据预计算的概率进行加权采样。
    """
    def __init__(self, tokenizer, mu, sigma):
        """
        初始化采样器。
        
        Args:
            tokenizer: Hugging Face的tokenizer对象。
            mu (float): 高斯分布的均值 (采样的中心Token ID)。
            sigma (float): 高斯分布的标准差 (采样的分散程度)。
        """
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        
        # 1. 定义词典分层 (可以根据需要调整范围)
        self.strata = {
            "control_and_common": (0, 2000),      # 控制符和最高频词
            "mid_frequency": (2001, 30000),     # 中频词
            "long_tail": (30001, self.vocab_size)   # 长尾/罕见词
        }
        
        print("正在预计算全局高斯概率分布...")
        # 2. 一次性预计算整个词典的高斯概率
        token_ids = np.arange(self.vocab_size)
        self.global_probabilities = norm.pdf(token_ids, loc=mu, scale=sigma)
        print("全局概率计算完成。")
        
        # 3. 为每个分层预计算其内部的概率和ID，方便后续使用
        self.strata_info = {}
        for name, (start, end) in self.strata.items():
            ids_in_stratum = np.arange(start, end)
            probs_in_stratum = self.global_probabilities[start:end]
            # 归一化，使得该分层内部的概率和为1
            normalized_probs = probs_in_stratum / np.sum(probs_in_stratum)
            
            self.strata_info[name] = {
                "ids": ids_in_stratum,
                "probs": normalized_probs
            }
            
    def generate_batch(self, batch_size, seq_len, strata_weights=None):
        """
        生成一个批次的dummy数据。
        
        Args:
            batch_size (int): 批次大小。
            seq_len (int): 每个序列的长度。
            strata_weights (dict, optional): 一个字典，指定从每个分层采样的概率。
                                             例如: {'control_and_common': 0.2, 'mid_frequency': 0.7, 'long_tail': 0.1}
                                             如果为None，则默认使用均匀权重。
        
        Returns:
            torch.Tensor: 生成的dummy token ID批次，形状为 (batch_size, seq_len)。
        """
        if strata_weights is None:
            # 如果未提供权重，则每个分层被选中的概率相等
            strata_names = list(self.strata.keys())
            strata_probs = [1.0 / len(strata_names)] * len(strata_names)
        else:
            strata_names = list(strata_weights.keys())
            strata_probs = list(strata_weights.values())

        dummy_batch = []
        for _ in range(batch_size):
            dummy_sequence = []
            for _ in range(seq_len):
                # 步骤1: 根据权重选择一个分层
                chosen_stratum_name = np.random.choice(strata_names, p=strata_probs)
                
                # 步骤2: 从选定的分层内部进行加权(高斯)采样
                stratum = self.strata_info[chosen_stratum_name]
                token_id = np.random.choice(
                    a=stratum["ids"],
                    p=stratum["probs"]
                )
                dummy_sequence.append(token_id)
            
            dummy_batch.append(dummy_sequence)
            
        return torch.tensor(dummy_batch, dtype=torch.long)

class ZipfSampler:
    """
    一个根据齐夫定律 (Zipf's Law) 从模型词典中进行全局采样的采样器。

    齐夫定律指出，词的频率与其排名成反比。我们假设 token ID 本身就代表了
    词频的排名（ID越小，排名越靠前）。
    该采样器的概率分布遵循 P(rank) ∝ 1 / (rank^a)，其中 'a' 是分布的指数。
    """
    def __init__(self, tokenizer, a=1.0):
        """
        初始化采样器。

        Args:
            tokenizer: Hugging Face的tokenizer对象。
            a (float, optional): 齐夫分布的指数，默认为1.0。
                                 a > 1 会使分布更陡峭（更集中于高频词）。
                                 a < 1 会使分布更平缓。
        """
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.a = a

        print(f"正在根据指数 a={self.a} 预计算齐夫概率分布...")
        
        # 1. 创建一个从1开始的排名数组 (rank = token_id + 1)
        # token ID 从 0 开始，但排名从 1 开始
        ranks = np.arange(1, self.vocab_size + 1)
        
        # 2. 根据齐夫定律计算每个rank的权重
        weights = 1.0 / (ranks ** self.a)
        
        # 3. 将权重归一化，得到全局概率分布
        self.probabilities = weights / np.sum(weights)
        print("全局概率计算完成。")

    def generate_batch(self, batch_size, seq_len):
        """
        生成一个批次的dummy数据。
        此版本经过优化，可一次性生成所有token，效率很高。

        Args:
            batch_size (int): 批次大小。
            seq_len (int): 每个序列的长度。

        Returns:
            torch.Tensor: 生成的dummy token ID批次，形状为 (batch_size, seq_len)。
        """
        # 计算总共需要生成的token数量
        num_tokens = batch_size * seq_len
        
        # 使用np.random.choice一次性从全局概率分布中采样所有需要的token
        sampled_token_ids = np.random.choice(
            a=self.vocab_size,         # 从词汇表 [0, 1, ..., vocab_size-1] 中选择
            size=num_tokens,
            p=self.probabilities,
            replace=True               # 允许重复采样
        )
        
        # 将一维的token数组重塑为 (batch_size, seq_len) 的形状
        batch_array = sampled_token_ids.reshape(batch_size, seq_len)
        
        # 转换为PyTorch张量
        return torch.tensor(batch_array, dtype=torch.long)