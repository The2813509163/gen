import numpy as np
import torch
import os
import pickle
from tqdm import tqdm
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

class PretrainDataSampler:
    """
    一个根据预计算的token频率文件 (.pkl) 从模型词典中进行采样的采样器。
    
    这个采样器不依赖任何理论分布（如高斯或齐夫），而是直接使用
    从真实预训练数据中统计出的token频率作为采样依据。
    """
    def __init__(self, tokenizer, freq_file_path):
        """
        初始化采样器。

        Args:
            tokenizer: Hugging Face的tokenizer对象。
            freq_file_path (str): 之前脚本生成的包含token频率(Counter对象)的.pkl文件路径。
        """
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        print(f"检测到词汇表大小 (vocab_size): {self.vocab_size}")

        if not os.path.exists(freq_file_path):
            raise FileNotFoundError(
                f"错误: 频率文件 '{freq_file_path}' 不存在。\n"
                f"请先运行之前的频率统计脚本来生成此文件。"
            )

        print(f"正在从 '{freq_file_path}' 加载预计算的token频率...")
        with open(freq_file_path, 'rb') as f:
            token_counts = pickle.load(f)
        print("频率文件加载成功。")

        print("正在构建基于真实频率的概率分布...")
        # 1. 创建一个长度为词汇表大小的全零数组，用于存放每个token ID的频率计数
        freq_array = np.zeros(self.vocab_size, dtype=np.float64)

        # 2. 遍历词汇表中的每一个token ID
        for token_id in tqdm(range(self.vocab_size), desc="构建概率分布"):
            # 将token ID转换为token字符串 (例如, 123 -> " Llama")
            token_str = self.tokenizer.convert_ids_to_tokens(token_id)
            
            # 从加载的Counter中获取该token的计数，如果不存在则默认为0
            count = token_counts.get(token_str, 0)
            
            # 将计数存放到数组的对应位置
            freq_array[token_id] = count
        
        # 3. 将计数数组归一化，得到最终的概率分布
        total_tokens = np.sum(freq_array)
        if total_tokens > 0:
            self.probabilities = freq_array / total_tokens
        else:
            # 如果文件中没有任何有效计数，则退回到均匀分布以避免错误
            print("警告: 文件中未找到有效token计数，将使用均匀分布。")
            self.probabilities = np.ones(self.vocab_size, dtype=np.float64) / self.vocab_size
            
        print("基于真实数据频率的概率分布计算完成。")
        
    def generate_batch(self, batch_size, seq_len):
        """
        生成一个基于预训练数据频率的dummy token序列。
        此方法经过优化，可一次性生成所有token，效率很高。

        Args:
            seq_len (int): 序列的长度。

        Returns:
            np.ndarray: 生成的dummy token ID序列。
        """
        # 计算总共需要生成的token数量
        num_tokens = batch_size * seq_len
        
        # 使用np.random.choice一次性从全局概率分布中采样所有需要的token
        sampled_token_ids = np.random.choice(
            a=self.vocab_size,       # 从词汇表 [0, 1, ..., vocab_size-1] 中选择
            size=num_tokens,            # 生成的token数量
            p=self.probabilities,    # 使用我们构建的真实概率分布
            replace=True             # 允许重复采样
        )
          # 将一维的token数组重塑为 (batch_size, seq_len) 的形状
        batch_array = sampled_token_ids.reshape(batch_size, seq_len)
        
        # 转换为PyTorch张量
        return torch.tensor(batch_array, dtype=torch.long)


class MixedDistributionSampler:
    """
    一个能在初始化时动态计算“差集混合分布”的采样器。
    
    它会加载预训练数据和数学数据的token频率文件，然后根据
    P_dummy ∝ P_pretrain / (P_math + ε) 的公式来构建最终的采样概率。
    """
    def __init__(self, tokenizer, pretrain_freq_path, math_freq_path, epsilon=1e-10):
        """
        初始化采样器并完成所有概率计算。

        Args:
            tokenizer: Hugging Face的tokenizer对象。
            pretrain_freq_path (str): 预训练数据token频率的.pkl文件路径。
            math_freq_path (str): 数学领域数据token频率的.pkl文件路径。
            epsilon (float): 一个极小值，用于防止计算中出现除以零的错误。
        """
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        print(f"检测到词汇表大小 (vocab_size): {self.vocab_size}")

        for path in [pretrain_freq_path, math_freq_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"错误: 频率文件 '{path}' 不存在。")
        
        print(f"正在从 '{pretrain_freq_path}' 和 '{math_freq_path}' 加载频率数据...")
        with open(pretrain_freq_path, 'rb') as f:
            pretrain_freq_map = pickle.load(f)
        with open(math_freq_path, 'rb') as f:
            math_freq_map = pickle.load(f)
        print("频率文件加载成功。")

        print("正在将频率计数转换为概率分布...")
        # 核心逻辑不变，调用修正后的辅助函数
        p_pretrain = self._freq_map_to_prob_dist(pretrain_freq_map, "预训练数据")
        p_math = self._freq_map_to_prob_dist(math_freq_map, "数学数据")
        
        print("正在计算核心的“差集混合分布”...")
        unnormalized_dummy_dist = p_pretrain / (p_math + epsilon)

        total_dummy_prob = np.sum(unnormalized_dummy_dist)
        if total_dummy_prob > 0:
            self.probabilities = unnormalized_dummy_dist / total_dummy_prob
        else:
            print("警告: 无法生成有效的混合分布，将退回到均匀分布。")
            self.probabilities = np.ones(self.vocab_size, dtype=np.float64) / self.vocab_size

        assert np.isclose(np.sum(self.probabilities), 1.0), "错误：最终概率和不为1！"
        print(f"基于“差集混合分布”的采样器初始化完成！最终概率和: {np.sum(self.probabilities):.6f}")

    # ==================== (!!! 核心修正区域 !!!) ====================
    def _freq_map_to_prob_dist(self, freq_map, name):
        """
        一个辅助函数，将频率字典转换为概率Numpy数组。
        此版本已修正，可以处理key为字符串token的情况。
        """
        dist_array = np.zeros(self.vocab_size, dtype=np.float64)
        
        # 将字符串token转换为ID
        # 使用 self.tokenizer.convert_tokens_to_ids() 批量转换，效率更高
        print(f"正在将 {name} 的字符串tokens批量转换为整数IDs...")
        tokens_str = list(freq_map.keys())
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens_str)
        
        print(f"正在填充 {name} 的频率到概率数组中...")
        # 现在我们同时迭代ID和字符串token
        for token_id, token_str in tqdm(zip(token_ids, tokens_str), total=len(tokens_str), desc=f"处理 {name} 频率"):
            
            # 检查转换后的ID是否是未知的 (UNK token)
            # LLaMA3 tokenizer中，unk_token_id 通常不是一个固定值，但我们可以检查返回的ID是否有效
            if token_id is not None and token_id < self.vocab_size:
                freq = freq_map[token_str]
                dist_array[token_id] += freq # 使用 += 以处理可能多个token映射到同一个ID的情况（虽然少见）

        total_freq = np.sum(dist_array)
        if total_freq == 0:
            print(f"警告: {name} 文件中未找到有效token计数，其分布将为零。")
            return np.zeros(self.vocab_size, dtype=np.float64)

        return dist_array / total_freq
    # ==================== (!!! 修正结束 !!!) ====================

    def generate_batch(self, batch_size, seq_len):
        """
        生成一个基于混合概率分布的dummy token序列。
        """
        # 计算总共需要生成的token数量
        num_tokens = batch_size * seq_len
        
        # 使用np.random.choice一次性从全局概率分布中采样所有需要的token
        sampled_token_ids = np.random.choice(
            a=self.vocab_size,       # 从词汇表 [0, 1, ..., vocab_size-1] 中选择
            size=num_tokens,            # 生成的token数量
            p=self.probabilities,    # 使用我们构建的真实概率分布
            replace=True             # 允许重复采样
        )
          # 将一维的token数组重塑为 (batch_size, seq_len) 的形状
        batch_array = sampled_token_ids.reshape(batch_size, seq_len)
        
        # 转换为PyTorch张量
        return torch.tensor(batch_array, dtype=torch.long)
