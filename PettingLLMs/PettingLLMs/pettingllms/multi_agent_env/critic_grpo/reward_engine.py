import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CriticRewardEngine:
    """
    Implementation of the Outcome-Supervised Reward based on SFT+DPO & GRPO Hybrid Paper.
    This Engine isolates the Reward logic specifically for the Critic Agent AT-GRPO pipeline.
    """
    
    @staticmethod
    def compute_critic_reward(
        model_output: str,
        ground_truth_matched: bool,
        alpha: float = 1.0,         # Weight for r_accuracy
        beta: float = 1.0,          # Weight for r_format 
        gamma: float = 0.0          # Optional: Weight for r_strength
    ) -> float:
        """
        Computes the step reward based on the combined objective:
        r_final = alpha * r_accuracy + beta * r_format + gamma * r_strength
        """
        # 1. 格式提取与验证 (r_format) - Eq (4)
        verdict_match = re.search(r'<verdict>(.*?)</verdict>', model_output, re.IGNORECASE | re.DOTALL)
        feedback_match = re.search(r'<feedback>(.*?)</feedback>', model_output, re.IGNORECASE | re.DOTALL)
        
        has_valid_format = bool(verdict_match and feedback_match)
        # 论文里错给 -0.5，但我们这套多Agent框架若格式乱了后面会全部雪崩，所以我们给严厉惩罚 -2.0
        r_format = 0.0 if has_valid_format else -2.0
        
        # 解析 Verdict 结果
        pred_accept = False
        if has_valid_format:
            verdict_text = verdict_match.group(1).strip().upper()
            if "ACCEPT" in verdict_text:
                pred_accept = True
            elif "REJECT" in verdict_text:
                pred_accept = False
            else:
                # 即使有标签，内容也是胡言乱语
                r_format -= 1.0
        
        # 2. 准确性判定奖励 (r_accuracy) - Eq (3)
        # 我们采用更适合多跳任务的"非对称"硬罚逻辑 (Asymmetric Reward)
        r_accuracy = 0.0
        
        if not has_valid_format:
            r_accuracy = 0.0 # 格式不对连判断资格都没有，避免因巧合猜对而获取奖励
        else:
            if pred_accept and ground_truth_matched:
                r_accuracy = 1.0    # 完美拦截：该过则过
            elif not pred_accept and not ground_truth_matched:
                r_accuracy = 1.0    # 完美拦截：该杀则杀
            elif pred_accept and not ground_truth_matched:
                r_accuracy = -2.0   # 致命错误：放水！烂计划当成了好计划
            elif not pred_accept and ground_truth_matched:
                r_accuracy = -0.5   # 轻微错误：错杀！好计划当成烂计划，顶多重做一次

        # 3. 强度感知奖励 (r_strength) - Eq (5)
        # 预留给后续提取 <strength>1-3</strength> 与黄金标签对比。目前设为 0。
        r_strength = 0.0 

        # 4. 最终缝合 - Eq (6)
        r_final = alpha * r_accuracy + beta * r_format + gamma * r_strength
        
        return float(r_final)
