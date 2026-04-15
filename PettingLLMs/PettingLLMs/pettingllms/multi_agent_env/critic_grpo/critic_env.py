import logging
import copy
from typing import Dict, Any, List
from dataclasses import dataclass, field

from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.critic_grpo.reward_engine import CriticRewardEngine

logger = logging.getLogger(__name__)

@dataclass
class CriticEnvState:
    """State class for Pure Offline Critic GRPO Environment"""
    question: str = ""
    ground_truth_answer: str = ""
    
    # 离线固定的 Planner 计划
    planner_plan: str = ""
    
    # 离线环境：由于是不跑 Reader 的阶段一，该计划到底是好是坏（Reader 跑完对不对），在数据集中就已经预先知道
    ground_truth_matched: bool = False 
    
    # Critic 动作状态
    critic_raw_response: str = ""
    critic_feedback: str = ""
    critic_verdict: str = ""
    
    # 单步奖励
    critic_reward: float = 0.0

class CriticGRPOEnv(Env):
    """
    Environment dedicated to Pure Offline Critic Training (Stage 1).
    This implies the Ground Truth Match is already known from the dataset offline,
    so there is NO need to simulate the Reader online here.
    """

    def __init__(self, env_idx: int, rollout_idx: int, max_turns: int, config: dict | None = None):
        super().__init__(env_idx=env_idx, rollout_idx=rollout_idx, config=config)
        self.state = CriticEnvState()

    def reset(self):
        """Reset intermediate generated states but retain offline loaded problem data"""
        self.state.critic_raw_response = ""
        self.state.critic_verdict = ""
        self.state.critic_feedback = ""
        self.state.critic_reward = 0.0

    async def step(self, role: str, action: str, env_worker: Any = None):
        """
        Execute an action. Only the 'critic' role is active in this Pure Offline environment.
        """
        if role == "critic":
            await self._critic_step(action)
        else:
            logger.warning(f"Role {role} step is not supported in the Pure Offline CriticGRPOEnv.")

    async def _critic_step(self, action: str):
        """
        Process the Critic's action. The reward is immediately computed based on 
        the offline GT match and the paper's Stage 1 implementation.
        """
        self.state.critic_raw_response = action
        
        # Immediate Reward Computation using our engine (LAMBDA_FEEDBACK = 0, alpha=1.0, beta=1.0)
        self.state.critic_reward = CriticRewardEngine.compute_critic_reward(
            model_output=action,
            ground_truth_matched=self.state.ground_truth_matched,
            alpha=1.0,
            beta=1.0,
            gamma=0.0
        )

    def render(self, mode=None):
        return f"CriticEnv [Reward: {self.state.critic_reward}] - Q: {self.state.question}"


class CriticGRPOEnvBatch:
    """Batch environment manager for Pure Offline Critic Environments."""
    
    def __init__(
        self, 
        env_idx_list: List[int], 
        env_indices: List[int],
        rollout_idx_list: List[int], 
        samples: int, 
        max_turns: int, 
        config: dict, 
        mode: str = "train", 
        *, 
        env_workers: List = None
    ):
        self.mode = mode
        self.env_list = []
        
        # Load offline data (Simulated loader placeholder)
        # You will replace `load_critic_offline_data` with your actual dataloader/HuggingFace dataset logic
        # format: [{"question": "...", "plan": "...", "gt_answer": "...", "is_plan_correct": True/False}]
        # self.problem_list = load_critic_offline_data(env_indices, mode=mode)
        
        # ----- TEMPORARY MOCK LOADER FOR PIPELINE VALIDATION -----
        self.problem_list = []
        for _ in env_indices:
            self.problem_list.append({
                "question": "Mock Question?",
                "plan": "Mock offline plan...",
                "ground_truth": "Mock Answer",
                "is_plan_correct": True # Randomly mix True/False based on your dataset
            })
        # ---------------------------------------------------------
            
        if mode == "validate":
            rollout_idx_list = range(len(self.problem_list) * samples)
            samples = 1

        for i, problem in enumerate(self.problem_list):
            state = CriticEnvState(
                question=problem["question"],
                ground_truth_answer=problem["ground_truth"],
                planner_plan=problem["plan"],
                ground_truth_matched=problem["is_plan_correct"]
            )
            
            # G=8 samples means identical state replicated `samples` times to collect G trajectories
            for s in range(samples):
                env = CriticGRPOEnv(
                    env_idx=i, 
                    rollout_idx=rollout_idx_list[i * samples + s], 
                    max_turns=max_turns, 
                    config=None
                )
                env.state = copy.deepcopy(state)
                self.env_list.append(env)
