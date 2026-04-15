import logging
from pettingllms.multi_agent_env.base.agent import BaseAgent
from pettingllms.multi_agent_env.critic_grpo.critic_env import CriticEnvState

logger = logging.getLogger(__name__)

class CriticAgent(BaseAgent):
    """
    Critic Agent for Offline GRPO setup.
    This agent merely passes the pre-defined format prompt to the ExecutionEngine
    to invoke generation from the LoRA policy.
    """
    def __init__(self, name: str, config: dict, env_config: dict | None = None):
        super().__init__(name=name, config=config, env_config=env_config)

    def formulate_request(self, state: CriticEnvState) -> str:
        """
        Formulate prompt from the offline dataset item. The Critic expects 
        the question and the planner's plan to give a verdict/feedback.
        """
        prompt = (
            f"You are an expert Reviewer (Critic) for a multi-hop question answering system.\n\n"
            f"Question:\n{state.question}\n\n"
            f"Planner's Plan:\n{state.planner_plan}\n\n"
            f"Your task is to judge whether the plan will successfully lead to the correct answer without omissions or logical leaps.\n"
            f"1. Provide a detailed <feedback> component (even though we don't grade it heavily yet).\n"
            f"2. Conclude with a strict <verdict>ACCEPT</verdict> or <verdict>REJECT</verdict> decision.\n\n"
            f"Response:"
        )
        return prompt

    def update_state(self, state: CriticEnvState, action: str):
        # We don't necessarily update agent internal state here, 
        # as the env.step("critic", action) handles adding to trajectory/rewards.
        pass
