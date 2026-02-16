from typing import Any, ClassVar, Dict, Type, TypeVar

import numpy as np
import torch as th

from torch.nn import functional as F

from  stable_baselines3.common.policies import BasePolicy
from  stable_baselines3.common.type_aliases import MaybeCallback, Schedule
from  stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from  stable_baselines3.dqn.dqn import DQN

SelfDQN = TypeVar("SelfDQN", bound="DQN")


class CustomDDQN(DQN):

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQNPolicy

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        
        losses = []
        for _ in range(gradient_steps):

            # Beta-tracking and sampling
            start_beta = .4
            end_beta = 1.
            beta_increment = end_beta - start_beta
            beta = start_beta + (self.num_timesteps / self._total_timesteps) * beta_increment
            replay_data, sample_weights, sample_idxs = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env, beta=beta)
            sample_weights = th.from_numpy(sample_weights).to(self.device)
                
            with th.no_grad():
                # Get the action from the Q-network
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)

                # Get the corresponding Q-value from the target network
                next_q_values = self.q_net_target(replay_data.next_observations).gather(1, next_actions)

                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)

                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = th.mean(F.smooth_l1_loss(current_q_values, target_q_values, reduction="none").squeeze() * sample_weights)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Update prioirities
            td_errors = current_q_values - target_q_values
            td_errors = np.abs(td_errors.detach().cpu().numpy().reshape(-1)) + 1e-6        
            self.replay_buffer.update_priorities(idxes=sample_idxs, td_errors=td_errors)

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DDQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )