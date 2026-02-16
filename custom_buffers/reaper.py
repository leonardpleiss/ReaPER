import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from typing import Optional, Union, List, Dict, Any, Tuple
from gymnasium import spaces
import torch as th

class ReaPER(ReplayBuffer):

    """
    A Replay Buffer with Reliability-Adjusted Prioritized Experience Replay (ReaPER).

    This buffer adjusts sampling probabilities based on both TD-error and
    reliability of transitions (i.e., how reliable they are relative to others
    in the same episode).
    """

    def __init__(        
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = False,
            alpha: float = .3,
            omega: float = .3,
    ) -> None:
        
        """
        Initialize the ReaPER replay buffer.

        Args:
            buffer_size (int): Maximum number of transitions to store in the buffer.
            observation_space (spaces.Space): Observation space of the environment.
            action_space (spaces.Space): Action space of the environment.
            device (Union[th.device, str]): PyTorch device or string identifier.
            n_envs (int): Number of parallel environments (currently must be 1).
            optimize_memory_usage (bool): If True, optimize memory by storing only necessary data.
            handle_timeout_termination (bool): If True, handle timeout terminations separately.
            alpha (float): Exponent applied to TD-error in computing sampling weights.
            omega (float): Exponent applied to reliability in computing sampling weights.

        Raises:
            AssertionError: If `n_envs` is not 1, as multi-env support is not implemented.
        """
            
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)

        if n_envs != 1:
            raise ValueError("ReaPER currently supports only a single environment (n_envs=1).")

        self._alpha = np.float64(alpha)
        self._omega = np.float64(omega)
        print(alpha, omega)

        # Setup episode storage
        self._current_episode = np.arange(1, self.n_envs+1, dtype=int)
        self.episodes = np.zeros((self.buffer_size, self.n_envs), dtype=int)
        self.episodes_played = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        
        # Setup td error, td sum and td cumsum storage
        self._max_td = 1.
        self._max_sum_td = 1.
        self.last_done = np.zeros(self.n_envs,dtype=np.bool_)
        self._cum_td = np.zeros(self.n_envs,)
        self.cum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.sum_tds = np.zeros((self.buffer_size, self.n_envs))
        self.td_errors = np.zeros((self.buffer_size, self.n_envs))
        self.sampling_weights = np.zeros((self.buffer_size, self.n_envs))

        # Setup timestep storage
        self._current_timestep = np.ones(self.n_envs,)
        self.timesteps = np.zeros((self.buffer_size, self.n_envs))

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        """
        Add a new transition to the replay buffer and update internal TD-error and reliability metrics.

        Args:
            obs (np.ndarray): Observation at current timestep.
            next_obs (np.ndarray): Observation at next timestep.
            action (np.ndarray): Action taken at current timestep.
            reward (np.ndarray): Reward received after taking the action.
            done (np.ndarray): Done flags indicating episode termination.
            infos (List[Dict[str, Any]]): Additional information from the environment.

        Notes:
            - Initializes TD-errors and sampling weights with maximum values.
            - Updates cumulative and total TD-errors.
            - Computes final sampling weights when an episode finishes. 
        """

        # Set preliminary sums
        self.sum_tds[self.pos] = self._max_sum_td
        
        # Add transition information
        self.td_errors[self.pos] = self._max_td
        self.episodes[self.pos] = self._current_episode
        self.episodes_played[self.pos] = np.ones((self.n_envs,), dtype=bool)
        self.timesteps[self.pos] = self._current_timestep
        self.cum_tds[self.pos] = (~self.last_done * self.cum_tds[self.pos-1]) + self._max_td

        # Initiate sampling weights at max td
        self.sampling_weights[self.pos] = self.td_errors[self.pos] ** self._alpha

        # Compute actual sampling weights once episode is done
        if done.any():
            ep_done_row_idxes, ep_done_col_idxes = np.where((self.episodes==self._current_episode) & done)
            self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] = self.cum_tds[self.pos][ep_done_col_idxes]
            self.calculate_sampling_weights_for_finished_runs(ep_done_row_idxes, ep_done_col_idxes)

        # Update tracking variables
        self._current_timestep = 1 + (self._current_timestep * ~done)
        self._current_episode += self.n_envs * done
        self.last_done = done

        super().add(obs, next_obs, action, reward, done, infos)
    
    def sample(self,
               batch_size: int,
               beta: float = .5,
               env: Optional[VecNormalize] = None
    ) -> Tuple[ReplayBufferSamples, np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        """
        Sample a batch of transitions from the replay buffer using prioritized sampling.

        Args:
            batch_size (int): Number of transitions to sample.
            beta (float): Exponent used to calculate importance-sampling weights.
            env (Optional[VecNormalize]): Optional VecNormalize wrapper to normalize observations.

        Returns:
            Tuple[ReplayBufferSamples, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                - Encoded batch of sampled transitions.
                - Importance-sampling weights for each transition.
                - Tuple of indices (row_idxes, col_idxes) used to locate the samples.
        
        Raises:
            AssertionError: If `beta` is not greater than 0.
        """

        assert beta > 0

        num_transitions_gathered = (self.pos if not self.full else self.buffer_size) * self.n_envs

        # Get sample_idxes
        row_weights = np.cumsum(self.sampling_weights[:num_transitions_gathered, :].sum(axis=1))
        sampling_weight_sum = row_weights[-1]
        sample_vals = np.random.rand(batch_size) * sampling_weight_sum
        row_idxes = np.searchsorted(row_weights, sample_vals)
        col_idxes = np.zeros(shape=(batch_size,), dtype=np.int64)

        # Encode
        encoded_sample = super()._get_samples(row_idxes, env=env)

        # Get importance sampling weights
        sampling_weight_sum = row_weights[-1]
        sampling_probas_of_batch = self.sampling_weights[row_idxes, col_idxes] / sampling_weight_sum
        p_min = self.sampling_weights[:num_transitions_gathered].min() / sampling_weight_sum
        max_weight = (p_min * num_transitions_gathered) ** (-beta)
        IS_weights = (sampling_probas_of_batch * num_transitions_gathered) ** (-beta) / max_weight

        return ReplayBufferSamples(*tuple(map(self.to_torch, encoded_sample))), IS_weights, (row_idxes, col_idxes)
    
    def update_priorities(
            self,
            idxes: Tuple[np.ndarray, np.ndarray],
            td_errors: np.ndarray,
        ) -> None:

        """
        Update the TD-error priorities and sampling weights for relevant buffer indices.

        Args:
            idxes (Tuple[np.ndarray, np.ndarray]): Tuple of row and column indices for the buffer.
            td_errors (np.ndarray): New TD-errors to assign.

        Raises:
            AssertionError: If `idxes` is not a tuple.
        
        Notes:
            - Applies changes only to unique transitions.
            - Updates cumulative and total TD-errors accordingly.
            - Recalculates sampling weights based on updated TD-errors and reliability.
        """

        assert isinstance(idxes, tuple), "Idxes is expected to consist of a tuple, (row_idxes, col_idxes)"
        row_idxes, col_idxes = idxes

        # Filter unique transitions
        _, unique_idx = np.unique(np.array(idxes), axis=1, return_index=True)
        td_errors = td_errors[unique_idx]
        row_idxes = row_idxes[unique_idx]
        col_idxes = col_idxes[unique_idx]

        # Mask relevant variables
        episodes_to_update = self.episodes[row_idxes, col_idxes]
        timesteps_to_update = self.timesteps[row_idxes, col_idxes]
        deltas = self.td_errors[row_idxes, col_idxes] - td_errors

        # Obtaining change masks
        played_and_change_mask = np.isin(self.episodes, episodes_to_update)

        # Updating TD errors
        self.td_errors[row_idxes, col_idxes] -= deltas

        # Updating sums
        sum_mask = self.episodes[played_and_change_mask][:, None] == episodes_to_update
        sum_deltas = deltas * (episodes_to_update != self._current_episode)
        self.sum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', sum_mask, sum_deltas[None, :])

        # Updating cumulative sums
        cum_mask = sum_mask & (self.timesteps[played_and_change_mask][:, None] >= timesteps_to_update)
        self.cum_tds[played_and_change_mask] -= np.einsum('ij,ij->i', cum_mask, deltas[None, :])
        
        # Overwriting max TD
        self._max_td = max(self._max_td, np.max(td_errors))

        # Update sampling weights
        self.update_sampling_weights(played_and_change_mask)


    def calculate_sampling_weights_for_finished_runs(
            self,
            ep_done_row_idxes:np.ndarray,
            ep_done_col_idxes:np.ndarray,
        ) -> None:

        """
        Compute and assign final sampling weights for completed episodes.

        Args:
            ep_done_row_idxes (np.ndarray): Row indices of completed episodes.
            ep_done_col_idxes (np.ndarray): Column indices of completed episodes.

        Notes:
            - Sampling weights are calculated using both TD-error and episode reliability.
            - Reliability is based on the proportion of TD-error occurring after a given transition.
        """
        
        subsequent_tds = self.sum_tds[ep_done_row_idxes, ep_done_col_idxes] - self.cum_tds[ep_done_row_idxes, ep_done_col_idxes]

        # Update max sum
        self._max_sum_td = max(self._max_sum_td, np.max(self.sum_tds))

        # Compute new sampling weights
        reliability = 1 - (subsequent_tds / self.sum_tds[ep_done_row_idxes, ep_done_col_idxes])
        td_errors = self.td_errors[ep_done_row_idxes, ep_done_col_idxes]
        self.sampling_weights[ep_done_row_idxes, ep_done_col_idxes] = td_errors**self._alpha * reliability**self._omega


    def update_sampling_weights(
            self,
            played_mask: Optional[np.ndarray] = None,
        ) -> None:

        """
        Recompute sampling weights for specified transitions based on updated TD-errors and reliability.

        Args:
            played_mask (np.ndarray, optional): Boolean mask identifying which transitions to update.
        
        Notes:
            - If `played_mask` is not provided, all played transitions are considered.
        """

        # Calculate subsequent TDs and update maximum subsequent TD
        subsequent_tds = self.sum_tds - self.cum_tds
        self._max_sum_td = np.max(self.sum_tds)

        # Mask relevant variables
        td_errors = self.td_errors[played_mask]

        # Compute reliability
        reliability = 1 - (subsequent_tds[played_mask] / self.sum_tds[played_mask])

        # Update sampling weights
        new_weights = (td_errors**self._alpha) * (reliability**self._omega)
        self.sampling_weights[played_mask] = new_weights



        
