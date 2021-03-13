import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rl_pg import PolicyAgent, TrainBatch, VanillaPolicyGradientLoss


class AACPolicyNet(nn.Module):
    def __init__(self, in_features: int, out_actions: int, **kw):
        """
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()
        # TODO:
        #  Implement a dual-head neural net to approximate both the
        #  policy and value. You can have a common base part, or not.
        # ====== YOUR CODE: ======
        self.fc1_actor = nn.Linear(in_features, 256)
        self.relu_1_actor = nn.ReLU()
        self.fc2_actor = nn.Linear(256, 128)
        self.relu_2_actor = nn.ReLU()
        self.fc3_actor = nn.Linear(128, out_actions)
        # self.relu_3_actor = nn.ReLU()
        # self.fc4_actor = nn.Linear(64,out_actions)

        self.fc1_critic = nn.Linear(in_features, 128)
        self.relu_1_critic = nn.ReLU()
        self.fc2_critic = nn.Linear(128, 64)
        self.relu_2_critic = nn.ReLU()
        self.fc3_critic = nn.Linear(64, 1)
        # self.relu_3_critic = nn.ReLU()
        # self.fc4_critic = nn.Linear(out_actions,1)
        # ========================

    def forward(self, x):
        """
        :param x: Batch of states, shape (N,O) where N is batch size and O
        is the observation dimension (features).
        :return: A tuple of action values (N,A) and state values (N,1) where
        A is is the number of possible actions.
        """
        # TODO:
        #  Implement the forward pass.
        #  calculate both the action scores (policy) and the value of the
        #  given state.
        # ====== YOUR CODE: ======
        # separate base
        # actor network
        action_scores = self.fc1_actor(x)
        action_scores = self.relu_1_actor(action_scores)
        action_scores = self.fc2_actor(action_scores)
        action_scores = self.relu_2_actor(action_scores)
        action_scores = self.fc3_actor(action_scores)

        # critic network
        state_values = self.fc1_critic(x)
        state_values = self.relu_1_critic(state_values)
        state_values = self.fc2_critic(state_values)
        state_values = self.relu_2_critic(state_values)
        state_values = self.fc3_critic(state_values)

        # ========================

        return action_scores, state_values

    @staticmethod
    def build_for_env(env: gym.Env, device='cpu', **kw):
        """
        Creates a A2cNet instance suitable for the given environment.
        :param env: The environment.
        :param kw: Extra hyperparameters.
        :return: An A2CPolicyNet instance.
        """
        # TODO: Implement according to docstring.
        # ====== YOUR CODE: ======
        net = AACPolicyNet(env.observation_space.shape[0], env.action_space.n)
        # ========================
        return net.to(device)


class AACPolicyAgent(PolicyAgent):

    def current_action_distribution(self) -> torch.Tensor:
        # TODO: Generate the distribution as described above.
        # ====== YOUR CODE: ======
        soft_max = nn.Softmax(dim=0)
        scores, _ = self.p_net(self.curr_state)
        actions_proba = soft_max(scores)
        # ========================
        return actions_proba


class AACPolicyGradientLoss(VanillaPolicyGradientLoss):
    def __init__(self, delta: float):
        """
        Initializes an AAC loss function.
        :param delta: Scalar factor to apply to state-value loss.
        """
        super().__init__()
        self.delta = delta

    def forward(self, batch: TrainBatch, model_output, **kw):

        # Get both outputs of the AAC model
        action_scores, state_values = model_output

        # TODO: Calculate the policy loss loss_p, state-value loss loss_v and
        #  advantage vector per state.
        #  Use the helper functions in this class and its base.
        # ====== YOUR CODE: ======
        loss_v = self._value_loss(batch, state_values)
        advantage = self._policy_weight(batch, state_values)

        loss_p = (-1/len(batch)) * self._policy_loss(batch, action_scores, advantage)
        # ========================

        loss_v *= self.delta
        loss_t = loss_p + loss_v
        return loss_t, dict(loss_p=loss_p.item(), loss_v=loss_v.item(),
                            adv_m=advantage.mean().item())

    def _policy_weight(self, batch: TrainBatch, state_values: torch.Tensor):
        # TODO:
        #  Calculate the weight term of the AAC policy gradient (advantage).
        #  Notice that we don't want to backprop errors from the policy
        #  loss into the state-value network.
        # ====== YOUR CODE: ======
        advantage = batch.q_vals.detach() - state_values.transpose(1, 0)
        advantage = advantage.view(-1)
        # ========================
        return advantage

    def _value_loss(self, batch: TrainBatch, state_values: torch.Tensor):
        # TODO: Calculate the state-value loss.
        # ====== YOUR CODE: ======
        loss_v = (1/len(batch)) * torch.sum((batch.q_vals - state_values.transpose(1, 0))**2)
        # ========================
        return loss_v