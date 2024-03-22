import logging

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.annotations import override
from ray.rllib.utils.tf_ops import one_hot
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from gym.spaces import Discrete, MultiDiscrete, Tuple
import numpy as np
from pyrlprob.utils import update
from pyrlprob.models.layers import *

tf1, tf, tfv = try_import_tf()
logger = logging.getLogger(__name__)


class ActorCriticModel(RecurrentNetwork):
    """General Actor-Critic model with different networks for the policy (actor) and value function (critic), 
    and possibly taking as input different observations for the actor and critic.
    The actor is an MLP, possibly preceded by a CNN, and possibly followed by an LSTM or GTrXL, 
    that outputs the logits (and, possibly, the sigmas). 
    The critic is an MLP, possibly preceded by a CNN, and possibly followed by an LSTM or GTrXL, 
    that outputs the value function estimate. 
    If the LSTM is used, the previous action and reward can be concatenated to the observation 
    before being sent to the LSTM.
    If instead a GTrXL is used, the previous n actions and rewards can be concatenated to the observation 
    before being sent to the GTrXL.

    Args:
        custom_model_kwargs:
            policy_net (dict): The parameters of the policy network.
                conv_filters (list): The number and type of filters of the convolutional layers. 
                    List of [out_channels, kernel, stride] for each filter.
                conv_activation (str): The activation function of the convolutional layers.

                mlp_hiddens (list): The sizes of the hidden layers of the MLP.
                mlp_activation (str): The activation function of the MLP.

                use_lstm (bool): Whether to use an LSTM layer after the MLP.
                lstm_cell_size (int): The size of the LSTM cell.
                lstm_use_prev_action (bool): Whether to use the previous action as input to the LSTM cell.
                lstm_use_prev_reward (bool): Whether to use the previous reward as input to the LSTM cell.

                use_gtrxl (bool): Whether to use a GTrXL layer after the MLP.
                num_transformer_units (int): The number of transformer units.
                attention_dim (int): The dimension of the transformer units.
                num_heads (int): The number of heads in the attention.
                head_dim (int): The dimension of each head.
                memory_inference (int): The number of previous outputs to use as memory input to each layer.
                position_wise_mlp_dim (int): The dimension of the position-wise MLP.
                init_gru_gate_bias (float): The initial bias of the GRU gate.
                use_n_prev_actions (int): The number of previous actions to use as input to the GTrXL.
                use_n_prev_rewards (int): The number of previous rewards to use as input to the GTrXL.

            vf_net (dict): The parameters of the value function network. 
                If any of the parameters is not specified, they are taken from the policy network.
                conv_filters (list): The number and type of filters of the convolutional layers. 
                    List of [out_channels, kernel, stride] for each filter.
                conv_activation (str): The activation function of the convolutional layers.

                mlp_hiddens (list): The sizes of the hidden layers of the MLP.
                mlp_activation (str): The activation function of the MLP.

                use_lstm (bool): Whether to use an LSTM layer after the MLP.
                lstm_cell_size (int): The size of the LSTM cell.
                lstm_use_prev_action (bool): Whether to use the previous action as input to the LSTM cell.
                lstm_use_prev_reward (bool): Whether to use the previous reward as input to the LSTM cell.

                use_gtrxl (bool): Whether to use a GTrXL layer after the MLP.
                num_transformer_units (int): The number of transformer units.
                attention_dim (int): The dimension of the transformer units.
                num_heads (int): The number of heads in the attention.
                head_dim (int): The dimension of each head.
                memory_inference (int): The number of previous outputs to use as memory input to each layer.
                position_wise_mlp_dim (int): The dimension of the position-wise MLP.
                init_gru_gate_bias (float): The initial bias of the GRU gate.
                use_n_prev_actions (int): The number of previous actions to use as input to the GTrXL.
                use_n_prev_rewards (int): The number of previous rewards to use as input to the GTrXL.
            
            diff_obs (bool): Whether the policy and value function networks have different observations. In this case, 
                the observation is a Tuple of (policy_obs, value_obs).
            
        model_config:
            free_log_std (bool): Whether to generate free-floating bias variables for the second half of the policy outputs.
            max_seq_len (int): The maximum length of the input observation sequence for the LSTM / GTrXL.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **custom_model_kwargs
    ):
        # Store the original observation space.
        self.original_space = obs_space.original_space if \
            hasattr(obs_space, "original_space") else obs_space
        
        super(ActorCriticModel, self).__init__(
            self.original_space, action_space, num_outputs, model_config, name
        )

        if logger.isEnabledFor(logging.INFO):
            print("custom_model_kwargs: {}".format(custom_model_kwargs))

        # Actor and Critic network parameters
        self.policy_net_config = custom_model_kwargs.get("policy_net", {})
        self.vf_net_config = custom_model_kwargs.get("policy_net", {})
        self.vf_net_config = update(self.vf_net_config, custom_model_kwargs.get("vf_net", {}))

        # Lstm or GTrXL
        self.actor_use_lstm = self.policy_net_config.get("use_lstm", False)
        if self.actor_use_lstm:
            self.actor_rec_size = self.policy_net_config.get("lstm_cell_size", 64)
            self.actor_lstm_use_prev_action = self.policy_net_config.get("lstm_use_prev_action", False)
            self.actor_lstm_use_prev_reward = self.policy_net_config.get("lstm_use_prev_reward", False)
        self.actor_use_gtrxl = self.policy_net_config.get("use_gtrxl", False)
        if self.actor_use_gtrxl:
            self.actor_rec_size = self.policy_net_config.get("attention_dim", 64)
            self.actor_num_transformer_units = self.policy_net_config.get("num_transformer_units", 1)
            self.actor_attention_dim = self.policy_net_config.get("attention_dim", 64)
            self.actor_use_n_prev_actions = self.policy_net_config.get("use_n_prev_actions", 0)
            self.actor_use_n_prev_rewards = self.policy_net_config.get("use_n_prev_rewards", 0)
            self.actor_memory_inference = self.policy_net_config.get("memory_inference", 16)
            self.memory_training = self.actor_memory_inference
        self.critic_use_lstm = self.vf_net_config.get("use_lstm", False)
        if self.critic_use_lstm:
            self.critic_rec_size = self.vf_net_config.get("lstm_cell_size", 64)
            self.critic_lstm_use_prev_action = self.vf_net_config.get("lstm_use_prev_action", False)
            self.critic_lstm_use_prev_reward = self.vf_net_config.get("lstm_use_prev_reward", False)
        self.critic_use_gtrxl = self.vf_net_config.get("use_gtrxl", False)
        if self.critic_use_gtrxl:
            self.critic_rec_size = self.vf_net_config.get("attention_dim", 64)
            self.critic_num_transformer_units = self.vf_net_config.get("num_transformer_units", 1)
            self.critic_attention_dim = self.vf_net_config.get("attention_dim", 64)
            self.critic_use_n_prev_actions = self.vf_net_config.get("use_n_prev_actions", 0)
            self.critic_use_n_prev_rewards = self.vf_net_config.get("use_n_prev_rewards", 0)
            self.critic_memory_inference = self.vf_net_config.get("memory_inference", 16)
            self.memory_training = self.critic_memory_inference
        
        if self.actor_use_gtrxl and self.critic_use_gtrxl:
            if self.actor_use_n_prev_actions != self.critic_use_n_prev_actions:
                if self.actor_use_n_prev_actions > 0 and self.critic_use_n_prev_actions > 0:
                    raise ValueError("The number of previous actions of the actor and critic GTrXL must be the same!")
            if self.actor_use_n_prev_rewards != self.critic_use_n_prev_rewards:
                if self.actor_use_n_prev_rewards > 0 and self.critic_use_n_prev_rewards > 0:
                    raise ValueError("The number of previous rewards of the actor and critic GTrXL must be the same!")
            if self.actor_memory_inference != self.critic_memory_inference:
                raise ValueError("The memory inference of the actor and critic GTrXL must be the same!")
            
        # Others
        self.diff_obs = custom_model_kwargs.get("diff_obs", False)
        self.max_seq_len = model_config.get("max_seq_len", 16)
        
        self.policy_net_config["max_seq_len"] = self.max_seq_len
        self.vf_net_config["max_seq_len"] = self.max_seq_len
        self.vf_net_config["free_log_std"] = False

        # Number of outputs
        self.num_outputs = num_outputs
        self.num_outputs_actor = num_outputs
        self.num_outputs_critic = 1

        # Number of actions
        self.action_dim = compute_action_dim(action_space)

        # Different observations
        if self.diff_obs:
            assert isinstance(self.original_space, (Tuple)), \
            "`obs_space` must be Tuple!"

            self.policy_obs_space = self.original_space[0]
            self.vf_obs_space = self.original_space[1]
        else:
            self.policy_obs_space = self.original_space
            self.vf_obs_space = self.original_space
        
        # If the policy and value function obs spaces are not a Tuple, make them a Tuple.
        if not isinstance(self.policy_obs_space, Tuple):
            self.policy_obs_space = Tuple([self.policy_obs_space])
        if not isinstance(self.vf_obs_space, Tuple):
            self.vf_obs_space = Tuple([self.vf_obs_space])
        
        # Build the actor network
        self.actor_cnns, self.actor_one_hot, self.actor_flatten, self.actor_mlp, \
            self.actor_output = conv_mlp_rec_model(
                self.policy_net_config, self.policy_obs_space, action_space, self.num_outputs_actor)
        
        # Build the critic network
        self.critic_cnns, self.critic_one_hot, self.critic_flatten, self.critic_mlp, \
            self.critic_output = conv_mlp_rec_model(
                self.vf_net_config, self.vf_obs_space, action_space, self.num_outputs_critic)
        
        # Trajectory view requirements
        self.view_requirements["obs"].space = self.original_space

        if self.actor_use_lstm or self.critic_use_lstm:

            use_prev_action = False
            use_prev_reward = False

            if self.actor_use_lstm:
                use_prev_action = use_prev_action or self.actor_lstm_use_prev_action
                use_prev_reward = use_prev_reward or self.actor_lstm_use_prev_reward
            
            if self.critic_use_lstm:
                use_prev_action = use_prev_action or self.critic_lstm_use_prev_action
                use_prev_reward = use_prev_reward or self.critic_lstm_use_prev_reward

            setup_trajectory_view_lstm(use_prev_action, use_prev_reward, action_space, self.view_requirements)
        
        if self.actor_use_gtrxl or self.critic_use_gtrxl:

            num_transformer_units_list = []
            attention_dim_list = []
            memory_inference_list = []
            use_n_prev_actions = 0
            use_n_prev_rewards = 0

            if self.actor_use_gtrxl:
                num_transformer_units_list.append(self.actor_num_transformer_units)
                attention_dim_list.append(self.actor_attention_dim)
                memory_inference_list.append(self.actor_memory_inference)
                use_n_prev_actions = max(use_n_prev_actions, self.actor_use_n_prev_actions)
                use_n_prev_rewards = max(use_n_prev_rewards, self.actor_use_n_prev_rewards)

            if self.critic_use_gtrxl:
                num_transformer_units_list.append(self.critic_num_transformer_units)
                attention_dim_list.append(self.critic_attention_dim)
                memory_inference_list.append(self.critic_memory_inference)
                use_n_prev_actions = max(use_n_prev_actions, self.critic_use_n_prev_actions)
                use_n_prev_rewards = max(use_n_prev_rewards, self.critic_use_n_prev_rewards)
            
            setup_trajectory_view_gtrxl(num_transformer_units_list, attention_dim_list, 
                memory_inference_list, self.max_seq_len,
                use_n_prev_actions, use_n_prev_rewards, 
                action_space, self.view_requirements)


    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        
        # Number of outputs
        self.num_outputs = self.num_outputs_actor
        
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict[SampleBatch.OBS],
                                                self.original_space, "tf")
        
        # Divide actor and critic observations
        if self.diff_obs:
            actor_obs = orig_obs[0]
            critic_obs = orig_obs[1]
        else:
            actor_obs = orig_obs
            critic_obs = orig_obs
        
        # If they are not iterables, make them iterables
        if not isinstance(actor_obs, (list, tuple)):
            actor_obs = [actor_obs]
        if not isinstance(critic_obs, (list, tuple)):
            critic_obs = [critic_obs]

        ## ACTOR FORWARD PASS ##
        
        # Push actor observations through the CNNs.
        actor_outs = []
        for i, component in enumerate(actor_obs):
            if i in self.actor_cnns:
                cnn_out = self.actor_cnns[i](component)
                actor_outs.append(cnn_out)
            elif i in self.actor_one_hot:
                if component.dtype in [tf.int32, tf.int64, tf.uint8]:
                    actor_outs.append(
                        one_hot(component, self.policy_obs_space.spaces[i]))
                else:
                    actor_outs.append(component)
            else:
                actor_outs.append(tf.reshape(component, [-1, self.actor_flatten[i]]))
        
        # Concat all outputs and the non-image inputs.
        actor_out = tf.concat(actor_outs, axis=1)

        # Push through MLP
        actor_out = self.actor_mlp(actor_out)

        # Push through (optional) recurrent layer.
        actor_state = []
        if self.actor_use_lstm or self.actor_use_gtrxl:

            # Extract the previous state from the state tensor.
            if self.actor_use_lstm:
                actor_state = state[0:2]
            elif self.actor_use_gtrxl:
                if len(state) >= self.actor_num_transformer_units:
                    actor_state = state[0:self.actor_num_transformer_units]
                else:
                    actor_state = state

            # If the rec layer is an LSTM, add the previous action and reward to the input.
            prev_a_r_actor = []
            if self.actor_use_lstm:
                if self.actor_lstm_use_prev_action:
                    prev_a = input_dict[SampleBatch.PREV_ACTIONS]
                    if isinstance(self.action_space, (Discrete, MultiDiscrete)):
                        prev_a = one_hot(prev_a, self.action_space)
                    prev_a_r_actor.append(
                        tf.reshape(tf.cast(prev_a, tf.float32), [-1, self.action_dim]))
                if self.actor_lstm_use_prev_reward:
                    prev_a_r_actor.append(
                        tf.reshape(
                            tf.cast(input_dict[SampleBatch.PREV_REWARDS], tf.float32),
                            [-1, 1]))
            
            # If the rec layer is a GTrXL, add the previous n actions and n rewards to the input.
            if self.actor_use_gtrxl:
                if self.actor_use_n_prev_actions:
                    if isinstance(self.action_space, Discrete):
                        for i in range(self.actor_use_n_prev_actions):
                            prev_a_r_actor.append(
                                one_hot(input_dict[SampleBatch.PREV_ACTIONS][:, i],
                                        self.action_space))
                    elif isinstance(self.action_space, MultiDiscrete):
                        for i in range(
                                self.actor_use_n_prev_actions,
                                step=self.action_space.shape[0]):
                            prev_a_r_actor.append(
                                one_hot(
                                    tf.cast(
                                        input_dict[SampleBatch.PREV_ACTIONS]
                                        [:, i:i + self.action_space.shape[0]],
                                        tf.float32), self.action_space))
                    else:
                        prev_a_r_actor.append(
                            tf.reshape(
                                tf.cast(input_dict[SampleBatch.PREV_ACTIONS],
                                        tf.float32),
                                [-1, self.actor_use_n_prev_actions * self.action_dim]))
                if self.actor_use_n_prev_rewards:
                    prev_a_r_actor.append(
                        tf.reshape(
                            tf.cast(input_dict[SampleBatch.PREV_REWARDS], tf.float32),
                            [-1, self.actor_use_n_prev_rewards]))
            
            if len(prev_a_r_actor) > 0:
                actor_out = tf.concat([actor_out] + prev_a_r_actor, axis=1)

            if self.actor_use_lstm:
                # Recurrent model
                self.rnn_model = self.actor_output

                # Input to the rec model
                input_dict["obs_flat"] = actor_out
                
                # Output
                actor_out, actor_state = super().forward(input_dict, actor_state, seq_lens)

            elif self.actor_use_gtrxl:
                # Recurrent model
                self.gtrxl_model = self.actor_output

                # Attention dim
                self.attention_dim = self.actor_attention_dim

                # Input to the rec model
                input_dict["obs_flat"] = input_dict["obs"] = actor_out

                # Output
                actor_out, actor_state = self.forward_gtrxl(input_dict, actor_state, seq_lens)
        else:
            # Push through final layer
            actor_out = self.actor_output(actor_out)

        ## CRITIC FORWARD PASS ##

        # Push critic observations through the CNNs.
        critic_outs = []
        for i, component in enumerate(critic_obs):
            if i in self.critic_cnns:
                cnn_out = self.critic_cnns[i](component)
                critic_outs.append(cnn_out)
            elif i in self.critic_one_hot:
                if component.dtype in [tf.int32, tf.int64, tf.uint8]:
                    critic_outs.append(
                        one_hot(component, self.vf_obs_space.spaces[i]))
                else:
                    critic_outs.append(component)
            else:
                critic_outs.append(tf.reshape(component, [-1, self.critic_flatten[i]]))
        
        # Concat all outputs and the non-image inputs.
        critic_out = tf.concat(critic_outs, axis=1)

        # Push through MLP
        critic_out = self.critic_mlp(critic_out)

        # Push through (optional) recurrent layer.
        critic_state = []
        if self.critic_use_lstm or self.critic_use_gtrxl:

            # Extract the previous state from the state tensor.
            if self.critic_use_lstm:
                if self.actor_use_lstm:
                    critic_state = state[2:]
                elif self.actor_use_gtrxl:
                    assert False, "Actor and Critic must be both lstms or gtrxls!"
                else:
                    critic_state = state[0:2]
            elif self.critic_use_gtrxl:
                if self.actor_use_lstm:
                    assert False, "Actor and Critic must be both lstms or gtrxls!"
                elif self.actor_use_gtrxl:
                    if len(state) >= (self.actor_num_transformer_units+1):
                        critic_state = state[self.actor_num_transformer_units:]
                    else:
                        critic_state = state
                else:
                    critic_state = state

            # If the rec layer is an LSTM, add the previous action and reward to the input.
            prev_a_r_critic = []
            if self.critic_use_lstm:
                if self.critic_lstm_use_prev_action:
                    prev_a = input_dict[SampleBatch.PREV_ACTIONS]
                    if isinstance(self.action_space, (Discrete, MultiDiscrete)):
                        prev_a = one_hot(prev_a, self.action_space)
                    prev_a_r_critic.append(
                        tf.reshape(tf.cast(prev_a, tf.float32), [-1, self.action_dim]))
                if self.critic_lstm_use_prev_reward:
                    prev_a_r_critic.append(
                        tf.reshape(
                            tf.cast(input_dict[SampleBatch.PREV_REWARDS], tf.float32),
                            [-1, 1]))
            
            # If the rec layer is a GTrXL, add the previous n actions and n rewards to the input.
            if self.critic_use_gtrxl:
                if self.critic_use_n_prev_actions:
                    if isinstance(self.action_space, Discrete):
                        for i in range(self.critic_use_n_prev_actions):
                            prev_a_r_critic.append(
                                one_hot(input_dict[SampleBatch.PREV_ACTIONS][:, i],
                                        self.action_space))
                    elif isinstance(self.action_space, MultiDiscrete):
                        for i in range(
                                self.critic_use_n_prev_actions,
                                step=self.action_space.shape[0]):
                            prev_a_r_critic.append(
                                one_hot(
                                    tf.cast(
                                        input_dict[SampleBatch.PREV_ACTIONS]
                                        [:, i:i + self.action_space.shape[0]],
                                        tf.float32), self.action_space))
                    else:
                        prev_a_r_critic.append(
                            tf.reshape(
                                tf.cast(input_dict[SampleBatch.PREV_ACTIONS],
                                        tf.float32),
                                [-1, self.critic_use_n_prev_actions * self.action_dim]))
                if self.critic_use_n_prev_rewards:
                    prev_a_r_critic.append(
                        tf.reshape(
                            tf.cast(input_dict[SampleBatch.PREV_REWARDS], tf.float32),
                            [-1, self.critic_use_n_prev_rewards]))
            
            if len(prev_a_r_critic) > 0:
                critic_out = tf.concat([critic_out] + prev_a_r_critic, axis=1)
            
            if self.critic_use_lstm:
                # Recurrent model
                self.rnn_model = self.critic_output

                # Input to the rec model
                input_dict["obs_flat"] = critic_out

                # Number of outputs
                self.num_outputs = self.num_outputs_critic
                
                # Output
                self._critic_out, critic_state = super().forward(input_dict, critic_state, seq_lens)
            
            elif self.critic_use_gtrxl:
                # Recurrent model
                self.gtrxl_model = self.critic_output

                # Attention dim
                self.attention_dim = self.critic_attention_dim

                # Input to the rec model
                input_dict["obs_flat"] = input_dict["obs"] = critic_out

                # Number of outputs
                self.num_outputs = self.num_outputs_critic

                # Output
                self._critic_out, critic_state = self.forward_gtrxl(input_dict, critic_state, seq_lens)
        else:
            # Push through final layer
            self._critic_out = self.critic_output(critic_out)

        # State outputs
        if len(actor_state) == 0 and len(critic_state) == 0:
            state_out = state
        else:
            state_out = actor_state + critic_state

        return actor_out, state_out


    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):

        model_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        
        return model_out, [h, c]
    

    def forward_gtrxl(self, input_dict, state, seq_lens):
        
        # Add the time dim to observations.
        B = tf.shape(seq_lens)[0]
        observations = input_dict[SampleBatch.OBS]

        shape = tf.shape(observations)
        T = shape[0] // B
        observations = tf.reshape(observations,
                                  tf.concat([[-1, T], shape[1:]], axis=0))

        all_out = self.gtrxl_model([observations] + state)

        out = tf.reshape(all_out[0], [-1, self.num_outputs])
        memory_outs = all_out[1:]

        return out, [
            tf.reshape(m, [-1, self.attention_dim]) for m in memory_outs
        ]


    # def forward_rnn(self, use_lstm, use_gtrxl, model, model_dim, obs, state, seq_lens):

    #     # Add the time dim to observations.
    #     B = tf.shape(seq_lens)[0]
    #     observations = obs
    #     shape = tf.shape(observations)
    #     T = shape[0] // B

    #     if use_lstm:
    #         new_batch_size = shape[0] // T
    #     elif use_gtrxl:
    #         new_batch_size = -1

    #     # Flatten the (B, T, ...) observations to (B * T, ...).
    #     observations = tf.reshape(observations,
    #                                 tf.concat([[new_batch_size, T], shape[1:]], axis=0))

    #     # Push through the recurrent model
    #     if use_lstm:
    #         all_out = model([observations, seq_lens] + state)
    #     elif use_gtrxl:
    #         all_out = model([observations] + state)

    #     # Output
    #     out = tf.reshape(all_out[0], [-1, model_dim])

    #     # Extract the state
    #     new_state = all_out[1:]

    #     return out, [
    #         tf.reshape(m, [-1, model_dim]) for m in new_state
    #     ]
    

    @override(ModelV2)
    def get_initial_state(self):
        if self.actor_use_lstm:
            actor_state = [np.zeros(self.actor_rec_size, np.float32), \
                           np.zeros(self.actor_rec_size, np.float32)]
        else:
            actor_state = []
        if self.critic_use_lstm:
            critic_state = [np.zeros(self.critic_rec_size, np.float32), \
                            np.zeros(self.critic_rec_size, np.float32)]
        else:
            critic_state = []

        return actor_state + critic_state


    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._critic_out, [-1])
