import logging

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.framework import try_import_tf
from typing import Dict, List
from ray.rllib.utils.typing import TensorType, List, ModelConfigDict
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import override
from ray.rllib.utils.tf_ops import one_hot
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.tf.layers import (
    GRUGate,
    RelativeMultiHeadAttention,
    SkipConnection,
)
from ray.rllib.models.tf.attention_net import PositionwiseFeedforward
import gym
from gym.spaces import Box, Discrete, MultiDiscrete
import numpy as np

tf1, tf, tfv = try_import_tf()
logger = logging.getLogger(__name__)


class FCModelforRNNs(TFModelV2):
    """Custom FC model to be used before an RNN/attention wrapper for policy gradient algorithms.
    The model is the same as the default FCModel, except that the last input observation is
    interpreted as the previous "estimated" reward and is directly concatenated
    to the output of the last hidden layer. This is done to allow the RNN/attention
    wrapper to learn a policy that is conditioned on the previous reward.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        super(FCModelforRNNs, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        hiddens = list(model_config.get("fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        activation = get_activation_fn(activation)
        
        # We are using obs_flat, so take the flattened shape as input, minus one (the reward)
        inputs = tf.keras.layers.Input(
            shape=(int(np.product(obs_space.shape) - 1),), name="observations"
        )
        # Last hidden layer output (before logits outputs).
        last_layer = inputs
        # The action distribution outputs.
        logits_out = None
        i = 1

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0),
            )(last_layer)
            i += 1

        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        if len(hiddens) > 0:
            last_layer = tf.keras.layers.Dense(
                hiddens[-1],
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0),
            )(last_layer)
        if num_outputs:
            logits_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01),
            )(last_layer)
        # Adjust num_outputs to be the number of nodes in the last layer.
        else:
            self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[
                -1
            ]

        last_vf_layer = None

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(last_vf_layer if last_vf_layer is not None else last_layer)

        self.base_model = tf.keras.Model(
            inputs, [(logits_out if logits_out is not None else last_layer), value_out]
        )

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> tuple[TensorType, List[TensorType]]:
        
        # Remove the last element from the observation
        input_obs_reduced = input_dict["obs_flat"][:, :-1]

        # Output of the fc model
        model_out, self._value_out = self.base_model(input_obs_reduced)

        # Last element of the observation (the reward)
        last_obs = tf.reshape(
            tf.cast(input_dict["obs_flat"][:, -1], tf.float32), [-1, 1]
        )

        # Previous reward
        prev_r = [last_obs]
        
        # New output
        model_out = tf.concat([model_out] + prev_r, axis=1)

        return model_out, state

    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])


class MLPplusLSTM(RecurrentNetwork):
    """LSTM-based model for meta-RL. The policy model is made up of an MLP encoder and an LSTM cell.
     The encoder is a simple feedforward network that preprocesses the observation before sending
     it to the LSTM cell, possible appending also the previous action and reward.
     The LSTM cell is a standard LSTM cell that outputs the logits.
     The value function estimate is computed either by a separate feedforward network that takes the
     same observation as input, or is an additional output of the LSTM cell.

     Args:
        custom_model_kwargs:
            lstm_cell_size (int): The size of the LSTM cell.
            encoder_hiddens (list): The sizes of the hidden layers of the encoder.
            encoder_activation (str): The activation function of the encoder.
            vf_hiddens (list): The sizes of the hidden layers of the value function.
            vf_activation (str): The activation function of the value function.
            lstm_use_prev_action (bool): Whether to use the previous action as input to the LSTM cell.
            lstm_use_prev_reward (bool): Whether to use the previous reward as input to the LSTM cell.
        model_config:
            max_seq_len (int): The maximum length of the input observation sequence.
            vf_share_layers (bool): Whether to share the layers of the value function with the policy.
            free_log_std (bool): Whether to generate free-floating bias variables for the second half of the model outputs.
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
        super(MLPplusLSTM, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        
        if logger.isEnabledFor(logging.INFO):
            print("custom_model_kwargs: {}".format(custom_model_kwargs))

        # Encoder parameters
        self.encoder_hiddens = list(custom_model_kwargs.get("encoder_hiddens", []))
        self.encoder_activation = custom_model_kwargs.get("encoder_activation")
        encoder_activation = get_activation_fn(self.encoder_activation)

        # LSTM parameters
        self.cell_size = custom_model_kwargs.get("lstm_cell_size")
        self.use_prev_action = custom_model_kwargs.get("lstm_use_prev_action")
        self.use_prev_reward = custom_model_kwargs.get("lstm_use_prev_reward")
        self.max_seq_len = model_config.get("max_seq_len")
        free_log_std = model_config.get("free_log_std")

        # Value function parameters
        self.vf_share_layers = model_config.get("vf_share_layers")
        if not self.vf_share_layers:
            self.vf_hiddens = list(custom_model_kwargs.get("vf_hiddens", []))
            self.vf_activation = custom_model_kwargs.get("vf_activation")
            vf_activation = get_activation_fn(self.vf_activation)
        
        # Number of outputs (logit dim)
        self.num_outputs = num_outputs

        # Generate free-floating bias variables for the second half of
        # the outputs, if free_log_std = True
        if free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2
            self.log_std_var = tf.Variable(
                [0.0] * num_outputs, dtype=tf.float32, name="log_std"
            )
        
        # Action dimension
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
        elif isinstance(action_space, MultiDiscrete):
            self.action_dim = np.sum(action_space.nvec)
        elif action_space.shape is not None:
            self.action_dim = int(np.product(action_space.shape))
        else:
            self.action_dim = int(len(action_space))

        ## DEFINE ENCODER LAYERS ##
            
        # Input layer
        input_layer_encoder = tf.keras.layers.Input(
            shape=(int(np.product(obs_space.shape)), ), name="observations"
        )

        # Encoder hidden layers
        last_layer_encoder = input_layer_encoder
        i = 1
        for size in self.encoder_hiddens:
            last_layer_encoder = tf.keras.layers.Dense(
                size,
                name="fc_encoder_{}".format(i),
                activation=encoder_activation,
                kernel_initializer=normc_initializer(1.0),
            )(last_layer_encoder)
            i += 1

        # Number of outputs (logit dim)
        self.encoder_out_size = self.encoder_hiddens[-1]

        ## DEFINE LSTM LAYERS ##

        # Add prev-action/reward nodes to input to LSTM.
        self.num_lstm_inputs = self.encoder_out_size
        if self.use_prev_action:
            self.num_lstm_inputs += self.action_dim
        if self.use_prev_reward:
            self.num_lstm_inputs += 1

        # Input layer
        input_layer_lstm = tf.keras.layers.Input(
            shape=(None, self.num_lstm_inputs), name="inputs_lstm"
        )
        state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # LSTM cell
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            self.cell_size, return_sequences=True, return_state=True, name="lstm"
        )(
            inputs=input_layer_lstm,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c],
        )

        # Output layer
        logits = tf.keras.layers.Dense(
            num_outputs, activation=tf.keras.activations.linear, name="logits"
        )(lstm_out)

        # Concat the log std vars to the end of the state-dependent means.
        if free_log_std:

            def tiled_log_std(x):
                return tf.tile(tf.expand_dims(tf.expand_dims(self.log_std_var, 0),1), \
                               [tf.shape(x)[0], tf.shape(x)[1], 1])

            log_std_out = tf.keras.layers.Lambda(tiled_log_std)(input_layer_lstm)
            logits = tf.keras.layers.Concatenate(axis=2)([logits, log_std_out])

        ## DEFINE VALUE FUNCTION LAYERS ##
            
        # Value function hiddens layers
        if not self.vf_share_layers:
            last_vf_layer = input_layer_encoder
            i = 1
            for size in self.vf_hiddens:
                last_vf_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_value_{}".format(i),
                    activation=vf_activation,
                    kernel_initializer=normc_initializer(1.0),
                )(last_vf_layer)
                i += 1
        else:
            last_vf_layer = lstm_out
        
        # Output layer
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(last_vf_layer)

        ## CREATE THE MODELS ##

        # Encoder model
        if self.vf_share_layers:
            self.encoder_model = tf.keras.Model(
                inputs=input_layer_encoder, outputs=last_layer_encoder
            )
        else:
            self.encoder_model = tf.keras.Model(
                inputs=input_layer_encoder, outputs=[last_layer_encoder, value_out]
            )
        
        # LSTM model
        if self.vf_share_layers:
            self.rnn_model = tf.keras.Model(
                inputs=[input_layer_lstm, seq_in, state_in_h, state_in_c],
                outputs=[logits, value_out, state_h, state_c],
            )
        else:
            self.rnn_model = tf.keras.Model(
                inputs=[input_layer_lstm, seq_in, state_in_h, state_in_c],
                outputs=[logits, state_h, state_c],
            )

        # Print out model summary in INFO logging mode.
        # if logger.isEnabledFor(logging.INFO):
        self.encoder_model.summary()
        self.rnn_model.summary()

        # Add prev-a/r to this model's view, if required.
        if self.use_prev_action:
            self.view_requirements[SampleBatch.PREV_ACTIONS] = \
                ViewRequirement(SampleBatch.ACTIONS, space=self.action_space,
                                shift=-1)
        if self.use_prev_reward:
            self.view_requirements[SampleBatch.PREV_REWARDS] = \
                ViewRequirement(SampleBatch.REWARDS, shift=-1)


    @override(RecurrentNetwork)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> tuple[TensorType, List[TensorType]]:
        
        assert seq_lens is not None

        # Push observations through encoder net's `forward()` first.
        if self.vf_share_layers:
            wrapped_out = self.encoder_model(input_dict["obs_flat"])
        else:
            wrapped_out, self._value_out = self.encoder_model(input_dict["obs_flat"])

        # Concat. prev-action/reward if required.
        prev_a_r = []
        if self.use_prev_action:
            prev_a = input_dict[SampleBatch.PREV_ACTIONS]
            if isinstance(self.action_space, (Discrete, MultiDiscrete)):
                prev_a = one_hot(prev_a, self.action_space)
            prev_a_r.append(
                tf.reshape(tf.cast(prev_a, tf.float32), [-1, self.action_dim]))
        if self.use_prev_reward:
            prev_a_r.append(
                tf.reshape(
                    tf.cast(input_dict[SampleBatch.PREV_REWARDS], tf.float32),
                    [-1, 1]))

        if prev_a_r:
            wrapped_out = tf.concat([wrapped_out] + prev_a_r, axis=1)

        # Then through our LSTM.
        input_dict["obs_flat"] = wrapped_out

        return super().forward(input_dict, state, seq_lens)
    

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):

        if self.vf_share_layers:
            model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        else:
            model_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        
        return model_out, [h, c]


    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]


    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MLPplusGTrXL(RecurrentNetwork):
    """GTrXL-based model for meta-RL. The policy model is made up of an encoder and a GTrXL net.
     The encoder is a simple feedforward network that preprocesses the observation before sending
     it to the GTrXL, plus, possibly, the previous n actions and rewards. The GTrXL then outputs the logits.
     The value function estimate is either computed by a separate feedforward network that takes the
     same observation as input, or is an additional output of the GTrXL.
    
     Args:
        custom_model_kwargs:
            num_transformer_units (int): The number of transformer units.
            attention_dim (int): The dimension of the transformer units.
            num_heads (int): The number of heads in the attention.
            head_dim (int): The dimension of each head.
            memory_inference (int): The number of previous outputs to use as memory input to each layer.
            position_wise_mlp_dim (int): The dimension of the position-wise MLP.
            init_gru_gate_bias (float): The initial bias of the GRU gate.
            encoder_hiddens (list): The sizes of the hidden layers of the encoder.
            encoder_activation (str): The activation function of the encoder.
            vf_hiddens (list): The sizes of the hidden layers of the value function.
            vf_activation (str): The activation function of the value function.
        model_config:
            max_seq_len (int): The maximum length of the input observation sequence.
            vf_share_layers (bool): Whether to share the layers of the value function with the policy.
            free_log_std (bool): Whether to generate free-floating bias variables for the second half of the model outputs.
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
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        if logger.isEnabledFor(logging.INFO):
            print("custom_model_kwargs: {}".format(custom_model_kwargs))

        # Encoder parameters
        self.encoder_hiddens = list(custom_model_kwargs.get("encoder_hiddens", []))
        self.encoder_activation = custom_model_kwargs.get("encoder_activation")
        encoder_activation = get_activation_fn(self.encoder_activation)

        # GTrXL parameters
        self.num_transformer_units = custom_model_kwargs.get("num_transformer_units")
        self.attention_dim = custom_model_kwargs.get("attention_dim")
        self.num_heads = custom_model_kwargs.get("num_heads")
        self.head_dim = custom_model_kwargs.get("head_dim")
        self.memory_inference = custom_model_kwargs.get("memory_inference")
        self.memory_training = self.memory_inference
        self.position_wise_mlp_dim = custom_model_kwargs.get("position_wise_mlp_dim")
        self.init_gru_gate_bias = custom_model_kwargs.get("init_gru_gate_bias")
        self.use_n_prev_actions = custom_model_kwargs.get("use_n_prev_actions")
        self.use_n_prev_rewards = custom_model_kwargs.get("use_n_prev_rewards")
        self.max_seq_len = model_config.get("max_seq_len")
        free_log_std = model_config.get("free_log_std")
        
        # Value function parameters
        self.vf_share_layers = model_config.get("vf_share_layers")
        if not self.vf_share_layers:
            self.vf_hiddens = list(custom_model_kwargs.get("vf_hiddens", []))
            self.vf_activation = custom_model_kwargs.get("vf_activation")
            vf_activation = get_activation_fn(self.vf_activation)

        # Number of outputs (logit dim)
        self.num_outputs = num_outputs

        # Generate free-floating bias variables for the second half of
        # the outputs.
        if free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2
            self.log_std_var = tf.Variable(
                [0.0] * num_outputs, dtype=tf.float32, name="log_std"
            )

        # Action dimension
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
        elif isinstance(action_space, MultiDiscrete):
            self.action_dim = np.sum(action_space.nvec)
        elif action_space.shape is not None:
            self.action_dim = int(np.product(action_space.shape))
        else:
            self.action_dim = int(len(action_space))

        ## DEFINE ENCODER LAYERS ##
            
        # Input layer
        input_layer_encoder = tf.keras.layers.Input(
            shape=(int(np.product(obs_space.shape)), ), name="observations"
        )

        # Encoder hidden layers
        last_layer_encoder = input_layer_encoder
        i = 1
        for size in self.encoder_hiddens:
            last_layer_encoder = tf.keras.layers.Dense(
                size,
                name="fc_encoder_{}".format(i),
                activation=encoder_activation,
                kernel_initializer=normc_initializer(1.0),
            )(last_layer_encoder)
            i += 1

        # Number of outputs (logit dim)
        self.encoder_out_size = self.encoder_hiddens[-1]

        ## DEFINE GTRXL LAYERS ##      

        # Add prev-action/reward nodes to input to GTrXL.
        self.num_gtrxl_inputs = self.encoder_out_size
        if self.use_n_prev_actions:
            self.num_gtrxl_inputs += self.use_n_prev_actions * self.action_dim
        if self.use_n_prev_rewards:
            self.num_gtrxl_inputs += self.use_n_prev_rewards

        # Input layer
        input_layer_gtrxl = tf.keras.layers.Input(
            shape=(None, self.num_gtrxl_inputs), name="inputs_gtrxl")
        memory_ins = [
            tf.keras.layers.Input(
                shape=(None, self.attention_dim),
                dtype=tf.float32,
                name="memory_in_{}".format(i))
            for i in range(self.num_transformer_units)
        ]
        E_out = tf.keras.layers.Dense(self.attention_dim)(input_layer_gtrxl)
        memory_outs = [E_out]

        # GTrXL layers
        for i in range(self.num_transformer_units):
            # RelativeMultiHeadAttention part.
            MHA_out = SkipConnection(
                RelativeMultiHeadAttention(
                    out_dim=self.attention_dim,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    input_layernorm=True,
                    output_activation=tf.nn.relu),
                fan_in_layer=GRUGate(self.init_gru_gate_bias),
                name="mha_{}".format(i + 1))(
                    E_out, memory=memory_ins[i])
            # Position-wise MLP part.
            E_out = SkipConnection(
                tf.keras.Sequential(
                    (tf.keras.layers.LayerNormalization(axis=-1),
                     PositionwiseFeedforward(
                         out_dim=self.attention_dim,
                         hidden_dim=self.position_wise_mlp_dim,
                         output_activation=tf.nn.relu))),
                fan_in_layer=GRUGate(self.init_gru_gate_bias),
                name="pos_wise_mlp_{}".format(i + 1))(MHA_out)
            memory_outs.append(E_out)

        # Output layer
        logits = tf.keras.layers.Dense(
            num_outputs, activation=tf.keras.activations.linear, name="logits"
        )(E_out)

        # Concat the log std vars to the end of the state-dependent means.
        if free_log_std:

            def tiled_log_std(x):
                return tf.tile(tf.expand_dims(tf.expand_dims(self.log_std_var, 0),1), \
                               [tf.shape(x)[0], tf.shape(x)[1], 1])

            log_std_out = tf.keras.layers.Lambda(tiled_log_std)(input_layer_gtrxl)
            logits = tf.keras.layers.Concatenate(axis=2)([logits, log_std_out])
        
        ## DEFINE VALUE FUNCTION LAYERS ##
        
        # Value function hiddens layers
        if not self.vf_share_layers:
            last_vf_layer = input_layer_encoder
            i = 1
            for size in self.vf_hiddens:
                last_vf_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_value_{}".format(i),
                    activation=vf_activation,
                    kernel_initializer=normc_initializer(1.0),
                )(last_vf_layer)
                i += 1
        else:
            last_vf_layer = E_out
        
        # Output layer
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(last_vf_layer)

        ## CREATE THE MODELS ##

        # Encoder model
        if self.vf_share_layers:
            self.encoder_model = tf.keras.Model(
                inputs=input_layer_encoder, outputs=last_layer_encoder
            )
        else:
            self.encoder_model = tf.keras.Model(
                inputs=input_layer_encoder, outputs=[last_layer_encoder, value_out]
            )

        # GTrXL model
        if self.vf_share_layers:
            outs = [logits, value_out]
        else:
            outs = [logits]
        self.gtrxl_model = tf.keras.Model(
            inputs=[input_layer_gtrxl] + memory_ins, outputs=outs + memory_outs[:-1])

        # Print out model summary in INFO logging mode.
        # if logger.isEnabledFor(logging.INFO):
        self.encoder_model.summary()
        self.gtrxl_model.summary()

        # Add prev steps to this model's view, if required.
        for i in range(self.num_transformer_units):
            space = Box(-1.0, 1.0, shape=(self.attention_dim, ))
            self.view_requirements["state_in_{}".format(i)] = \
                ViewRequirement(
                    "state_out_{}".format(i),
                    shift="-{}:-1".format(self.memory_inference),
                    # Repeat the incoming state every max-seq-len times.
                    batch_repeat_value=self.max_seq_len,
                    space=space)
            self.view_requirements["state_out_{}".format(i)] = \
                ViewRequirement(
                    space=space,
                    used_for_training=False)
        self.view_requirements["obs"].space = self.obs_space
        
        # Add prev-a/r to this model's view, if required.
        if self.use_n_prev_actions:
            self.view_requirements[SampleBatch.PREV_ACTIONS] = \
                ViewRequirement(
                    SampleBatch.ACTIONS,
                    space=self.action_space,
                    shift="-{}:-1".format(self.use_n_prev_actions))
        if self.use_n_prev_rewards:
            self.view_requirements[SampleBatch.PREV_REWARDS] = \
                ViewRequirement(
                    SampleBatch.REWARDS,
                    shift="-{}:-1".format(self.use_n_prev_rewards))


    @override(ModelV2)
    def forward(self, input_dict, state: List[TensorType],
                seq_lens: TensorType) -> tuple[TensorType, List[TensorType]]:
        
        assert seq_lens is not None

        # Push observations through encoder net's `forward()` first.
        if self.vf_share_layers:
            wrapped_out = self.encoder_model(input_dict["obs_flat"])
        else:
            wrapped_out, self._value_out = self.encoder_model(input_dict["obs_flat"])

        # Concat. prev-action/reward if required.
        prev_a_r = []
        if self.use_n_prev_actions:
            if isinstance(self.action_space, Discrete):
                for i in range(self.use_n_prev_actions):
                    prev_a_r.append(
                        one_hot(input_dict[SampleBatch.PREV_ACTIONS][:, i],
                                self.action_space))
            elif isinstance(self.action_space, MultiDiscrete):
                for i in range(
                        self.use_n_prev_actions,
                        step=self.action_space.shape[0]):
                    prev_a_r.append(
                        one_hot(
                            tf.cast(
                                input_dict[SampleBatch.PREV_ACTIONS]
                                [:, i:i + self.action_space.shape[0]],
                                tf.float32), self.action_space))
            else:
                prev_a_r.append(
                    tf.reshape(
                        tf.cast(input_dict[SampleBatch.PREV_ACTIONS],
                                tf.float32),
                        [-1, self.use_n_prev_actions * self.action_dim]))
        if self.use_n_prev_rewards:
            prev_a_r.append(
                tf.reshape(
                    tf.cast(input_dict[SampleBatch.PREV_REWARDS], tf.float32),
                    [-1, self.use_n_prev_rewards]))
        
        if prev_a_r:
            wrapped_out = tf.concat([wrapped_out] + prev_a_r, axis=1)
        
        # Then through the GTrXL
        input_dict["obs_flat"] = input_dict["obs"] = wrapped_out

        return self.forward_gtrxl(input_dict, state, seq_lens)


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
        if self.vf_share_layers:
            self._value_out = all_out[1]
            memory_outs = all_out[2:]
        else:
            memory_outs = all_out[1:]

        return out, [
            tf.reshape(m, [-1, self.attention_dim]) for m in memory_outs
        ]

    # @override(RecurrentNetwork)
    # def get_initial_state(self) -> List[np.ndarray]:
    #     return [
    #         tf.zeros(self.view_requirements["state_in_{}".format(i)].space.shape)
    #         for i in range(self.num_transformer_units)
    #     ]
    
    @override(RecurrentNetwork)
    def get_initial_state(self) -> List[np.ndarray]:
        return []

    @override(ModelV2)
    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])
