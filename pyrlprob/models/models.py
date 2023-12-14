import logging

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.framework import try_import_tf
from typing import Dict, List
from ray.rllib.utils.typing import TensorType, List, ModelConfigDict
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import override
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.tf.layers import (
    GRUGate,
    RelativeMultiHeadAttention,
    SkipConnection,
)
from ray.rllib.models.tf.attention_net import PositionwiseFeedforward
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete
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
    ) -> (TensorType, List[TensorType]):
        
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
     it to the LSTM cell. The LSTM cell is a standard LSTM cell that outputs the logits.
     The value function estimate is computed by a separate feedforward network that takes the
     same observation as input.

     Args:
        custom_model_kwargs:
            lstm_cell_size (int): The size of the LSTM cell.
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
        super(MLPplusLSTM, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        
        # if logger.isEnabledFor(logging.INFO):
        print("custom_model_kwargs: {}".format(custom_model_kwargs))

        self.cell_size = custom_model_kwargs.get("lstm_cell_size")
        self.encoder_hiddens = list(custom_model_kwargs.get("encoder_hiddens", []))
        self.encoder_activation = custom_model_kwargs.get("encoder_activation")
        encoder_activation = get_activation_fn(self.encoder_activation)
        vf_share_layers = model_config.get("vf_share_layers")
        free_log_std = model_config.get("free_log_std")
        if not vf_share_layers:
            self.vf_hiddens = list(custom_model_kwargs.get("vf_hiddens", []))
            self.vf_activation = custom_model_kwargs.get("vf_activation")
            vf_activation = get_activation_fn(self.vf_activation)
        self.max_seq_len = model_config.get("max_seq_len")
    
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
        
        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs"
        )
        state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with the encoder_hiddens and send the output to the LSTM cell
        last_layer = input_layer
        i = 1
        for size in self.encoder_hiddens:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_encoder_{}".format(i),
                activation=encoder_activation,
                kernel_initializer=normc_initializer(1.0),
            )(last_layer)
            i += 1

        # LSTM cell
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            self.cell_size, return_sequences=True, return_state=True, name="lstm"
        )(
            inputs=last_layer,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c],
        )

        # Postprocess LSTM output with another hidden layer and compute logits
        logits = tf.keras.layers.Dense(
            num_outputs, activation=tf.keras.activations.linear, name="logits"
        )(lstm_out)

        # Concat the log std vars to the end of the state-dependent means.
        if free_log_std:

            def tiled_log_std(x):
                return tf.tile(tf.expand_dims(tf.expand_dims(self.log_std_var, 0),1), [tf.shape(x)[0], tf.shape(x)[1], 1])

            log_std_out = tf.keras.layers.Lambda(tiled_log_std)(input_layer)
            logits = tf.keras.layers.Concatenate(axis=2)([logits, log_std_out])

        # Compute value estimate with the vf_hiddens
        if not vf_share_layers:
            last_vf_layer = input_layer
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
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(last_vf_layer)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, value_out, state_h, state_c],
        )

        # Print out model summary in INFO logging mode.
        # if logger.isEnabledFor(logging.INFO):
        self.rnn_model.summary()

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        
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
     it to the GTrXL. The GTrXL then outputs the logits.
     The value function estimate is computed by a separate feedforward network that takes the
     same observation as input.
    
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

        # if logger.isEnabledFor(logging.INFO):
        print("custom_model_kwargs: {}".format(custom_model_kwargs))

        self.num_transformer_units = custom_model_kwargs.get("num_transformer_units")
        self.attention_dim = custom_model_kwargs.get("attention_dim")
        self.num_heads = custom_model_kwargs.get("num_heads")
        self.head_dim = custom_model_kwargs.get("head_dim")
        self.memory_inference = custom_model_kwargs.get("memory_inference")
        self.position_wise_mlp_dim = custom_model_kwargs.get("position_wise_mlp_dim")
        self.init_gru_gate_bias = custom_model_kwargs.get("init_gru_gate_bias")
        self.max_seq_len = model_config.get("max_seq_len")
        
        self.encoder_hiddens = list(custom_model_kwargs.get("encoder_hiddens", []))
        self.encoder_activation = custom_model_kwargs.get("encoder_activation")
        encoder_activation = get_activation_fn(self.encoder_activation)

        vf_share_layers = model_config.get("vf_share_layers")
        free_log_std = model_config.get("free_log_std")
        if not vf_share_layers:
            self.vf_hiddens = list(custom_model_kwargs.get("vf_hiddens", []))
            self.vf_activation = custom_model_kwargs.get("vf_activation")
            vf_activation = get_activation_fn(self.vf_activation)

        self.obs_dim = obs_space.shape[0]
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

        # Observation input: sequence of last max_seq_len observations
        input_layer = tf.keras.layers.Input(
            shape=(self.max_seq_len, self.obs_dim), name="inputs"
        )

        # Memory inputs: sequence of last memory_inference outputs,
        # each with dimension (max_seq_len, attention_dim), of each transformer unit
        memory_ins = [
            tf.keras.layers.Input(
                shape=(self.memory_inference*self.max_seq_len, self.attention_dim),
                dtype=tf.float32,
                name="memory_in_{}".format(i),
            )
            for i in range(self.num_transformer_units)
        ]

        # Preprocess observation with the encoder_hiddens and send the output to the LSTM cell
        last_layer = input_layer
        if len(self.encoder_hiddens) > 0:
            i = 1
            for size in self.encoder_hiddens:
                last_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_encoder_{}".format(i),
                    activation=encoder_activation,
                    kernel_initializer=normc_initializer(1.0),
                )(last_layer)
                i += 1

        # Map encoder dim to input/output transformer (attention) dim.
        E_out = tf.keras.layers.Dense(self.attention_dim)(last_layer)

        # Output, collected and concat'd to build the internal
        # memory units used for additional contextual information.
        memory_outs = []

        # 2) Create N Transformer blocks
        for i in range(self.num_transformer_units):
            # RelativeMultiHeadAttention part.
            MHA_out = SkipConnection(
                RelativeMultiHeadAttention(
                    out_dim=self.attention_dim,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    input_layernorm=True,
                    output_activation=tf.nn.relu,
                ),
                fan_in_layer=GRUGate(self.init_gru_gate_bias),
                name="mha_{}".format(i + 1),
            )(E_out, memory=memory_ins[i])

            # Position-wise MLP part.
            E_out = SkipConnection(
                tf.keras.Sequential(
                    (
                        tf.keras.layers.LayerNormalization(axis=-1),
                        PositionwiseFeedforward(
                            out_dim=self.attention_dim,
                            hidden_dim=self.position_wise_mlp_dim,
                            output_activation=tf.nn.relu,
                        ),
                    )
                ),
                fan_in_layer=GRUGate(self.init_gru_gate_bias),
                name="pos_wise_mlp_{}".format(i + 1),
            )(MHA_out)
            # Output of position-wise MLP == E(l-1), which is concat'd
            # to the current Mem block (M(l-1)) to yield E~(l-1), which is then
            # used by the next transformer block.
            memory_outs.append(E_out)

        # Postprocess TrXL output with another hidden layer and compute values.
        logits = tf.keras.layers.Dense(
            num_outputs, activation=tf.keras.activations.linear, name="logits"
        )(E_out)

        # Concat the log std vars to the end of the state-dependent means.
        if free_log_std:

            def tiled_log_std(x):
                return tf.tile(tf.expand_dims(tf.expand_dims(self.log_std_var, 0),1), [tf.shape(x)[0], tf.shape(x)[1], 1])

            log_std_out = tf.keras.layers.Lambda(tiled_log_std)(input_layer)
            logits = tf.keras.layers.Concatenate(axis=2)([logits, log_std_out])
        
        # Compute value estimate with the vf_hiddens
        if not vf_share_layers:
            last_vf_layer = input_layer
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
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(last_vf_layer)

        outs = [logits, value_out]

        self.gtrxl_model = tf.keras.models.Model(inputs=[input_layer] + memory_ins, \
                                                 outputs=outs + memory_outs)

        # Print out model summary in INFO logging mode.
        # if logger.isEnabledFor(logging.INFO):
        self.gtrxl_model.summary()

    @override(RecurrentNetwork)
    def forward_rnn(
        self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType
    ) -> (TensorType, List[TensorType]):
        
        # Last max_seq_len observations
        observations = state[0]
        observations = tf.concat((observations, inputs), axis=1)[:, -self.max_seq_len :, :]

        # Memory inputs
        memory_ins = state[1:]
        
        # Pass the observations and the previous memory to the model
        all_outs = self.gtrxl_model([observations] + memory_ins)

        # Logits and value estimate
        logits = all_outs[0]
        T = tf.shape(inputs)[1]  # Length of input segment (time).
        logits = logits[:, -T:, :]  # Only take the last T logits.
        self._value_out = all_outs[1]

        # Memory outputs
        memory_outs = all_outs[2:]

        # New memory inputs
        memory_ins_new = []
        for i in range(self.num_transformer_units):
            memory_ins_new.append(tf.concat((memory_ins[i], memory_outs[i]), axis=1)[:, -self.memory_inference*self.max_seq_len :, :])

        # New state
        new_state = [observations] + memory_ins_new

        return logits, new_state

    @override(RecurrentNetwork)
    def get_initial_state(self) -> List[np.ndarray]:
        # Initialize last max_seq_len observations to zeros
        state = [np.zeros((self.max_seq_len, self.obs_dim), np.float32)]

        # Initialize memory inputs to zeros
        for _ in range(self.num_transformer_units):
            state.append(np.zeros((self.memory_inference*self.max_seq_len, self.attention_dim), np.float32))
        return state

    @override(ModelV2)
    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])


# class TrXLforMetaRL(RecurrentNetwork):
#     """A TrXL net Model described in [1]."""

#     def __init__(
#         self,
#         obs_space,
#         action_space,
#         num_outputs,
#         model_config,
#         name,
#         **custom_model_kwargs
#     ):
#         super().__init__(
#             obs_space, action_space, num_outputs, model_config, name
#         )

#         # if logger.isEnabledFor(logging.INFO):
#         print("custom_model_kwargs: {}".format(custom_model_kwargs))

#         self.num_transformer_units = custom_model_kwargs.get("num_transformer_units")
#         self.attention_dim = custom_model_kwargs.get("attention_dim")
#         self.num_heads = custom_model_kwargs.get("num_heads")
#         self.head_dim = custom_model_kwargs.get("head_dim")
#         self.position_wise_mlp_dim = custom_model_kwargs.get("position_wise_mlp_dim")
#         self.max_seq_len = model_config.get("max_seq_len")
#         self.obs_dim = obs_space.shape[0]
#         self.num_outputs = num_outputs

#         inputs = tf.keras.layers.Input(
#             shape=(self.max_seq_len, self.obs_dim), name="inputs"
#         )
#         E_out = tf.keras.layers.Dense(self.attention_dim)(inputs)

#         for _ in range(self.num_transformer_units):
#             MHA_out = SkipConnection(
#                 RelativeMultiHeadAttention(
#                     out_dim=self.attention_dim,
#                     num_heads=self.num_heads,
#                     head_dim=self.head_dim,
#                     input_layernorm=False,
#                     output_activation=None,
#                 ),
#                 fan_in_layer=None,
#             )(E_out)
#             E_out = SkipConnection(
#                 PositionwiseFeedforward(self.attention_dim, self.position_wise_mlp_dim)
#             )(MHA_out)
#             E_out = tf.keras.layers.LayerNormalization(axis=-1)(E_out)

#         # Postprocess TrXL output with another hidden layer and compute values.
#         logits = tf.keras.layers.Dense(
#             self.num_outputs, activation=tf.keras.activations.linear, name="logits"
#         )(E_out)
#         value_out = tf.keras.layers.Dense(
#             1,
#             name="value_out",
#             activation=None,
#             kernel_initializer=normc_initializer(0.01),
#         )(E_out)

#         self.base_model = tf.keras.models.Model([inputs], [logits, value_out])

#         self.base_model.summary()

#     @override(RecurrentNetwork)
#     def forward_rnn(
#         self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType
#     ) -> (TensorType, List[TensorType]):
#         # To make Attention work with current RLlib's ModelV2 API:
#         # We assume `state` is the history of L recent observations (all
#         # concatenated into one tensor) and append the current inputs to the
#         # end and only keep the most recent (up to `max_seq_len`). This allows
#         # us to deal with timestep-wise inference and full sequence training
#         # within the same logic.
#         observations = state[0]
#         observations = tf.concat((observations, inputs), axis=1)[:, -self.max_seq_len :]
#         logits, self._value_out = self.base_model([observations])
#         T = tf.shape(inputs)[1]  # Length of input segment (time).
#         logits = logits[:, -T:]

#         return logits, [observations]

#     @override(RecurrentNetwork)
#     def get_initial_state(self) -> List[np.ndarray]:
#         # State is the T last observations concat'd together into one Tensor.
#         # Plus all Transformer blocks' E(l) outputs concat'd together (up to
#         # tau timesteps).
#         return [np.zeros((self.max_seq_len, self.obs_dim), np.float32)]
    
#     @override(ModelV2)
#     def value_function(self):
#         return tf.reshape(self._value_out, [-1])
    
