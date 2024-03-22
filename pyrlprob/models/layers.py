from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.tf.layers import (
    GRUGate,
    RelativeMultiHeadAttention,
    SkipConnection,
)
from ray.rllib.models.tf.attention_net import PositionwiseFeedforward
from gym.spaces import Box, Discrete, MultiDiscrete
import numpy as np

tf1, tf, tfv = try_import_tf()

def compute_action_dim(action_space):
    """ 
    Computes the action dimension.
    """

    if isinstance(action_space, Discrete):
        return action_space.n
    elif isinstance(action_space, MultiDiscrete):
        return np.sum(action_space.nvec)
    elif action_space.shape is not None:
        return int(np.product(action_space.shape))
    else:
        return int(len(action_space))


def cnn_plus_input_model(model_config, obs_space):
    """ 
    Builds a convolutional + input model.
    """

    # Model parameters
    conv_filters = list(model_config.get("conv_filters", []))
    conv_activation = model_config.get("conv_activation", "relu")

    # Process image inputs with CNN(s), and concat the
    # output with discrete and real inputs
    cnns = {}
    one_hot = {}
    flatten = {}
    num_outputs = 0
    for i, component in enumerate(obs_space):
        # Image inputs -> CNN.
        if len(component.shape) == 3:
            # Activation function
            activation = get_activation_fn(
                conv_activation, framework="tf")

            # Model
            input_layer_cnn = tf.keras.layers.Input(
                shape=component.shape, name="inputs_cnn_{}".format(i)
            )

            # CNN layers
            last_layer = input_layer_cnn
            for i, (out_size, kernel, stride) in enumerate(conv_filters[:-1], 1):
                last_layer = tf.keras.layers.Conv2D(
                    out_size,
                    kernel, 
                    strides=stride
                    if isinstance(stride, (list, tuple)) else (stride, stride), 
                    activation=activation,
                    padding="same", 
                    data_format="channels_last",
                    name="conv_{}".format(i)
                )(last_layer)
            
            # Last CNN layer
            out_size, kernel, stride = conv_filters[-1]
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=stride
                if isinstance(stride, (list, tuple)) else (stride, stride),
                activation=activation,
                padding="valid",
                data_format="channels_last",
                name="conv_out"
            )(last_layer)

            # Flatten
            last_layer = tf.keras.layers.Flatten(name="flatten", 
                                                 data_format="channels_last")(last_layer)
            
            # Model
            cnn = tf.keras.Model(
                inputs=input_layer_cnn, outputs=last_layer
            )

            cnn.summary()

            # Add to list
            cnns[i] = cnn

            num_outputs += last_layer.shape[1]
        # Discrete inputs -> One-hot encode.
        elif isinstance(component, Discrete):
            one_hot[i] = True
            num_outputs += component.n
        # Float inputs -> Flatten.
        else:
            flatten[i] = int(np.product(component.shape))
            num_outputs += flatten[i]
    
    return cnns, one_hot, flatten, num_outputs


def mlp_model(model_config, num_inputs):
    """ 
    Builds an MLP model.
    """

    # Model parameters
    mlp_hiddens = list(model_config.get("mlp_hiddens", []))
    mlp_activation = model_config.get("mlp_activation", "tanh")

    # Activation function
    activation = get_activation_fn(mlp_activation, framework="tf")

    # Input layer
    input_layer_mlp = tf.keras.layers.Input(
        shape=(num_inputs, ), name="inputs_mlp"
    )

    # MLP layers
    last_layer = input_layer_mlp
    for i, size in enumerate(mlp_hiddens, 1):
        last_layer = tf.keras.layers.Dense(
            size, 
            activation=activation, 
            name="fc_{}".format(i),
            kernel_initializer=normc_initializer(1.0)
        )(last_layer)
    
    # Number of outputs
    num_outputs = mlp_hiddens[-1] if mlp_hiddens else num_inputs
    
    # Model
    model = tf.keras.Model(
        inputs=input_layer_mlp, outputs=last_layer
    )

    model.summary()

    return model, num_outputs


def lstm_plus_output_model(model_config, num_inputs, num_outputs, action_space):
    """ 
    Builds an LSTM + output model.
    """

    # Model parameters
    lstm_cell_size = model_config.get("lstm_cell_size", 64)
    use_prev_action = model_config.get("lstm_use_prev_action", False)
    use_prev_reward = model_config.get("lstm_use_prev_reward", False)
    free_log_std = model_config.get("free_log_std", False)

    # Generate free-floating bias variables for the second half of
    # the outputs, if free_log_std = True
    if free_log_std:
        assert num_outputs % 2 == 0, (
            "num_outputs must be divisible by two",
            num_outputs,
        )
        num_outputs = num_outputs // 2
        log_std_var = tf.Variable(
            [0.0] * num_outputs, dtype=tf.float32, name="log_std"
        )

    # Action dimension
    action_dim = compute_action_dim(action_space)
    
    # Add prev-action/reward nodes to input to LSTM.
    if use_prev_action:
        num_inputs = num_inputs + action_dim
    if use_prev_reward:
        num_inputs = num_inputs + 1

    # Input layer
    input_layer_lstm = tf.keras.layers.Input(
        shape=(None, num_inputs), name="inputs_lstm"
    )
    state_in_h = tf.keras.layers.Input(shape=(lstm_cell_size,), name="h")
    state_in_c = tf.keras.layers.Input(shape=(lstm_cell_size,), name="c")
    seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

    # LSTM cell
    lstm_out, state_h, state_c = tf.keras.layers.LSTM(
        lstm_cell_size, return_sequences=True, return_state=True, name="lstm"
    )(
        inputs=input_layer_lstm,
        mask=tf.sequence_mask(seq_in),
        initial_state=[state_in_h, state_in_c],
    )

    # Output layer
    out = tf.keras.layers.Dense(
        num_outputs,
        activation=tf.keras.activations.linear,
        name="outs")(lstm_out)
    
    # Concat the log std vars to the end of the state-dependent means.
    if free_log_std:

        def tiled_log_std(x):
            return tf.tile(tf.expand_dims(tf.expand_dims(log_std_var, 0),1), \
                            [tf.shape(x)[0], tf.shape(x)[1], 1])

        log_std_out = tf.keras.layers.Lambda(tiled_log_std)(input_layer_lstm)
        out = tf.keras.layers.Concatenate(axis=2)([out, log_std_out])

    # Model
    model = tf.keras.Model(
        inputs=[input_layer_lstm, seq_in, state_in_h, state_in_c],
        outputs=[out, state_h, state_c]
    )

    model.summary()

    return model


def gtrxl_plus_output_model(model_config, num_inputs, num_outputs, action_space):
    """ 
    Builds a GTrXL model + output model.
    """

    # Model parameters
    num_transformer_units = model_config.get("num_transformer_units", 1)
    attention_dim = model_config.get("attention_dim", 64)
    num_heads = model_config.get("num_heads", 1)
    head_dim = model_config.get("head_dim", 64)
    position_wise_mlp_dim = model_config.get("position_wise_mlp_dim", 64)
    init_gru_gate_bias = model_config.get("init_gru_gate_bias", 2.0)
    use_n_prev_actions = model_config.get("use_n_prev_actions", 0)
    use_n_prev_rewards = model_config.get("use_n_prev_rewards", 0)
    free_log_std = model_config.get("free_log_std", False)

    # Generate free-floating bias variables for the second half of
    # the outputs, if free_log_std = True
    if free_log_std:
        assert num_outputs % 2 == 0, (
            "num_outputs must be divisible by two",
            num_outputs,
        )
        num_outputs = num_outputs // 2
        log_std_var = tf.Variable(
            [0.0] * num_outputs, dtype=tf.float32, name="log_std"
        )

    # Action dimension
    action_dim = compute_action_dim(action_space)

    # Add prev-action/reward nodes to input to GTrXL.
    if use_n_prev_actions:
        num_inputs = num_inputs + action_dim * use_n_prev_actions
    if use_n_prev_rewards:
        num_inputs = num_inputs + use_n_prev_rewards

    # Input layer
    input_layer_gtrxl = tf.keras.layers.Input(
        shape=(None, num_inputs), name="inputs_gtrxl"
    )
    memory_ins = [
        tf.keras.layers.Input(
            shape=(None, attention_dim),
            dtype=tf.float32,
            name="memory_in_{}".format(i))
        for i in range(num_transformer_units)
    ]
    E_out = tf.keras.layers.Dense(attention_dim)(input_layer_gtrxl)
    memory_outs = [E_out]

    # GTrXL layers
    for i in range(num_transformer_units):
        # RelativeMultiHeadAttention part.
        MHA_out = SkipConnection(
            RelativeMultiHeadAttention(
                out_dim=attention_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                input_layernorm=True,
                output_activation=tf.nn.relu),
            fan_in_layer=GRUGate(init_gru_gate_bias),
            name="mha_{}".format(i + 1))(
                E_out, memory=memory_ins[i])
        # Position-wise MLP part.
        E_out = SkipConnection(
            tf.keras.Sequential(
                (tf.keras.layers.LayerNormalization(axis=-1),
                    PositionwiseFeedforward(
                        out_dim=attention_dim,
                        hidden_dim=position_wise_mlp_dim,
                        output_activation=tf.nn.relu))),
            fan_in_layer=GRUGate(init_gru_gate_bias),
            name="pos_wise_mlp_{}".format(i + 1))(MHA_out)
        memory_outs.append(E_out)
    
    # Output layer
    out = tf.keras.layers.Dense(
        num_outputs,
        activation=tf.keras.activations.linear,
        name="outs")(E_out)

    # Concat the log std vars to the end of the state-dependent means.
    if free_log_std:

        def tiled_log_std(x):
            return tf.tile(tf.expand_dims(tf.expand_dims(log_std_var, 0),1), \
                            [tf.shape(x)[0], tf.shape(x)[1], 1])

        log_std_out = tf.keras.layers.Lambda(tiled_log_std)(input_layer_gtrxl)
        out = tf.keras.layers.Concatenate(axis=2)([out, log_std_out])

    # Model
    model = tf.keras.Model(
        inputs=[input_layer_gtrxl] + memory_ins,
        outputs=[out] + memory_outs[:-1],
    )

    model.summary()

    return model


def output_layer(model_config, num_inputs, num_outputs):
    """ 
    Builds a linear output layer.
    """

    # Model parameters
    free_log_std = model_config.get("free_log_std", False)

    # Generate free-floating bias variables for the second half of
    # the outputs, if free_log_std = True
    if free_log_std:
        assert num_outputs % 2 == 0, (
            "num_outputs must be divisible by two",
            num_outputs,
        )
        num_outputs = num_outputs // 2
        log_std_var = tf.Variable(
            [0.0] * num_outputs, dtype=tf.float32, name="log_std"
        )
    
    # Input layer
    input_layer = tf.keras.layers.Input(
        shape=(num_inputs, ), name="inputs_final_layer"
    )
    
    # Output layer
    out = tf.keras.layers.Dense(
        num_outputs,
        activation=tf.keras.activations.linear,
        name="outs",
        kernel_initializer=normc_initializer(0.01))(input_layer)

    # Concat the log std vars to the end of the state-dependent means.
    if free_log_std:

        def tiled_log_std(x):
            return tf.tile(
                tf.expand_dims(log_std_var, 0), [tf.shape(x)[0], 1])

        log_std_out = tf.keras.layers.Lambda(tiled_log_std)(input_layer)
        out = tf.keras.layers.Concatenate(axis=1)(
            [out, log_std_out])

    # Model
    model = tf.keras.Model(
        inputs=input_layer, outputs=out
    )

    model.summary()

    return model


def conv_mlp_rec_model(model_config, obs_space, action_space, num_outputs):
    """ 
    Builds a convolutional + MLP model, with an optional LSTM or GTrXL layer, 
    and a final linear layer.
    """

    # CNN model + input layer
    config = {
        "conv_filters": list(model_config.get("conv_filters", [])),
        "conv_activation": model_config.get("conv_activation", "relu"),
    }

    cnns, one_hot, flatten, num_outputs_cnn = cnn_plus_input_model(config, obs_space)

    # MLP model
    config = {
        "mlp_hiddens": list(model_config.get("mlp_hiddens", [])),
        "mlp_activation": model_config.get("mlp_activation", "tanh"),
    }

    mlp, num_outputs_mlp = mlp_model(config, num_outputs_cnn)

    # Recurrent + output model
    use_lstm = model_config.get("use_lstm", False)
    use_gtrxl = model_config.get("use_gtrxl", False)
    if use_lstm:
        # LSTM model
        config = {
            "lstm_cell_size": model_config.get("lstm_cell_size", 64),
            "lstm_use_prev_action": model_config.get("lstm_use_prev_action", False),
            "lstm_use_prev_reward": model_config.get("lstm_use_prev_reward", False),
            "free_log_std": model_config.get("free_log_std", False)
        }

        output_model = lstm_plus_output_model(
            config, num_outputs_mlp, num_outputs, action_space)
    elif use_gtrxl:
        # GTrXL model
        config = {
            "num_transformer_units": model_config.get("num_transformer_units", 1),
            "attention_dim": model_config.get("attention_dim", 64),
            "num_heads": model_config.get("num_heads", 1),
            "head_dim": model_config.get("head_dim", 64),
            "position_wise_mlp_dim": model_config.get("position_wise_mlp_dim", 64),
            "init_gru_gate_bias": model_config.get("init_gru_gate_bias", 2.0),
            "use_n_prev_actions": model_config.get("use_n_prev_actions", 0),
            "use_n_prev_rewards": model_config.get("use_n_prev_rewards", 0),
            "free_log_std": model_config.get("free_log_std", False)
        }

        output_model = gtrxl_plus_output_model(
            config, num_outputs_mlp, num_outputs, action_space)
    else:
        # Linear output layer
        config = {
            "free_log_std": model_config.get("free_log_std", False)
        }

        output_model = output_layer(
            config, num_outputs_mlp, num_outputs
        )

    return cnns, one_hot, flatten, mlp, output_model


def setup_trajectory_view_lstm(use_prev_action, use_prev_reward, action_space, view_requirements):
    """ 
    Sets up the trajectory view for LSTM models.
    """

    if use_prev_action:
        view_requirements[SampleBatch.PREV_ACTIONS] = \
            ViewRequirement(SampleBatch.ACTIONS, space=action_space, shift=-1)
    if use_prev_reward:
        view_requirements[SampleBatch.PREV_REWARDS] = \
            ViewRequirement(SampleBatch.REWARDS, shift=-1)


def setup_trajectory_view_gtrxl(num_transformer_units_list, attention_dim_list, 
            memory_inference_list, max_seq_len,
            use_n_prev_actions, use_n_prev_rewards, 
            action_space, view_requirements):
    """ 
    Sets up the trajectory view for GTrXL models.
    """

    for i in range(len(num_transformer_units_list)):
        prev_transformer_units = num_transformer_units_list[i-1] if i > 0 else 0
        for j in range(num_transformer_units_list[i]):
            space = Box(-1.0, 1.0, shape=(attention_dim_list[i], ))
            view_requirements["state_in_{}".format(prev_transformer_units + j)] = \
                ViewRequirement(
                    "state_out_{}".format(prev_transformer_units + j),
                    shift="-{}:-1".format(memory_inference_list[i]),
                    # Repeat the incoming state every max-seq-len times.
                    batch_repeat_value=max_seq_len,
                    space=space)
            view_requirements["state_out_{}".format(prev_transformer_units + j)] = \
                ViewRequirement(
                    space=space,
                    used_for_training=False)
            
    if use_n_prev_actions:
        view_requirements[SampleBatch.PREV_ACTIONS] = \
            ViewRequirement(
                SampleBatch.ACTIONS,
                space=action_space,
                shift="-{}:-1".format(use_n_prev_actions))
    if use_n_prev_rewards:
        view_requirements[SampleBatch.PREV_REWARDS] = \
            ViewRequirement(
                SampleBatch.REWARDS,
                shift="-{}:-1".format(use_n_prev_rewards))