import tensorflow.keras.layers


def attention_3d_block(hidden_states):
    """
    Many-to-one attention mechanism for Keras.
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, 128)
    @author: felixhao28.
    """
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention Luong's multiplicative style score
    score_first_part = tensorflow.keras.layers.Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = tensorflow.keras.layers.Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = tensorflow.keras.layers.dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = tensorflow.keras.layers.Activation('softmax', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = tensorflow.keras.layers.dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = tensorflow.keras.layers.concatenate([context_vector, h_t], name='attention_output')
    attention_vector = tensorflow.keras.layers.Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector