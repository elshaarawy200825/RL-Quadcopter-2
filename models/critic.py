from keras import layers, models, optimizers
from keras import backend as K
class Critic:

    def __init__(self, state_size, action_size):
       
        self.state_size = state_size
        self.action_size = action_size


        self.build_model()

    def build_model(self):

        states_layer = layers.Input(shape=(self.state_size,), name='states')
        actions_layer = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states_layer = layers.Dense(units=256, kernel_regularizer=layers.regularizers.l2(1e-6), activation='relu')(states_layer)
        net_states_layer = layers.BatchNormalization()(net_states_layer)
        net_states_layer = layers.Dense(units=256, kernel_regularizer=layers.regularizers.l2(1e-6), activation='relu')(net_states_layer)

        # hidden layer
        net_actions_layer = layers.Dense(units=256, kernel_regularizer=layers.regularizers.l2(1e-6), activation='relu')(actions_layer)
        net_actions = layers.BatchNormalization()(net_actions_layer)
        net_actions = layers.Dense(units=256, kernel_regularizer=layers.regularizers.l2(1e-6), activation='relu')(net_actions_layer)

        
        net = layers.Add()([net_states_layer, net_actions_layer])
        net = layers.Activation('relu')(net)


        # Add final output layer to prduce action values (Q values)
        q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states_layer, actions_layer], outputs=q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(q_values, actions_layer)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)