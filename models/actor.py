from keras import layers, models, optimizers
from keras import backend as K
class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""

        # i/p state layer
        states_layer = layers.Input(shape=(self.state_size,), name='states')

        # h layers
        h_layer = layers.Dense(units=256, kernel_regularizer=layers.regularizers.l2(1e-6), activation='relu')(states_layer)
        h_layer = layers.BatchNormalization()(h_layer)
        h_layer = layers.Dense(units=256, kernel_regularizer=layers.regularizers.l2(1e-6), activation='relu')(h_layer)
        h_layer = layers.BatchNormalization()(h_layer)
        h_layer = layers.Dense(units=256, kernel_regularizer=layers.regularizers.l2(1e-6), activation='relu')(h_layer)


        # o/p layer with sigmoid to get probability of the result 
        raw_actions_layer = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(h_layer)

        actions_layer = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions_layer)

        # Create Keras model
        self.model = models.Model(inputs=states_layer, outputs=actions_layer)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions_layer)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
