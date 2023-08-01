import tensorflow as tf
from keras.layers import Dense, Input

class QNet(tf.keras.Model):
    """Q-Network. """

    def __init__(self, state_size, action_size):
        """
        state_size: Dimension of state space
        action_size: Dimension of action space
        """
        fc_units = [128, 64, 32, 16, 8]
        self.state_size = state_size
        self.action_size = action_size
        self.fc_units = fc_units
        self.depth = len(fc_units)

        super(QNet, self).__init__()
        self.fc = [Dense(units = x, activation = tf.nn.relu) for x in fc_units[:-1]]
        self.fc.append(Dense(units = self.action_size))
        

    def call(self, state, **kwargs):
        """ forward path """
        x = self.fc[0](state)
        for i in range(1, self.depth):
            x = self.fc[i](x)
        return x
    
if __name__ == "__main__":
    my_model = QNet(4, 2, [2])
    my_model.build(input_shape=(1,4))
    variables = my_model.variables
    print (variables)
    x = tf.constant([[1,2,3,4], [3,4,2,1]])
    y = my_model.call(x)
    print (y.numpy())
    print (tf.reduce_max(y, axis=-1, keepdims=True))
