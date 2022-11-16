
import tensorflow as tf

class deep:
    """ This is the class managing the low level operation for tensorflow like
    creating variables, adding placeholder, etc
    """
    def __init__(self):
        """The init defines the default setup for the builder
        """
        self.initializer = tf.random_normal_initializer(mean =0, stddev = 0.05)
        self.dtype = tf.float16
        self.activation = tf.nn.relu
    
    def createVariable(self,
        name,
        shape,
        initializer = None,
        trainable = True,
        constraint = None,
        dtype =None
        ):
        """The shape name is required to create the variable.
        """
        if dtype is None: dtype = self.dtype 
        if initializer is None: initializer = self.initializer
        # create the initial value for making variable 
        # (tf requires initial values for creating variables)
        initial_val = initializer(shape, dtype)
        return tf.Variable(
            name = name,
            initial_value = initial_val,
            shape = shape,
            trainable = trainable,
            constraint = constraint,
            dtype = dtype
        )

class layerBase(tf.Module):
    """The dtype and trainable is global variable for this object:
    all the variables create within this class need to have the same
    dtype and trainable setting.
    kind: the internal code for the layer type, 0 means layerBase
    """
    def __init__(self, 
        name,
        trainable = True,
        dtype = None,
        activation = None,
        kind = 0):
        super().__init__(name = name)
        self.trainable = trainable
        self.dtype = dtype
        if activation is None : self.actiavtion = tf.nn.relu
        else : self.actiavtion = activation


builder = deep()