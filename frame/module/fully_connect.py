from frame.module.core import layerBase
from frame.module.core import builder
from tensorflow.python.ops.math_ops import matmul as tfmatmul
from tensorflow.python.ops.math_ops import add as tfadd

class fullyConnected(layerBase):
    """fully connected layer is also known as dense layer.
    The input/output has to be a vector (rank-1 tensor).
    example: inShape = [3] or 3
    The fully connected layer can be expressed as
        y = x*W + b
    where b (bias) with dimension outShape and W (weight) is a matrix with shape
    [outShape, inShape]
    """
    def __init__(
        self, 
        name,
        inShape,
        outShape,
        use_bias = True,
        trainable = True,
        actiavtion = None,
        dtype = None
    ):
        super().__init__(
            name,
            trainable,
            dtype,
            actiavtion,
            1)
        if hasattr(inShape, "__len__"):
            if len(inShape) != 1:
                raise Exception("inShape has to be either a list with size =1 or a integer")
            else:
                self.inShape = inShape[0]
        if hasattr(outShape, "__len__"):
            if len(outShape) != 1:
                raise Exception("outShape has to be either a list with size =1 or a integer")
            else:
                self.outShape = outShape[0]
        self.inShape = inShape
        self.outShape = outShape
        self.use_bias = use_bias

    def createWeight(self):
        self.weight = builder.createVariable(
            name = self.name+"_weight",
            shape = [self.outShape, self.inShape]
        )
    def createBias(self):
        self.bias = builder.createVariable(
            name = self.name+"_bias",
            shape = [self.outShape]
        )
    
    def build(self):
        self.weight = self.createWeight()
        if self.use_bias: self.bias = self.createBias()
    
    def __call__(self, x):
        out = tfmatmul(x, self.weight, name = self.name+"_matmulOp")
        if self.use_bias:
            out = tfadd(out, self.bias, name=self.name+"_addOp")
        return self.actiavtion(out, name = self.name+"_actOp")
