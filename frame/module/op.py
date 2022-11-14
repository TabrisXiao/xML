
from tensorflow import Module

class opBase(Module):
    def __init__(self, name, opKind):
        super(opBase, self).__init__(name)
        self.kind = opKind