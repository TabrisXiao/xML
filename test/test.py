from frame.module import fullyConnected

layer = fullyConnected(
    name = "testLayer",
    inShape = 3,
    outShape = 5
)
layer.build()