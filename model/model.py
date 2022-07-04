import tez


class Model(tez.Model):
    def __init__(self):
        super().__init__()
        self.add_layer(tez.Dense(1, activation="sigmoid"))
        self.compile(optimizer="adam", loss="binary_crossentropy")

    def fit(self, x, y, epochs=100, batch_size=32):
        super().fit(x, y, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        return super().predict(x)

    def evaluate(self, x, y):
        return super().evaluate(x, y)


class cifar10_model(Model):
    def __init__(self):
        super().__init__()
        self.add_layer(tez.Dense(128, activation="relu"))
        self.add_layer(tez.Dense(64, activation="relu"))
        self.add_layer(tez.Dense(32, activation="relu"))
        self.add_layer(tez.Dense(10, activation="softmax"))
        self.compile(optimizer="adam", loss="categorical_crossentropy")

    def fit(self, x, y, epochs=100, batch_size=32):
        super().fit(x, y, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        return super().predict(x)

    def evaluate(self, x, y):
        return super().evaluate(x, y)
