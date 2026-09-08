def predict(network, inputs):

    probability = network.predict(inputs)

    return probability


def calculate_loss(network, X, y):

    total_loss = 0.0

    for i in range(len(X)):

        prediction = network.predict(X[i])

        # error = prediction - target
        error = prediction - y[i]

        # L = 1/2 * (prediction - target)^2
        loss = 0.5 * (error ** 2)

        total_loss += loss

    return total_loss / len(X)