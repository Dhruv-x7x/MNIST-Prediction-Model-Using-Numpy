
'''
PARAM INITIALIZATION FOR 4 LAYERS

    W1 = np.random.randn(64, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((64, 1))
    W2 = np.random.randn(64, 64) * np.sqrt(2 / 64)
    b2 = np.zeros((64, 1))
    W3 = np.random.randn(32, 64) * np.sqrt(2 / 64)
    b3 = np.zeros((32, 1))
    W4 = np.random.randn(10, 32) * np.sqrt(2 / 32)
    b4 = np.zeros((10, 1))

'''

def learn(W1, b1, W2, b2, W3, b3, W4, b4, lr, iterations):
    m = train_x.shape[0]
    A0 = train_x.T

    for it in range(iterations):
        # -------- Forward Pass --------
        Z1 = np.dot(W1, A0) + b1
        A1 = leaky_relu(Z1)

        Z2 = np.dot(W2, A1) + b2
        A2 = leaky_relu(Z2)

        Z3 = np.dot(W3, A2) + b3
        A3 = leaky_relu(Z3)

        Z4 = np.dot(W4, A3) + b4
        A4 = softmax(Z4)

        # Prediction
        Y_predicted = np.argmax(A4, axis=0)
        correct = np.sum(Y_predicted == train_y)
        accuracy = correct / m * 100

        # One-hot encode labels
        Y_onehot = np.zeros((10, m))
        Y_onehot[train_y, np.arange(m)] = 1

        # -------- Backward Pass --------
        dZ4 = A4 - Y_onehot
        dW4 = (1/m) * np.dot(dZ4, A3.T)
        db4 = (1/m) * np.sum(dZ4, axis=1, keepdims=True)

        dA3 = np.dot(W4.T, dZ4)
        dZ3 = dA3 * leaky_relu_deriv(Z3)
        dW3 = (1/m) * np.dot(dZ3, A2.T)
        db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

        dA2 = np.dot(W3.T, dZ3)
        dZ2 = dA2 * leaky_relu_deriv(Z2)
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * leaky_relu_deriv(Z1)
        dW1 = (1/m) * np.dot(dZ1, A0.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # Gradient norms
        grad_norms = {
            "||dW1||": np.linalg.norm(dW1),
            "||db1||": np.linalg.norm(db1),
            "||dW2||": np.linalg.norm(dW2),
            "||db2||": np.linalg.norm(db2),
            "||dW3||": np.linalg.norm(dW3),
            "||db3||": np.linalg.norm(db3),
            "||dW4||": np.linalg.norm(dW4),
            "||db4||": np.linalg.norm(db4),
        }

        # -------- Parameter Update --------
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3
        W4 -= lr * dW4
        b4 -= lr * db4

        if it % 10 == 0:
            print(f"Iteration {it+1}, Accuracy: {accuracy:.2f}%")
            print("Gradient Norms:", grad_norms)

    return W1, b1, W2, b2, W3, b3, W4, b4
