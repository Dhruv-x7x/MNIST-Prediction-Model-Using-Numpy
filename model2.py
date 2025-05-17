def learn(W1, b1, W2, b2, lr, iterations, beta=0.9):

    m = train_x.shape[0]
    A0 = train_x.T

    # Initialize velocities
    vW1 = np.zeros_like(W1)
    vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2)
    vb2 = np.zeros_like(b2)

    for it in range(iterations):
        # Forward pass
        Z1 = np.dot(W1, A0) + b1
        A1 = leaky_relu(Z1)

        Z2 = np.dot(W2, A1) + b2
        A2 = softmax(Z2)

        Y_predicted = np.argmax(A2, axis=0)
        correct = np.sum(Y_predicted == train_y)
        accuracy = correct / m * 100

        # One-hot encode labels
        Y_onehot = np.zeros((10, m))
        Y_onehot[train_y, np.arange(m)] = 1

        # Backward pass
        dZ2 = A2 - Y_onehot
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * leaky_relu_deriv(Z1)
        dW1 = (1/m) * np.dot(dZ1, A0.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # Momentum updates
        vW1 = beta * vW1 + (1 - beta) * dW1
        vb1 = beta * vb1 + (1 - beta) * db1
        vW2 = beta * vW2 + (1 - beta) * dW2
        vb2 = beta * vb2 + (1 - beta) * db2

        W1 -= lr * vW1
        b1 -= lr * vb1
        W2 -= lr * vW2
        b2 -= lr * vb2

        if it % 10 == 0:
            grad_norms = {
                "||dW1||": np.linalg.norm(dW1),
                "||db1||": np.linalg.norm(db1),
                "||dW2||": np.linalg.norm(dW2),
                "||db2||": np.linalg.norm(db2),
            }
            print(f"Iteration {it+1}, Accuracy: {accuracy:.2f}%")
            print("Gradient Norms:", grad_norms)

    return W1, b1, W2, b2