def learn(W1, b1, W2, b2, lr, iterations, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = train_x.shape[0]
    A0 = train_x.T

    # Initialize Adam moment estimates
    mW1 = np.zeros_like(W1)
    vW1 = np.zeros_like(W1)
    mb1 = np.zeros_like(b1)
    vb1 = np.zeros_like(b1)
    
    mW2 = np.zeros_like(W2)
    vW2 = np.zeros_like(W2)
    mb2 = np.zeros_like(b2)
    vb2 = np.zeros_like(b2)

    for it in range(1, iterations + 1):  # start from 1 for bias correction
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

        # Adam updates for W1 and b1
        mW1 = beta1 * mW1 + (1 - beta1) * dW1
        vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
        mW1_hat = mW1 / (1 - beta1 ** it)
        vW1_hat = vW1 / (1 - beta2 ** it)
        W1 -= lr * mW1_hat / (np.sqrt(vW1_hat) + epsilon)

        mb1 = beta1 * mb1 + (1 - beta1) * db1
        vb1 = beta2 * vb1 + (1 - beta2) * (db1 ** 2)
        mb1_hat = mb1 / (1 - beta1 ** it)
        vb1_hat = vb1 / (1 - beta2 ** it)
        b1 -= lr * mb1_hat / (np.sqrt(vb1_hat) + epsilon)

        # Adam updates for W2 and b2
        mW2 = beta1 * mW2 + (1 - beta1) * dW2
        vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
        mW2_hat = mW2 / (1 - beta1 ** it)
        vW2_hat = vW2 / (1 - beta2 ** it)
        W2 -= lr * mW2_hat / (np.sqrt(vW2_hat) + epsilon)

        mb2 = beta1 * mb2 + (1 - beta1) * db2
        vb2 = beta2 * vb2 + (1 - beta2) * (db2 ** 2)
        mb2_hat = mb2 / (1 - beta1 ** it)
        vb2_hat = vb2 / (1 - beta2 ** it)
        b2 -= lr * mb2_hat / (np.sqrt(vb2_hat) + epsilon)

        if it % 10 == 0:
            grad_norms = {
                "||dW1||": np.linalg.norm(dW1),
                "||db1||": np.linalg.norm(db1),
                "||dW2||": np.linalg.norm(dW2),
                "||db2||": np.linalg.norm(db2),
            }
            print(f"Iteration {it}, Accuracy: {accuracy:.2f}%")
            print("Gradient Norms:", grad_norms)

    return W1, b1, W2, b2
