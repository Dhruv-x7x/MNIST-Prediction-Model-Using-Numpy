import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time

class MLP:

    def __init__(self, layer_dims, activations, keep_probs=None, l2_lambda=0.001, random_seed=None):
       
        # SET SEED
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # INITIALIZE
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1
        self.activations = activations
        self.l2_lambda = l2_lambda
        self.keep_probs = [1.0] * self.L if keep_probs is None else keep_probs
        
        # INIT LAYER PARAMS
        self.parameters = self._initialize_parameters()
        
        # INIT ADAM PARAMS
        self.adam_params = {}
        for l in range(1, self.L + 1):
            self.adam_params[f"mW{l}"] = np.zeros((layer_dims[l], layer_dims[l-1]))
            self.adam_params[f"mb{l}"] = np.zeros((layer_dims[l], 1))
            self.adam_params[f"vW{l}"] = np.zeros((layer_dims[l], layer_dims[l-1]))
            self.adam_params[f"vb{l}"] = np.zeros((layer_dims[l], 1))
            
        # INIT METRICS TRACKING
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        
    def _initialize_parameters(self):
        parameters = {}
        for l in range(1, self.L + 1):
            # HE INIT
            scale = np.sqrt(2.0 / self.layer_dims[l-1])
            parameters[f"W{l}"] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * scale
            parameters[f"b{l}"] = np.ones((self.layer_dims[l], 1)) * 0.01 if self.activation[l-1] in ['relu', 'leaky_relu'] else np.zeros((self.layer_dims[l], 1))
            parameters[f"lambda{l}"] = self.l2_lambda
            
        return parameters
    
    def _activate(self, Z, activation):

        if activation == 'relu':
            return np.maximum(0, Z)
        elif activation == 'leaky_relu':
            return np.where(Z > 0, Z, 0.01 * Z)
        elif activation == 'tanh':
            return np.tanh(Z)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        elif activation == 'softmax':
            Z_stable = Z - np.max(Z, axis=0, keepdims=True)
            exps = np.exp(Z_stable)
            return exps / np.sum(exps, axis=0, keepdims=True)
        else:
            raise ValueError(f"Activation {activation} not recognized")
    
    def _activate_derivative(self, Z, activation, A=None):
       
        if activation == 'relu':
            return np.where(Z > 0, 1, 0)
        elif activation == 'leaky_relu':
            return np.where(Z > 0, 1, 0.01)
        elif activation == 'tanh':
            return 1 - np.power(np.tanh(Z), 2)
        elif activation == 'sigmoid':
            if A is None:
                A = self._activate(Z, 'sigmoid')
            return A * (1 - A)
        elif activation == 'softmax':
            return 1  # Placeholder - not actually used because we deal with it differently in backpropagation
        else:
            raise ValueError(f"Activation {activation} not recognized")
    
    def _apply_batch_norm(self, Z, gamma, beta, cache=None, mode='train', epsilon=1e-12):
       
        if mode == 'train':
            # BATCH NORM
            mu = np.mean(Z, axis=1, keepdims=True)
            var = np.var(Z, axis=1, keepdims=True)
            
            # NORMALIZE
            Z_norm = (Z - mu) / np.sqrt(var + epsilon)
            
            # SCALE AND SHIFT
            out = gamma * Z_norm + beta
            
            # CACHE OUT
            cache = (Z, Z_norm, mu, var, gamma, beta, epsilon)
            
            # RUNNING AVERAGE
            if not hasattr(self, 'bn_params'):
                self.bn_params = {}
            
            layer_idx = len(self.bn_params) + 1
            if f'running_mean{layer_idx}' not in self.bn_params:
                self.bn_params[f'running_mean{layer_idx}'] = mu
                self.bn_params[f'running_var{layer_idx}'] = var
            else:
                momentum = 0.9
                self.bn_params[f'running_mean{layer_idx}'] = momentum * self.bn_params[f'running_mean{layer_idx}'] + (1 - momentum) * mu
                self.bn_params[f'running_var{layer_idx}'] = momentum * self.bn_params[f'running_var{layer_idx}'] + (1 - momentum) * var
                
        elif mode == 'test':
            layer_idx = len(cache) + 1
            mu = self.bn_params[f'running_mean{layer_idx}']
            var = self.bn_params[f'running_var{layer_idx}']
            Z_norm = (Z - mu) / np.sqrt(var + epsilon)
            out = gamma * Z_norm + beta
            cache = None
            
        return out, cache
    
    def forward_propagate(self, X, mode='train'):

        A = X
        caches = []
        
        # FORWARD PROP
        for l in range(1, self.L + 1):
            A_prev = A
            
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]
            Z = np.dot(W, A_prev) + b

            linear_cache = (A_prev, W, b)
            
            activation = self.activations[l-1]
            A = self._activate(Z, activation)
            
            D = None
            if mode == 'train' and self.keep_probs[l-1] < 1.0:
                D = np.random.rand(*A.shape) < self.keep_probs[l-1]
                A = (A * D) / self.keep_probs[l-1]
            
            activation_cache = (Z, D, activation)
            caches.append((linear_cache, activation_cache))
            
        return A, caches
    
    def compute_cost(self, AL, Y, caches=None, include_regularization=True):
        m = Y.shape[1]
        
        # CROSS ENTROPY
        epsilon = 0.1
        Y_smooth = (1 - epsilon) * Y + epsilon / Y.shape[0]
        
        logprobs = np.multiply(np.log(AL + 1e-10), Y_smooth)
        cost = -np.sum(logprobs) / m
        
        if include_regularization and caches is not None:
            L2_cost = 0
            for l in range(1, self.L + 1):
                W = self.parameters[f"W{l}"]
                L2_cost += np.sum(np.square(W))
            
            L2_cost = (self.l2_lambda / (2 * m)) * L2_cost
            cost += L2_cost
            
        return cost
    
    def backward_propagate(self, AL, Y, caches):
        m = Y.shape[1]
        gradients = {}
        
        epsilon = 0.1
        Y_smooth = (1 - epsilon) * Y + epsilon / Y.shape[0]
        
        dZ = AL - Y_smooth
        
        # FINAL LAYER CALCULATION
        linear_cache, activation_cache = caches[self.L - 1]
        A_prev, W, b = linear_cache
        Z, D, activation = activation_cache
        
        # COMPUTE GRADIENTS
        gradients[f"dW{self.L}"] = (np.dot(dZ, A_prev.T) / m) + (self.l2_lambda * W / m)
        gradients[f"db{self.L}"] = np.sum(dZ, axis=1, keepdims=True) / m
        
        dA = np.dot(W.T, dZ)
        
        # OTHER LAYERS
        for l in range(self.L - 1, 0, -1):
            linear_cache, activation_cache = caches[l-1]
            A_prev, W, b = linear_cache
            Z, D, activation = activation_cache
            
            dA = (dA * D) / self.keep_probs[l-1] if D is not None else dA
                
            dZ = dA * self._activate_derivative(Z, activation)
            
            gradients[f"dW{l}"] = (np.dot(dZ, A_prev.T) / m) + (self.l2_lambda * W / m)
            gradients[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m
            
            if l > 1:
                dA = np.dot(W.T, dZ)
                
        return gradients
    
    def update_parameters(self, gradients, learning_rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        
        # GRADIENT CLIPPING
        clip_value = 5.0
        
        for l in range(1, self.L + 1):
            dW = gradients[f"dW{l}"]
            db = gradients[f"db{l}"]
            
            dW = np.clip(dW, -clip_value, clip_value)
            db = np.clip(db, -clip_value, clip_value)
            
            mW = self.adam_params[f"mW{l}"]
            mb = self.adam_params[f"mb{l}"]
            vW = self.adam_params[f"vW{l}"]
            vb = self.adam_params[f"vb{l}"]
            
            mW = beta1 * mW + (1 - beta1) * dW
            mb = beta1 * mb + (1 - beta1) * db
            vW = beta2 * vW + (1 - beta2) * np.square(dW)
            vb = beta2 * vb + (1 - beta2) * np.square(db)
            
            mW_corrected = mW / (1 - beta1**t)
            mb_corrected = mb / (1 - beta1**t)
            vW_corrected = vW / (1 - beta2**t)
            vb_corrected = vb / (1 - beta2**t)
            
            self.parameters[f"W{l}"] -= learning_rate * mW_corrected / (np.sqrt(vW_corrected) + epsilon)
            self.parameters[f"b{l}"] -= learning_rate * mb_corrected / (np.sqrt(vb_corrected) + epsilon)
            
            self.adam_params[f"mW{l}"] = mW
            self.adam_params[f"mb{l}"] = mb
            self.adam_params[f"vW{l}"] = vW
            self.adam_params[f"vb{l}"] = vb
    
    def one_hot_encode(self, y, num_classes=None):
        num_classes = np.max(y) + 1 if num_classes == None else num_classes
        m = y.shape[0]
        one_hot = np.zeros((num_classes, m))
        one_hot[y, np.arange(m)] = 1
        return one_hot
    
    def compute_accuracy(self, X, y, mode='test'):
        # FORWARD PROP
        AL, _ = self.forward_propagate(X, mode=mode)
    
        # GET PREDICTIONS
        predictions = np.argmax(AL, axis=0)
        accuracy = np.mean(predictions == y)
        return accuracy * 100
    
    def compute_confusion_matrix(self, X, y, mode='test'):
        n_classes = 10  # For MNIST
        
        # FORWARD PROP
        AL, _ = self.forward_propagate(X, mode=mode)
        
        # GET PREDICTION
        predictions = np.argmax(AL, axis=0)
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(len(y)):
            conf_matrix[y[i], predictions[i]] += 1
        return conf_matrix
    
    def train(self, X_train, y_train, X_val=None, y_val=None, learning_rate=0.01, num_epochs=10, batch_size=128, print_interval=10, patience=None):
        m = X_train.shape[1]
        num_batches = int(np.ceil(m / batch_size))
        
        Y_train_onehot = self.one_hot_encode(y_train)
        
        # EARLY STOPPING CRITERION - PATIENCE
        if patience is not None:
            best_val_acc = 0
            patience_counter = 0
        
        # TRAINING
        t = 0  # Adam time step
        start_time = time()
        
        for epoch in range(num_epochs):
            #  SHUFFLE
            permutation = np.random.permutation(m)
            X_train_shuffled = X_train[:, permutation]
            Y_train_shuffled = Y_train_onehot[:, permutation]
            y_train_shuffled = y_train[permutation]
            
            # MINI BATCH GD
            for batch in range(num_batches):
                t += 1  # Increment time step
                
                # GET BATCH
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, m)
                X_batch = X_train_shuffled[:, start_idx:end_idx]
                Y_batch = Y_train_shuffled[:, start_idx:end_idx]
                
                # FOWARD PROP
                AL, caches = self.forward_propagate(X_batch, mode='train')
                
                # LOSS
                cost = self.compute_cost(AL, Y_batch, caches)
                
                # BACKWARD PROP
                gradients = self.backward_propagate(AL, Y_batch, caches)
                
                # UPDATE PARAMS
                self.update_parameters(gradients, learning_rate, t)
            
            # METRICS
            train_accuracy = self.compute_accuracy(X_train, y_train, mode='train')
            self.train_acc_history.append(train_accuracy)
            
            AL_train, caches_train = self.forward_propagate(X_train, mode='train')
            train_loss = self.compute_cost(AL_train, Y_train_onehot, caches_train)
            self.train_loss_history.append(train_loss)
            
            val_accuracy = None
            val_loss = None
            
            if X_val is not None and y_val is not None:
                val_accuracy = self.compute_accuracy(X_val, y_val, mode='test')
                self.val_acc_history.append(val_accuracy)
                
                Y_val_onehot = self.one_hot_encode(y_val)
                AL_val, _ = self.forward_propagate(X_val, mode='test')
                val_loss = self.compute_cost(AL_val, Y_val_onehot, include_regularization=False)
                self.val_loss_history.append(val_loss)
                
                if patience is not None:
                    if val_accuracy > best_val_acc:
                        best_val_acc = val_accuracy
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
            
            # PRINT METRICS
            if (epoch + 1) % print_interval == 0:
                time_elapsed = time() - start_time
                print(f"Epoch {epoch+1}/{num_epochs} - {time_elapsed:.2f}s - Loss: {train_loss:.4f} - Acc: {train_accuracy:.2f}%", 
                      end="")
                
                if val_accuracy is not None:
                    print(f" - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.2f}%")
                else:
                    print()
        
        # CONFUSION MATRIX
        train_conf_matrix = self.compute_confusion_matrix(X_train, y_train)
        
        # RESULTS
        history = {
            'train_loss': self.train_loss_history,
            'train_acc': self.train_acc_history,
            'val_loss': self.val_loss_history,
            'val_acc': self.val_acc_history,
            'train_conf_matrix': train_conf_matrix
        }
        
        return history
    
    def predict(self, X):
        # FORWARD PROP
        AL, _ = self.forward_propagate(X, mode='test')
        
        # GET PREDICTIONS
        predictions = np.argmax(AL, axis=0)
        
        return predictions
    
    def plot_training_history(self, history=None):
        if history is None:
            history = {
                'train_loss': self.train_loss_history,
                'train_acc': self.train_acc_history,
                'val_loss': self.val_loss_history,
                'val_acc': self.val_acc_history
            }
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training and validation loss
        ax1.plot(history['train_loss'], label='Training Loss')
        if history['val_loss']:
            ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss over epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training and validation accuracy
        ax2.plot(history['train_acc'], label='Training Accuracy')
        if history['val_acc']:
            ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Accuracy over epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, confusion_matrix):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        
        # Set labels
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        # Set ticks
        classes = np.arange(10)  # MNIST has 10 classes (0-9)
        ax.set_xticks(classes)
        ax.set_yticks(classes)
        
        # Add values in cells
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{confusion_matrix[i, j]}", 
                        ha="center", va="center", 
                        color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")
        
        plt.tight_layout()
        plt.show()
        
    def plot_weights(self):
        # Get first layer weights
        W1 = self.parameters["W1"]
        
        # Calculate grid size
        num_neurons = W1.shape[0]
        grid_size = int(np.ceil(np.sqrt(num_neurons)))
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        
        # Plot each filter
        for i in range(num_neurons):
            if i < grid_size * grid_size:
                row = i // grid_size
                col = i % grid_size
                
                # Reshape weight to 28x28 image
                weight_img = W1[i].reshape(28, 28)
                
                # Plot
                axes[row, col].imshow(weight_img, cmap='viridis')
                axes[row, col].axis('off')
                axes[row, col].set_title(f"Neuron {i}")
        
        # Hide unused subplots
        for i in range(num_neurons, grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.show()

def main():

    # LOAD DATA
    print("Loading data...")
    data = pd.read_csv("data/train.csv")
    print(f"Data shape: {data.shape}")
    
    # CONVERT TO NUMPY ARRAY
    data_array = np.array(data)
    
    # SPLIT
    labels = data_array[:, 0]
    features = data_array[:, 1:] / 255.0  # NORMALIZE to 0,1
    
    # SHUFFLE
    indices = np.random.permutation(data_array.shape[0])
    features = features[indices]
    labels = labels[indices]
    
    # Split into train, validation, and test sets (80%, 10%, 10%)
    test_size = int(0.1 * features.shape[0])
    val_size = int(0.1 * features.shape[0])
    
    X_test = features[:test_size].T
    y_test = labels[:test_size]
    
    X_val = features[test_size:test_size+val_size].T
    y_val = labels[test_size:test_size+val_size]
    
    X_train = features[test_size+val_size:].T
    y_train = labels[test_size+val_size:]
    
    print(f"Training samples: {X_train.shape[1]}")
    print(f"Validation samples: {X_val.shape[1]}")
    print(f"Test samples: {X_test.shape[1]}")
    
    # MODEL
    model = MLP(
        layer_dims=[784, 128, 64, 32, 10],
        activations=['leaky_relu', 'leaky_relu', 'leaky_relu', 'softmax'],
        keep_probs=[0.7, 0.7, 0.7, 1.0],
        l2_lambda=0.01,
        random_seed=42
    )

    # TRAIN
    print("\nTraining model...")
    history = model.train(
        X_train, y_train, 
        X_val=X_val, y_val=y_val,
        learning_rate=0.01,
        num_epochs=50,
        batch_size=128,
        print_interval=5,
        patience=10
    )
    
    # EVALUATION
    test_accuracy = model.compute_accuracy(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.2f}%")
    
    test_conf_matrix = model.compute_confusion_matrix(X_test, y_test)
    
    print("\nPlotting training history...")
    model.plot_training_history(history)
    
    print("\nPlotting confusion matrix...")
    model.plot_confusion_matrix(test_conf_matrix)
    
    print("\nPlotting weights visualization...")
    model.plot_weights()
    
    predictions = model.predict(X_test)
    
    print("\nPer-class accuracy:")
    for digit in range(10):
        digit_indices = np.where(y_test == digit)[0]
        if len(digit_indices) > 0:
            digit_accuracy = np.mean(predictions[digit_indices] == digit) * 100
            print(f"Digit {digit}: {digit_accuracy:.2f}%")

if __name__ == "__main__":
    main()