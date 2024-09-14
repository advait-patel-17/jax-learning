import jax.numpy as jnp
from jax import grad, jit, lax, vmap
import matplotlib.pyplot as plt
import time

@jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

@jit
def loss(weights, features, labels):
    def loss_per_sample(feature, label):
        return label * jnp.log(sigmoid(jnp.dot(weights, feature))) + ( 1 - label)*jnp.log(1 - sigmoid(jnp.dot(weights, feature)))
    #vmap the loss_per_sample function
    loss_per_sample_vmap = vmap(loss_per_sample)
    return -jnp.mean(loss_per_sample_vmap(features, labels))

@jit
def logistic_regression(weights, features):
    return sigmoid(jnp.dot(weights, features))

@jit
def train_logistic_regression(initial_weights, features, labels, learning_rate=0.1, num_epochs=100):
    grad_loss = grad(loss)
    def update(weights, _):
        grads = grad_loss(weights, features, labels)
        weights = weights - learning_rate * grads
        return weights, None
    weights, _ = lax.scan(update, initial_weights, None, length=num_epochs)
    return weights

def plot_decision_boundary(features, preds, labels):
    plt.scatter(features[:, 0], features[:, 1], c=preds)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Logistic Regression Decision Boundary")
    plt.show()


weights = jnp.array([0.0, 0.0])
#have 100 samples, 2 features
features = jnp.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13]])
labels = jnp.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# start timer
start_time = time.time()
trained_weights = train_logistic_regression(weights, features, labels)
#end timer
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds. includes jit compilation")


start_time = time.time()
trained_weights = train_logistic_regression(weights, features, labels)
#end timer
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds. no jit compilation")

#time taken without jit: 2.92 seconds
#time taken with jit (including compilation): 0.096 seconds
#time take with jit (already compiled): 2.3e-5

print(trained_weights)


preds = jnp.array([logistic_regression(trained_weights, feature) for feature in features])
print(preds)

print(f"Trained weights: {trained_weights}")

plot_decision_boundary(features, preds, labels)