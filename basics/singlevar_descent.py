import jax.numpy as jnp
from jax import grad
from jax import random
from jax import jit
import time

def f(x):
    return (x - 3) ** 2

grad_f = grad(f)

jit_f = jit(f)
jit_grad_f = jit(grad_f)

key = random.key(42)


current = random.uniform(key, (1,), minval=-20, maxval=20)
current = current[0]
print(f"current: {current}, f(current): {jit_f(current)}")
lr = 0.1
last = -1
counter = 0
start_time = time.time()
while (jnp.abs(current - last) > 0.01 and counter < 10000):
    gr = jit_grad_f(current)
    print(f"current: {current}, f(current): {jit_f(current)}")
    print(f"grad: ", gr)
    last = current
    current = current - lr*gr
    counter += 1
end_time = time.time()
print(f"Final current: ", current)
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")

#about 40% of the time w/ jit