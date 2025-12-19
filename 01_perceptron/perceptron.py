import numpy as np

x = np.array([
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0]
])

y = np.array([0, 1, 0, 0])

weight = np.array([2, 3])
bias = 0.0   

#stepfunction
def step(v):
    if v > 0:
        return 1
    else:
        return 0

#forward propogation
def forward(x, weight, bias):
    val = np.dot(x, weight) + bias
    out = [step(v) for v in val]
    return out

print("results before training")
print(forward(x, weight, bias))

# training
lr = .5

for epoch in range(1000):
    for i in range(len(x)):
        val = np.dot(x[i], weight) + bias
        out = step(val)

        error = y[i] - out

        if error != 0:
            weight = weight + lr * error *x[i]
            bias = bias + lr * error   # <-- bias update

print("results after training")
print(forward(x, weight, bias))

print("final weights:", weight)
print("final bias:", bias)
