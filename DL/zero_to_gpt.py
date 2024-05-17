r"""°°°
# zero to GPT

follow up this video: https://www.youtube.com/watch?v=l-CjXFmcVzY
°°°"""
#|%%--%%| <iPZY7VPQEm|ysn5SuQWHK>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch
from torch.nn import functional as F

#|%%--%%| <ysn5SuQWHK|ibUwAqosYJ>

r"""°°°
Without learning 
°°°"""
#|%%--%%| <ibUwAqosYJ|59Nt7fSxnb>
xs = np.asarray([
    [0,1,0,1,0],
    [0,0,1,1,0],
    [1,1,0,1,0],
    [1,1,0,0,1],
    [0,0,0,1,0],
])


ys = np.asarray([
    [0],
    [1],
    [1],
    [1],
    [0],
])


ins = 5
outs = 1

def weights(ins, outs):
    ws = np.random.randn(ins, outs)
    return ws

ws = weights(ins, outs)
# print(ws)


ers = []
for i in range(5000):
    yh = xs @ ws
    e = yh - ys
    e = np.sum(np.abs(e))
    if(e < 0.05):
        print("Found solution")
        print(ws)
    else:
        ws = weights(ins, outs)

    ers.append(e)

plt.figure(1)
plt.plot(ers)

#|%%--%%| <59Nt7fSxnb|O6ArD5I03L>
r"""°°°
# With learning just linear regression
°°°"""
#|%%--%%| <O6ArD5I03L|aKh6Covpwr>

xs = np.asarray([
    [0,1,0,1,0],
    [0,0,1,1,0],
    [1,1,0,1,0],
    [1,1,0,0,1],
    [0,0,0,1,0],
])


ys = np.asarray([
    [0],
    [1],
    [1],
    [1],
    [0],
])


ins = 5
outs = 1

def weights(ins, outs):
    ws = np.random.randn(ins, outs)
    return ws

ws = weights(ins, outs)
# print(ws)


ers = []
for i in range(5000):
    yh = xs @ ws
    e = yh - ys
    e = np.sum(np.abs(e))
    if(e < 0.05):
        print("Found solution")
        print(ws)
        break
    else:
        mutation = weights(ins, outs) * 0.03
        cw = ws + mutation

        yh = xs @ cw 
        ce = yh - ys
        ce = np.sum(np.abs(ce))

        if(ce < e):
            ws = cw

    ers.append(e)

plt.figure(1)
plt.plot(ers)

#|%%--%%| <aKh6Covpwr|KOMDzy4pjv>
r"""°°°
# With learning just linear with extra data in input
°°°"""
#|%%--%%| <KOMDzy4pjv|VlP9zhAqjt>

xs = np.asarray([
    [0,1,0,1,0],
    [0,0,1,1,0],
    [1,1,0,1,0],
    [1,1,0,0,1],
    [0,0,0,1,0],
])


ys = np.asarray([
    [0],
    [1],
    [1],
    [3],
    [3],
])

xs = np.hstack((xs, np.ones([xs.shape[0], 1])))

ins = 5
outs = 1

def weights(ins, outs):
    ws = np.random.randn(ins, outs)
    return ws

ws = weights(ins+1, outs)
# print(ws)


ers = []
for i in range(5000):
    yh = xs @ ws
    e = yh - ys
    e = np.sum(np.abs(e))
    if(e < 0.05):
        print("Found solution")
        print(ws)
        break
    else:
        mutation = weights(ins+1, outs) * 0.03
        cw = ws + mutation

        yh = xs @ cw 
        ce = yh - ys
        ce = np.sum(np.abs(ce))

        if(ce < e):
            ws = cw

    ers.append(e)

plt.figure(1)
plt.plot(ers)

#|%%--%%| <VlP9zhAqjt|lL6r3cETkh>
r"""°°°
# With learning just non-linear with sin actvion function and a new layer of nodes.
°°°"""
#|%%--%%| <lL6r3cETkh|CRI1B5Z3kn>

xs = np.asarray([
    [0,1,0,1,0],
    [0,0,1,1,0],
    [1,1,0,1,0],
    [1,1,0,0,1],
    [0,0,0,1,0],
])


ys = np.asarray([
    [0],
    [1],
    [1],
    [3],
    [3],
])

xs = np.hstack((xs, np.ones([xs.shape[0], 1])))


ins = 5
outs = 1
nodes = 15

def weights(ins, outs):
    ws = np.random.randn(ins, outs)
    return ws

wi = weights(ins+1, nodes)
ws = weights(nodes, outs)

ers = []
epoch = 5000
for i in range(epoch):

    x = xs @ wi
    x = np.sin(x)

    yh = x @ ws
    
    e = yh - ys
    e = np.sum(np.abs(e))
    if(e < 0.05):
        print("Found solution")
        print(ws)
        break
    else:
        mutation = weights(nodes, outs) * 0.03
        cw = ws + mutation

        x = xs @ wi
        x = np.sin(x)

        yh = x @ cw 
        ce = yh - ys
        ce = np.sum(np.abs(ce))

        if(ce < e):
            ws = cw

    ers.append(e)

plt.figure(1)
plt.plot(ers)


#|%%--%%| <CRI1B5Z3kn|7vMff3d6uD>
r"""°°°
# Learning with gradient descent + non-linear + sin actvion function + a new layer of nodes.
°°°"""
#|%%--%%| <7vMff3d6uD|ld4fEaBtJ6>

xs = np.asarray([
    [0,1,0,1,0],
    [0,0,1,1,0],
    [1,1,0,1,0],
    [1,1,0,0,1],
    [0,0,0,1,0],
])


ys = np.asarray([
    [0],
    [1],
    [1],
    [3],
    [3],
])

xs = np.hstack((xs, np.ones([xs.shape[0], 1])))


ins = 5
outs = 1
nodes = 5

def weights(ins, outs):
    ws = np.random.randn(ins, outs)
    return ws


wi = weights(ins+1, nodes)
ws = weights(nodes, outs)
# print(ws)


ers = []
epoch = 5000
for i in range(epoch):

    x = xs @ wi
    x = np.sin(x)

    yh = x @ ws
    
    e = yh - ys * 1
    ws -= (x.T @ e) * 0.03
    e = np.sum(np.abs(e)) 


    ers.append(e)

plt.figure(1)
plt.plot(ers)


#|%%--%%| <ld4fEaBtJ6|evXShZ3O41>
r"""°°°
# Learning with gradient descent + non-linear + sin actvion function + multi-layer of nodes.
°°°"""
#|%%--%%| <evXShZ3O41|0btrgTnzhS>
# xs = np.asarray([
#     [0,1,0,1,0],
#     [0,0,1,1,0],
#     [1,1,0,1,0],
#     [1,1,0,0,1],
#     [0,0,0,1,0],
# ])


# ys = np.asarray([
#     [0],
#     [1],
#     [1],
#     [3],
#     [3],
# ])

xs = np.asarray([
    [1,0],
    [0,1],
    [1,1],
    [0,0],
])

ys = np.asarray([
    [1],
    [1],
    [0],
    [0],
])


xs = np.hstack((xs, np.ones([xs.shape[0], 1])))


ins = 2
outs = 1
nodes = 2
lr = 0.1



def weights(ins, outs):
    ws = np.random.randn(ins, outs)
    return ws


w0 = weights(ins+1, nodes)
w1 = weights(nodes, nodes)
w2 = weights(nodes, outs)


ers = []
epoch = 5000
for i in range(epoch):

    # Forward pass
    x0 = xs

    z0 = x0 @ w0; x1 = np.sin(z0)
    z1 = x1 @ w1; x2 = np.sin(z1)
    yh = x2 @ w2

    

    # Backward pass
    e = (yh - ys) * 1

    e2 = (e)        * 1
    e1 = (e2 @ w2.T) * np.cos(z1)
    e0 = (e1 @ w1.T) * np.cos(z0)

    # Update weights
    w2 -= (x2.T @ e2) * lr
    w1 -= (x1.T @ e1) * lr
    w0 -= (x0.T @ e0) * lr

    # calculate loss function
    e = np.sum(np.abs(e)) 


    ers.append(e)

plt.figure(1)
plt.plot(ers)

#|%%--%%| <0btrgTnzhS|4XpnErlT8k>
r"""°°°
# Deep Learning with gradient descent + non-linear + sin actvion function + multi-layer of nodes.
°°°"""
#|%%--%%| <4XpnErlT8k|mEoFPy1oAt>
xs = np.asarray([
    [-10],
    [-8],
    [-6],
    [-4],
    [-2],
    [0],
    [2],
    [4],
    [6],
    [8],
    [10]
])

# ys = 3 * xs -2
# ys = 0.5 * xs + 7
ys = xs ** 2


xs = np.hstack((xs, np.ones([xs.shape[0], 1])))


ins = 1
outs = 1
nodes = 100
lr = 0.000001

def weights(ins, outs):
    ws = np.random.randn(ins, outs)
    return ws


w0 = weights(ins+1, nodes)
w1 = weights(nodes, nodes)
w2 = weights(nodes, outs)


ers = []
epoch = 10000
for i in range(epoch):

    # Forward pass
    x0 = xs

    z0 = x0 @ w0; x1 = np.sin(z0)
    z1 = x1 @ w1; x2 = np.sin(z1)
    yh = x2 @ w2

    

    # Backward pass
    e = (yh - ys) * 1

    e2 = (e)        * 1
    e1 = (e2 @ w2.T) * np.cos(z1)
    e0 = (e1 @ w1.T) * np.cos(z0)

    # Update weights
    w2 -= (x2.T @ e2) * lr
    w1 -= (x1.T @ e1) * lr
    w0 -= (x0.T @ e0) * lr

    # calculate loss function
    e = np.sum(np.abs(e)) 


    ers.append(e)

plt.figure(1)
plt.plot(ers)



plt.figure(2)
plt.plot(ys)
plt.plot(yh)

#|%%--%%| <mEoFPy1oAt|PGrkimH804>
r"""°°°
# Deep Learning using pytorch with gradient descent + non-linear + sin actvion function + multi-layer of nodes.
°°°"""
#|%%--%%| <PGrkimH804|iIdi2bOGi9>
xs = torch.asarray([
    [-10],
    [-8],
    [-6],
    [-4],
    [-2],
    [0],
    [2],
    [4],
    [6],
    [8],
    [10]
])

# ys = 3 * xs -2
# ys = 0.5 * xs + 7
ys = xs ** 2
xs = torch.hstack((xs, torch.ones([xs.shape[0], 1])))

xs = torch.tensor(xs).float()
ys = torch.tensor(ys).float()


ins = 1
outs = 1
nodes = 150
lr = 0.001

def weights(ins, outs):
    ws = torch.randn(ins, outs)
    ws = ws.requires_grad_(True)
    return ws


w0 = weights(ins+1, nodes)
w1 = weights(nodes, nodes)
w2 = weights(nodes, outs)

# optimizer= torch.optim.SGD([w0, w1, w2], lr)
optimizer= torch.optim.Adam([w0, w1, w2], lr)

epoch = 500
ers = []
for i in range(epoch):

    # Forward pass
    x0 = xs

    z0 = x0 @ w0; x1 = torch.sin(z0)
    z1 = x1 @ w1; x2 = torch.sin(z1)
    yh = x2 @ w2

    

    loss = F.mse_loss(yh, ys)
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # calculate loss function
    e = loss.item()
    if i % 500:
        print(e)
    ers.append(e)

plt.figure(1)
plt.plot(ers)

plt.figure(2)
plt.plot(ys)
plt.plot(yh.detach.numpy())



#|%%--%%| <iIdi2bOGi9|WlS0gN0Aae>
r"""°°°
# Deep Learning using pytorch with classes with gradient descent + non-linear + sin actvion function + multi-layer of nodes.
°°°"""
#|%%--%%| <WlS0gN0Aae|3hdiYo9Le1>
xs = torch.asarray([
    [-10],
    [-8],
    [-6],
    [-4],
    [-2],
    [0],
    [2],
    [4],
    [6],
    [8],
    [10]
])

# ys = 3 * xs -2
# ys = 0.5 * xs + 7
ys = xs ** 2
xs = torch.hstack((xs, torch.ones([xs.shape[0], 1])))

xs = torch.tensor(xs).float()
ys = torch.tensor(ys).float()


ins = 1
outs = 1
nodes = 200
lr = 0.001

params = []
def weights(ins, outs):
    ws = torch.randn(ins, outs)
    ws = ws.requires_grad_(True)
    params.append(ws)
    return ws


class Model():
    def __init__(self):
        self.w0 = weights(ins+1, nodes)
        self.w1 = weights(nodes, nodes)
        self.w2 = weights(nodes, outs)

    def forward(self, x):
        x = torch.sin(x @ self.w0)
        x = torch.sin(x @ self.w1)
        yh = x @ self.w2
        return yh



model = Model()
# optimizer= torch.optim.SGD([w0, w1, w2], lr)
# optimizer= torch.optim.Adam([w0, w1, w2], lr)
optimizer= torch.optim.Adam(params, lr)

epoch = 500
ers = []
for i in range(epoch):
    yh = model.forward(xs)

    loss = F.mse_loss(yh, ys)
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # calculate loss function
    e = loss.item()
    if i % 500:
        print(e)
    ers.append(e)

plt.figure(1)
plt.plot(ers)

plt.figure(2)
plt.plot(ys)
plt.plot(yh.detach().numpy())


test_val = -5
test_val =  torch.tensor([test_val, 1]).float()
result = model.forward(test_val)
print(result)



#|%%--%%| <3hdiYo9Le1|YtuQJPmPjk>
r"""°°°
# Deep Learning + AUTO Regression + dropout regularion using pytorch with classes with gradient descent + non-linear + sin actvion function + multi-layer of nodes.
°°°"""
#|%%--%%| <YtuQJPmPjk|xrKSLCLpeR>
with open("file.txt", 'r', encoding='utf-8') as f:
    text = f.read()

text = text.lower()
chars = sorted(list(set(text)))
stoi = {c:i for i, c in enumerate(chars)}
data =  [stoi[c] for c in text]
vocab_size = len(chars)

ns = 5
outs = 1
nodes = 100
lr = 0.003

data = torch.tensor(data).float()
xs = torch.stack([data[i:i+ins] for i in range(len(data)-ins)])
ys = torch.stack([data[i+ins:i+ins+1] for i in range(len(data)-ins)])

params = []
def weights(ins, outs):
    ws = torch.randn(ins, outs) *  0.1
    ws = ws.requires_grad_(True)
    params.append(ws)
    return ws


class Model():
    def __init__(self):
        self.w0 = weights(ins, nodes)
        self.w1 = weights(nodes, nodes)
        self.w2 = weights(nodes, outs)

    def forward(self, x):
        # x = torch.sin(x @ self.w0)
        # x = torch.sin(x @ self.w1)

        x = torch.relu(x @ self.w0)
        x = torch.relu(x @ self.w1)


        yh = x @ self.w2
        return yh



model = Model()
# optimizer= torch.optim.SGD([w0, w1, w2], lr)
# optimizer= torch.optim.Adam([w0, w1, w2], lr)
optimizer= torch.optim.Adam(params, lr)

epoch = 500
ers = []
for i in range(epoch):
    yh = model.forward(xs)

    loss = F.mse_loss(yh, ys)
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # calculate loss function
    e = loss.item()
    if i % 500:
        print(e)
    ers.append(e)

plt.figure(1)
plt.plot(ers)

plt.figure(2)
plt.plot(ys)
plt.plot(yh.detach().numpy())

#|%%--%%| <xrKSLCLpeR|d82YYivMak>
r"""°°°
# Deep Learning + AUTO Regression + batches + dropout regularion using pytorch with classes with gradient descent + non-linear + sin actvion function + multi-layer of nodes.
°°°"""
#|%%--%%| <d82YYivMak|5LDuGTJeVb>
with open("file.txt", 'r', encoding='utf-8') as f:
    text = f.read()

text = text.lower()
chars = sorted(list(set(text)))
stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}
data =  [stoi[c] for c in text]
vocab_size = len(chars)

ins = 64
outs = vocab_size
nodes = 200
lr = 0.001

data = torch.tensor(data).float()
params = []
def weights(ins, outs):
    ws = torch.randn(ins, outs) *  0.1
    ws = ws.requires_grad_(True)
    params.append(ws)
    return ws


class Model():
    def __init__(self):
        self.w0 = weights(ins, nodes)
        self.w1 = weights(nodes, nodes)
        self.w2 = weights(nodes, outs)

    def forward(self, x):
        # x = torch.sin(x @ self.w0)
        # x = torch.sin(x @ self.w1)

        x = torch.relu(x @ self.w0)
        x = torch.relu(x @ self.w1)


        yh = x @ self.w2
        return yh



model = Model()
# optimizer= torch.optim.SGD([w0, w1, w2], lr)
# optimizer= torch.optim.Adam([w0, w1, w2], lr)
optimizer= torch.optim.Adam(params, lr)

epoch = 500
ers = []
for i in range(epoch):

    b = torch.randint(len(data)-ins, (100,))
    xs = torch.stack([data[i:i+ins] for i in b])
    ys = torch.stack([data[i+ins:i+ins+1] for i in b])


    yh = model.forward(xs)
    loss = F.cross_entropy(yh.view(-1, vocab_size), ys.long().view(-1))
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # calculate loss function
    e = loss.item()
    if i % 500:
        print(e)
    ers.append(e)

plt.figure(1)
plt.plot(ers)

plt.figure(2)
plt.plot(ys)
plt.plot(torch.argmax(yh.detach(), dim=-1))


test_val = xs[0]

# yh =  model.forward(test_val)
# prob =  F.softmax(yh, dim=0)
# prob2 =  torch.softmax(yh * 0.7, dim=0).item()
# pred =  torch.argmax(yh, dim=0).item()
# print(itos[pred])

gen_text = ""
for i in range(3000):

    yh =  model.forward(test_val)
    prob =  F.softmax(yh, dim=0)
    # pred = torch.argmax(yh).item()
    pred = torch.multinomial(prob, num_samples=1).item()

    test_val = torch.roll(test_val, -1)
    test_val[-1] = pred

    gen_text += itos[pred]


print(gen_text)


#|%%--%%| <5LDuGTJeVb|7jvsYmIFcJ>

r"""°°°
# Deep Learning + Conveoluational Neural Network + AUTO Regression + batches + dropout regularion using pytorch with classes with gradient descent + non-linear + sin actvion function + multi-layer of nodes.
°°°"""
#|%%--%%| <7jvsYmIFcJ|BOjaZywTFU>
with open("file.txt", 'r', encoding='utf-8') as f:
    text = f.read()

text = text.lower()
chars = sorted(list(set(text)))
stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}
data =  [stoi[c] for c in text]
vocab_size = len(chars)

ins = 64
outs = vocab_size
nodes = 200
lr = 0.001
n_emb = 64
embed = torch.randn(vocab_size, n_emb)
pos = torch.randn(ins, n_emb)

data = torch.tensor(data).long()
params = []
def weights(ins, outs):
    ws = torch.randn(ins, outs) *  0.1
    ws = ws.requires_grad_(True)
    params.append(ws)
    return ws


class Head():
    def __init__(self):
        self.wv = weights(n_emb, n_emb//4) # Diveded by number of heads for avoid expandding
    
    def forward(self, x):
        x = x @ self.wv
        # x = torch.sum(x, dim=-2)
        ones = torch.ones(ins, ins)
        tril = torch.tril(ones)
        tril = tril.masked_fill(tril==0, -1e10)
        rew = F.softmax(tril, dim=-1)
        x = rew @ x
        return x


class Model():
    def __init__(self):
        self.heads = [Head(), Head(), Head(), Head()]
        self.w0 = weights(n_emb, nodes)
        self.w1 = weights(nodes, nodes)
        self.w2 = weights(nodes, outs)

    def forward(self, x):
        x = embed[x] * pos 
        x = torch.cat([head.forward(x) for head in self.heads], dim=-1)

        x = torch.relu(x @ self.w0)
        x = torch.relu(x @ self.w1)


        yh = x @ self.w2
        return yh



model = Model()
optimizer= torch.optim.Adam(params, lr)

epoch = 500
ers = []
for i in range(epoch):

    b = torch.randint(len(data)-ins, (100,))
    xs = torch.stack([data[i:i+ins] for i in b])
    ys = torch.stack([data[i+1:i+ins+1] for i in b])


    yh = model.forward(xs)
    loss = F.cross_entropy(yh.view(-1, vocab_size), ys.long().view(-1))
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # calculate loss function
    e = loss.item()
    if i % 500:
        print(e)
    ers.append(e)

plt.figure(1)
plt.plot(ers)

plt.figure(2)
plt.plot(ys)
plt.plot(torch.argmax(yh.detach(), dim=-1))


test_val = xs[0]

# yh =  model.forward(test_val)
# prob =  F.softmax(yh, dim=0)
# prob2 =  torch.softmax(yh * 0.7, dim=0).item()
# pred =  torch.argmax(yh, dim=0).item()
# print(itos[pred])

gen_text = ""
for i in range(3000):

    yh =  model.forward(test_val)
    prob =  F.softmax(yh[-1, :], dim=0)
    # pred = torch.argmax(yh).item()
    pred = torch.multinomial(prob, num_samples=1).item()

    test_val = torch.roll(test_val, -1)
    test_val[-1] = pred

    gen_text += itos[pred]


print(gen_text)
#|%%--%%| <BOjaZywTFU|fnU2NWogZx>
r"""°°°
# Deep Learning + Attention + Conveoluational Neural Network + AUTO Regression + batches + dropout regularion using pytorch with classes with gradient descent + non-linear + sin actvion function + multi-layer of nodes.
°°°"""
#|%%--%%| <fnU2NWogZx|bxqsAtaCbh>
with open("file.txt", 'r', encoding='utf-8') as f:
    text = f.read()

text = text.lower()
chars = sorted(list(set(text)))
stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}
data =  [stoi[c] for c in text]
vocab_size = len(chars)

ins = 64
outs = vocab_size
nodes = 200
lr = 0.001
n_emb = 64
embed = torch.randn(vocab_size, n_emb)
pos = torch.randn(ins, n_emb)

data = torch.tensor(data).long()
params = []
def weights(ins, outs):
    ws = torch.randn(ins, outs) *  0.1
    ws = ws.requires_grad_(True)
    params.append(ws)
    return ws


class Head():
    def __init__(self):
        self.wv = weights(n_emb, n_emb//4) # Diveded by number of heads for avoid expandding
        self.wq = weights(n_emb, n_emb//4) # Diveded by number of heads for avoid expandding
        self.wk = weights(n_emb, n_emb//4) # Diveded by number of heads for avoid expandding
    
    def forward(self, x):
        v = x @ self.wv
        q = x @ self.wq
        k = x @ self.wk

        attn = (q @ k.transpose(-2, -1)) / k.shape[0]**0.5
        tril = torch.tril(attn)
        tril = tril.masked_fill(tril==0, -1e10)
        rew = F.softmax(tril, dim=-1)
        x = rew @ v

        return x


class Model():
    def __init__(self):
        self.heads = [Head(), Head(), Head(), Head()]
        self.w0 = weights(n_emb, nodes)
        self.w1 = weights(nodes, nodes)
        self.w2 = weights(nodes, outs)

    def forward(self, x):
        x = embed[x] + pos 
        x = torch.cat([head.forward(x) for head in self.heads], dim=-1)

        x = torch.relu(x @ self.w0)
        x = torch.relu(x @ self.w1)


        yh = x @ self.w2
        return yh



model = Model()
optimizer= torch.optim.Adam(params, lr)

epoch = 1000
ers = []
for i in range(epoch):

    b = torch.randint(len(data)-ins, (100,))
    xs = torch.stack([data[i:i+ins] for i in b])
    ys = torch.stack([data[i+1:i+ins+1] for i in b])


    yh = model.forward(xs)
    loss = F.cross_entropy(yh.view(-1, vocab_size), ys.long().view(-1))
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # calculate loss function
    e = loss.item()
    if i % 500:
        print(e)
    ers.append(e)

plt.figure(1)
plt.plot(ers)

plt.figure(2)
plt.plot(ys)
plt.plot(torch.argmax(yh.detach(), dim=-1))


test_val = xs[0]

# yh =  model.forward(test_val)
# prob =  F.softmax(yh, dim=0)
# prob2 =  torch.softmax(yh * 0.7, dim=0).item()
# pred =  torch.argmax(yh, dim=0).item()
# print(itos[pred])

gen_text = ""
for i in range(3000):

    yh =  model.forward(test_val)
    prob =  F.softmax(yh[-1, :], dim=0)
    # pred = torch.argmax(yh).item()
    pred = torch.multinomial(prob, num_samples=1).item()

    test_val = torch.roll(test_val, -1)
    test_val[-1] = pred

    gen_text += itos[pred]


print(gen_text)

#|%%--%%| <bxqsAtaCbh|XcQ0XjiBB4>
r"""°°°
# Deep Learning + Blocks of Attention + Conveoluational Neural Network + AUTO Regression + batches + dropout regularion using pytorch with classes with gradient descent + non-linear + sin actvion function + multi-layer of nodes.
°°°"""
#|%%--%%| <XcQ0XjiBB4|waNszqrFGH>
with open("file.txt", 'r', encoding='utf-8') as f:
    text = f.read()

text = text.lower()
chars = sorted(list(set(text)))
stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}
data =  [stoi[c] for c in text]
vocab_size = len(chars)

ins = 64
outs = vocab_size
nodes = 200
lr = 0.001
n_emb = 64
embed = torch.randn(vocab_size, n_emb)
pos = torch.randn(ins, n_emb)

data = torch.tensor(data).long()
params = []
def weights(ins, outs):
    ws = torch.randn(ins, outs) *  0.1
    ws = ws.requires_grad_(True)
    params.append(ws)
    return ws


class Head():
    def __init__(self):
        self.wv = weights(n_emb, n_emb//4) # Diveded by number of heads for avoid expandding
        self.wq = weights(n_emb, n_emb//4) # Diveded by number of heads for avoid expandding
        self.wk = weights(n_emb, n_emb//4) # Diveded by number of heads for avoid expandding
    
    def forward(self, x):
        v = x @ self.wv
        q = x @ self.wq
        k = x @ self.wk

        attn = (q @ k.transpose(-2, -1)) / k.shape[0]**0.5
        tril = torch.tril(attn)
        tril = tril.masked_fill(tril==0, -1e10)
        rew = F.softmax(tril, dim=-1)
        x = rew @ v
        return x


class Block():

    def __init__(self):
        self.heads = [Head(), Head(), Head(), Head()]
        self.w0 = weights(n_emb, nodes)
        self.w1 = weights(nodes, n_emb)
 

    def forward(self, x):
        x = torch.cat([head.forward(x) for head in self.heads], dim=-1)
        x = torch.relu(x @ self.w0)
        x = torch.relu(x @ self.w1)
        return x




class Model():
    def __init__(self):
        self.blocks = [Block(), Block(), Block()]
        self.w2 = weights(n_emb, outs)

    def forward(self, x):
        x = embed[x] + pos 
        x = x + self.blocks[0].forward(x)
        x = x + self.blocks[1].forward(x)
        x = x + self.blocks[2].forward(x)

        yh = x @ self.w2
        return yh



model = Model()
optimizer= torch.optim.Adam(params, lr)

epoch = 1000
ers = []
for i in range(epoch):

    b = torch.randint(len(data)-ins, (100,))
    xs = torch.stack([data[i:i+ins] for i in b])
    ys = torch.stack([data[i+1:i+ins+1] for i in b])


    yh = model.forward(xs)
    loss = F.cross_entropy(yh.view(-1, vocab_size), ys.long().view(-1))
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # calculate loss function
    e = loss.item()
    if i % 500 == 0:
        print(e)
    ers.append(e)

plt.figure(1)
plt.plot(ers)

plt.figure(2)
plt.plot(ys)
plt.plot(torch.argmax(yh.detach(), dim=-1))


test_val = xs[0]

# yh =  model.forward(test_val)
# prob =  F.softmax(yh, dim=0)
# prob2 =  torch.softmax(yh * 0.7, dim=0).item()
# pred =  torch.argmax(yh, dim=0).item()
# print(itos[pred])

gen_text = ""
for i in range(3000):

    yh =  model.forward(test_val)
    prob =  F.softmax(yh[-1, :], dim=0)
    # pred = torch.argmax(yh).item()
    pred = torch.multinomial(prob, num_samples=1).item()

    test_val = torch.roll(test_val, -1)
    test_val[-1] = pred

    gen_text += itos[pred]


print(gen_text)

#|%%--%%| <waNszqrFGH|uvzHMpgkQO>
r"""°°°
# Deep Learning + Re-weight + Blocks of Attention + Conveoluational Neural Network + AUTO Regression + batches + dropout regularion using pytorch with classes with gradient descent + non-linear + sin actvion function + multi-layer of nodes.
°°°"""
#|%%--%%| <uvzHMpgkQO|afmjPKRWgP>
with open("file.txt", 'r', encoding='utf-8') as f:
    text = f.read()

text = text.lower()
chars = sorted(list(set(text)))
stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}
data =  [stoi[c] for c in text]
vocab_size = len(chars)

device = "cpu"
ins = 64
outs = vocab_size
nodes = 150
lr = 0.001
n_emb = 120
embed = torch.randn(vocab_size, n_emb)
pos = torch.randn(ins, n_emb)

embed = embed.to(device)
pos = pos.to(device)


data = torch.tensor(data).long()
params = []
def weights(ins, outs):
    ws = torch.randn(ins, outs) *  0.1
    ws = ws.to(device)
    ws = ws.requires_grad_(True)
    params.append(ws)
    return ws


class Head():
    def __init__(self):
        self.wv = weights(n_emb, n_emb//4) # Diveded by number of heads for avoid expandding
        # self.wq = weights(n_emb, n_emb//4) # Diveded by number of heads for avoid expandding
        # self.wk = weights(n_emb, n_emb//4) # Diveded by number of heads for avoid expandding
        self.wr = weights(n_emb, ins)
    def forward(self, x):
        v = x @ self.wv
        # q = x @ self.wq
        # k = x @ self.wk

        # attn = (q @ k.transpose(-2, -1)) / k.shape[0]**0.05
        # tril = torch.tril(att)
        re_weight =  x @ self.wr
        tril = torch.tril(re_weight)
        tril = tril.masked_fill(tril==0, -1e10)
        rew = F.softmax(tril, dim=-1)
        x = rew @ v
        return x


class Block():

    def __init__(self):
        self.heads = [Head(), Head(), Head(), Head()]
        self.w0 = weights(n_emb, nodes)
        self.w1 = weights(nodes, n_emb)
 

    def forward(self, x):
        x = torch.cat([head.forward(x) for head in self.heads], dim=-1)
        x = torch.relu(x @ self.w0)
        x = torch.relu(x @ self.w1)
        return x




class Model():
    def __init__(self):
        self.blocks = [Block(), Block(), Block()]
        self.w2 = weights(n_emb, outs)

    def forward(self, x):
        x = embed[x] + pos 
        x = x + self.blocks[0].forward(x)
        x = x + self.blocks[1].forward(x)
        x = x + self.blocks[2].forward(x)

        yh = x @ self.w2
        return yh



model = Model()
optimizer= torch.optim.Adam(params, lr)

epoch = 700
ers = []
for i in range(epoch):

    b = torch.randint(len(data)-ins, (100,))
    b = b.to(device)
    xs = torch.stack([data[i:i+ins] for i in b])
    ys = torch.stack([data[i+1:i+ins+1] for i in b])
    xs = xs.to(device)
    ys = ys.to(device)


    yh = model.forward(xs)
    loss = F.cross_entropy(yh.view(-1, vocab_size), ys.long().view(-1))
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # calculate loss function
    e = loss.item()
    # if i % 500 == 0:
    print(e)
    ers.append(e)

plt.figure(1)
plt.plot(ers)

plt.figure(2)
plt.plot(ys)
plt.plot(torch.argmax(yh.detach(), dim=-1))


test_val = xs[0]

# yh =  model.forward(test_val)
# prob =  F.softmax(yh, dim=0)
# prob2 =  torch.softmax(yh * 0.7, dim=0).item()
# pred =  torch.argmax(yh, dim=0).item()
# print(itos[pred])
gen_text = ""
for i in range(3000):

    yh =  model.forward(test_val)
    prob =  F.softmax(yh[-1, :]*0.7, dim=0)
    # pred = torch.argmax(yh).item()
    pred = torch.multinomial(prob, num_samples=1).item()

    test_val = torch.roll(test_val, -1)
    test_val[-1] = pred

    gen_text += itos[pred]


print("--------------[Generated text]--------------")
print(gen_text)
print("--------------[The text]--------------")
print(text)
