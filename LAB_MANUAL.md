## Prg 1 

```python
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 

img = plt.imread('cliff-edge.jpg')

r = img[:, :, 0] 
g = img[:, :, 1] 
b = img[:, :, 2] 

gray = (.299*r + .597*g + .114*b)

binary = np.where(gray > 127, 255, 0)
plt.imshow(img)
plt.title('normal')

plt.imshow(gray, cmap='gray')
plt.title('gray')
plt.show()

plt.imshow(binary, cmap='gray')
plt.title('binary')

perc_white = np.sum(binary==255)/binary.size
perc_white
```

## Prg 2 

```python
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 

img = np.ones((300,300, 3), np.uint8)*255
cv2.rectangle(img, (200, 200), (100,100), (255,0,0), 2)
plt.title('Orginal')
plt.imshow(img)

c , angle, tx, ty = (150,150), 45, 50, 30
m_rot = cv2.getRotationMatrix2D(c, angle, 1.0)
rot = cv2.warpAffine(img, m_rot, (300,300))
plt.imshow(rot)
plt.title('Rotated')

m_trans = np.float32([[1,0,tx], [0,1,ty]])
trans = cv2.warpAffine(img, m_trans, (300,300))
plt.title('Translated image')
plt.imshow(trans)

m_comb = m_rot.copy()
m_comb[:,2] += [tx,ty]
comb = cv2.warpAffine(img, m_comb, (300,300))
plt.title('Combined')
plt.imshow(comb)
``` 

## Prg 3 

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np 

img = np.zeros((200,200), dtype=np.uint8)
cv2.circle(img, (100,100), 70, 255, -1)
noisy = img.copy()
noise = np.random.random(img.shape)
noisy[noise < 0.2] = 0
noisy[noise > 0.98]  = 255

gaussian = cv2.GaussianBlur(noisy, (5,5), 0)
plt.imshow(gaussian, cmap='gray')

plt.imshow(noisy, cmap='gray')
median = cv2.medianBlur(noisy, 5)
plt.imshow(median, cmap='gray')
plt.imshow(img, cmap='gray')
```

## Prg 4 

```python
import torch
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt 

torch.manual_seed(42)
x = torch.randn(200,2)
y = (x[:,0] + x[:, 1]>0).float().view(-1,1)

class SimpleNN(nn.Module): 
    def __init__(self, a_fn):
        super().__init__()
        self.fc1 = nn.Linear(2,8)
        self.fc2 = nn.Linear(8,1) 
        self.a = a_fn
    
    def forward(self, x): 
        x = self.a(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)) 
        return x 

def train(l_r, a_fn, epoch = 50): 
    model = SimpleNN(a_fn)
    cr = nn.BCELoss()
    opt = optim.SGD(model.parameters(), lr = l_r) 
    
    losses = []
    
    for _ in range(epoch): 
        opt.zero_grad()
        
        ops = model(x)
        loss = cr(ops, y)
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
    
    return losses
    
l_s = 0.001 
l_m = 0.01
l_l = 0.1


plt.figure() 
plt.plot(train(l_s, nn.ReLU()), label="Small")
plt.plot(train(l_m, nn.ReLU()), label='Medium')
plt.plot(train(l_l, nn.ReLU()), label='large')
plt.legend()
plt.show()

plt.figure()
plt.plot(train(0.01, nn.ReLU()), label='Relu')
plt.plot(train(0.01, nn.Sigmoid()), label='Sigmoid')
plt.plot(train(0.01, nn.Tanh()), label='Tanh')
plt.legend()
plt.show()
```

## Prg 5 

```python
import numpy as np 

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x, d=False): 
    if d: 
        return x * (1-x)
    return 1 / (1+np.exp(-x))

def get_loss(op, y, l_t): 
    e = 1e-8
    if l_t=='mse': 
        return op - y
    elif l_t=='bse': 
        return (op-y)/((op+e)*(1-op+e))

def init_p(): 
    #42 state 
    np.random.seed(42)
    #its randn not rand  
    w1 = np.random.randn(2,4)*0.5 
    #its zeros 
    b1 = np.zeros((1,4))

    w2 = np.random.randn(4,1) * 0.5 
    b2 = np.zeros((1,1))
    return w1, b1, w2, b2
 
def train(x, y, lr=0.1, l_t='mse', epochs=500): 
    w1, b1, w2, b2 = init_p() 
    m = x.shape[0]

    for _ in range(epochs): 
        z1 = np.dot(x,w1) + b1 
        a1 = sigmoid(z1) 
        z2 = np.dot(a1, w2) + b2 
        a2 = sigmoid(z2)
    
        dz2 = get_loss(a2, y, l_t) * sigmoid(a2, True)
        dw2 = (1/m) * np.dot(a1.T, dz2) 
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
    
        da1 = np.dot(dz2, w2.T)
        dz1 = da1 * sigmoid(a1, True) 
        #x here not bloody a1 (back propagating) and its mult not addition 
        dw1 = (1/m) * np.dot(x.T, dz1) 
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
    
        w1 -= lr*dw1 
        w2 -= lr*dw2
        b1 -= lr*db1 
        b2 -= lr*db2
    return w1, b1, w2, b2
def get_acc(x,y,w1,b1,w2,b2): 
    z1 = np.dot(x,w1) + b1 
    a1 = sigmoid(z1) 
    z2 = np.dot(a1,w2) + b2 
    a2 = sigmoid(z2) 
    return np.mean((a2 > 0.5)==y)*100

for l in [0.01, 0.1, 1.0]: 
    w1, b1, w2, b2 = train(x,y, lr=l, l_t='mse')
    acc = get_acc(x,y,w1,b1,w2,b2)
    print(f"LR= {l}, Acc: {acc:.1f}%")

plt.figure()
plt.plot(train(0.01, nn.ReLU()), label='Relu')
plt.plot(train(0.01, nn.Sigmoid()), label='Sigmoid')
plt.plot(train(0.01, nn.Tanh()), label='Tanh')
plt.legend()
plt.show()
```

## Prg 6 

```python
import numpy as np 
import matplotlib.pyplot as plt 

def softmax(x): 
    x = np.array(x)
    exp = np.exp(x - np.max(x))
    return exp/(np.sum(exp))

def cce(y_t, y_p): 
    y_p = np.clip(y_p, 1e-12, 1-1e-12)
    return -np.sum(y_t * np.log(y_p))/y_t.shape[0]

def fwd(x,w): 
    y_p = softmax(np.dot(x,w))
    print(np.sum(y_p))
    return y_p 

def bwd(x, y_t, y_p, lr): 
    grad = cce(y_t, y_p)
    dw = np.dot(x.T, grad)/x.shape[0]
    return dw * lr 

x =  np.array([[1,2], [1.5,2.5], [2,3]])
w = np.array([[-0.5, 0.2, 0.3], [-0.6, 0.4, 0.7]])
y_t = np.array([[0,0,1], [0,1,0], [0,0,1]])

y_p = fwd(x,w) 
update = bwd(x,y_t, y_p, 0.1)
loss = cce(y_t, y_p)
print("Loss: ",loss) 
print("Updtaed: ", update)

x = [-2,-1,0,1,2]
plt.plot(x, softmax(x))
plt.grid()
plt.xlabel('Logits')
plt.ylabel('Probs')
```