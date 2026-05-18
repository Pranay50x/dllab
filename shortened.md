## Program 7 

```python

import torch, torchvision
import torch.nn as nn, torch.optim as optim
from torchvision import transforms as T
from torch.utils.data import DataLoader, Subset

t = T.ToTensor()
trd = DataLoader(Subset(torchvision.datasets.CIFAR10('./data', True,  transform=t, download=True), range(200)), 10, True)
ted = DataLoader(Subset(torchvision.datasets.CIFAR10('./data', False, transform=t, download=True), range(50)),  10)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(4096,512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512,10)
        )
    def forward(self, x): return self.net(x)
def train(model, opt, loss_fn, epochs):
    for e in range(epochs):
        model.train(); tl=tc=tt=0
        for x,y in trd:
            opt.zero_grad()
            out=model(x)
            l=loss_fn(out,y)
            l.backward()
            opt.step()
            tl+=l.item()
            tc+=(out.argmax(1)==y).sum().item()
            tt+=len(y)
        model.eval()
        vl=vc=vt=0
        with torch.no_grad():
            for x,y in ted:
                out=model(x); 
                l=loss_fn(out,y)
                vl+=l.item();
                vc+=(out.argmax(1)==y).sum().item(); 
                vt+=len(y)
        print(f"Epoch{e+1} | Train Loss: {tl/len(trd):.2f} Accuracy: {tc/tt*100:.1f}% | Test Loss: {vl/len(ted):.2f} Accuracy: {vc/vt*100:.1f}%")

model = CNN()
train(model, optim.Adam(model.parameters(), 1e-3), nn.CrossEntropyLoss(), 30)
```

## Program 8

```python
import torch, torchvision
import torch.nn as nn, torch.optim as optim
from torchvision import transforms as T
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

t = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trd = DataLoader(torchvision.datasets.CIFAR10('./data',True, transform=t,download=True),64,True)
ted = DataLoader(torchvision.datasets.CIFAR10('./data',False,transform=t,download=True),64)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(4096,128), nn.ReLU(), nn.Linear(128,10)
        )
    def forward(self,x): return self.net(x)

def train(model, opt, crit, epochs=5):
    trl,tel,acl = [],[],[]
    for e in range(epochs):
        model.train(); rl=0
        for x,y in trd:
            opt.zero_grad(); o=model(x); l=crit(o,y); l.backward(); opt.step(); rl+=l.item()
        model.eval(); tl=c=n=0
        with torch.no_grad():
            for x,y in ted:
                o=model(x); l=crit(o,y); tl+=l.item(); c+=(o.argmax(1)==y).sum().item(); n+=len(y)
        tr,te,ac = rl/len(trd), tl/len(ted), c/n
        trl.append(tr); tel.append(te); acl.append(ac)
        print(f"E{e+1} | Train {tr:.4f} | Test {te:.4f} | Acc {ac:.2f}")
    return trl,tel,acl

crit = nn.CrossEntropyLoss()
ms,ma = CNN(),CNN()
print("SGD");  trs,tes,acs = train(ms, optim.SGD(ms.parameters(),0.01,momentum=0.9), crit)
print("Adam"); tra,tea,aca = train(ma, optim.Adam(ma.parameters(),1e-3),              crit)

ep = range(1,6)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(ep,trs,'b--',label='SGD Train'); plt.plot(ep,tes,'b-',label='SGD Test')
plt.plot(ep,tra,'r--',label='Adam Train'); plt.plot(ep,tea,'r-',label='Adam Test')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend()
plt.subplot(1,2,2)
plt.plot(ep,acs,label='SGD'); plt.plot(ep,aca,label='Adam')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy'); plt.legend()
plt.tight_layout(); plt.show()
```

## Program 9

```python
import torch, torchvision
import torch.nn as nn, torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

t = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
trd = DataLoader(torchvision.datasets.MNIST('./data', True,  transform=t, download=True), 64, True)
ted = DataLoader(torchvision.datasets.MNIST('./data', False, transform=t, download=True), 64)


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,6,5), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(6,16,5), nn.ReLU(), nn.AvgPool2d(2),
            nn.Flatten(), nn.Linear(400,120), nn.ReLU(),
            nn.Linear(120,84), nn.ReLU(), nn.Linear(84,10)
        )
    def forward(self, x):
        return self.net(x)

model = LeNet5()
crit = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.001)


for e in range(5):
    model.train()
    rl = 0
    for x, y in trd:
        opt.zero_grad()
        o = model(x)
        l = crit(o, y)
        l.backward()
        opt.step()
        rl += l.item()
    print(f"E{e+1} | Loss: {rl/len(trd):.4f}")


model.eval()
c = n = 0
with torch.no_grad():
    for x, y in ted:
        o = model(x)
        c += (o.argmax(1)==y).sum().item()
        n += len(y)
print(f"Test Accuracy: {100*c/n:.2f}%")
```

## Program 10: 

```python
import torch
import torch.nn as nn 
import torchvision
import torch.optim as optim 
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224,224)), 
    transforms.Grayscale(3), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5, ), (0.5, ))
])

train_loader = DataLoader(torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(torchvision.datasets.MNIST(root='./data', download=True, train=False, transform=transform), batch_size=1000)

class AlexNet(nn.Module): 
    def __init__(self):
        super().__init__()
        self.feats = nn.Sequential(
            nn.Conv2d(3, 64, 11, 4, 2), nn.ReLU(), nn.MaxPool2d(3,2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3,2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(256,256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3,2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(256*6*6, 4096), 
            nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(), 
            nn.Linear(4096,10)
        )
    
    def forward(self, x): 
        x =  self.feats(x)
        x =  x.view(x.size(0), -1)
        x = self.classifier(x)
        return x 

device = torch.device("cpu")
net = AlexNet().to(device)
crit = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr=0.001)

for e in range(1): 
    net.train()
    r_loss = 0
    
    for i, (im, la) in enumerate(train_loader): 
        
        im, la = im.to(device), la.to(device)
        
        opt.zero_grad()
        loss = crit(net(im), la)
        loss.backward()
        
        opt.step()
        
        r_loss += loss.item()
        
        if i % 100==0: 
            print(f"Epoch: {e+1}, Batch: {i+1}, loss: {r_loss:.3f}")
            r_loss = 0 

net.eval()
corr = 0
tot = 0 

with torch.no_grad():
    for im, lab in test_loader:
        im, la = im.to(device), la.to(device)
        tot += la.size(0)
        corr += (net(im).argmax(1)==la).sum().item()

print(f"Final Acc: {100*(corr/tot):.2f}")
```

## Program 11: 

```python
import torch, torch.nn as nn

# 1. Corpus & Simple Vocabulary
corpus = ["the movie was fantastic", "the movie was boring", "i loved the acting", "i hated the plot", "the plot was amazing", "the acting was terrible"]
words = sorted(list(set(" ".join(corpus).split())))
w2i = {w: i+1 for i, w in enumerate(words)} | {"<pad>": 0}
i2w = {i: w for w, i in w2i.items()}
vocab_size, max_len = len(w2i), max(len(line.split()) for line in corpus)

# 2. Build N-grams & Padding
X, y = [], []
for line in corpus:
    tokens = [w2i[w] for w in line.split()]
    for i in range(1, len(tokens)):
        seq = [0] * (max_len - len(tokens[:i+1])) + tokens[:i+1] # Left-padding
        X.append(seq[:-1])
        y.append(seq[-1])

X_train, y_train = torch.tensor(X), torch.tensor(y)

# 3. Clean Vanilla RNN Model
class TextModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.rnn = nn.RNN(32, 32, batch_first=True) # Straightforward vanilla RNN
        self.fc = nn.Linear(32, vocab_size)
        
    def forward(self, x):
        _, h = self.rnn(self.embed(x))
        return self.fc(h[-1]) # h[-1] automatically grabs the final hidden state

# 4. Training
model = TextModel(vocab_size)
optimizer, criterion = torch.optim.Adam(model.parameters(), lr=0.01), nn.CrossEntropyLoss()

for epoch in range(300):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0: print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5. Simple Predict Function
def predict(seed):
    tokens = [w2i.get(w, 0) for w in seed.lower().split()]
    padded = [0] * (max_len - 1 - len(tokens)) + tokens
    with torch.no_grad():
        pred = model(torch.tensor([padded])).argmax(1).item()
    return i2w.get(pred, "<unknown>")

print("the movie was ->", predict("the movie was"))
print("i loved the ->", predict("i loved the"))
```

## Program 12: 

```python
import torch, torch.nn as nn

# 1. Vocabulary & Data Setup
text = "hello world, this is a simple text generation using LSTMs."
chars = sorted(list(set(text)))
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for i, c in enumerate(chars)}

X, y = [], []
for i in range(len(text) - 10):
    X.append([c2i[c] for c in text[i:i+10]])
    y.append(c2i[text[i+10]])
X_train, y_train = torch.tensor(X), torch.tensor(y)

# 2. Clean LSTM Architecture
class TextLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 16)
        self.lstm = nn.LSTM(16, 128, batch_first=True)
        self.fc = nn.Linear(128, vocab_size)

    def forward(self, x):
        _, (h, _) = self.lstm(self.embed(x))
        return self.fc(h[-1]) # h[-1] automatically gets the final state without tensor slicing

# 3. Training
model = TextLSTM(len(chars))
optimizer, criterion = torch.optim.Adam(model.parameters(), lr=0.01), nn.CrossEntropyLoss()

for epoch in range(50):
    optimizer.zero_grad()
    criterion(model(X_train), y_train).backward()
    optimizer.step()

# 4. Smart Text Generation
def generate(seed, length=50):
    model.eval()
    curr_seq = [c2i[c] for c in seed]
    for _ in range(length):
        with torch.no_grad():
            # Grabs the last 10 historical characters dynamically
            pred = model(torch.tensor([curr_seq[-10:]])).argmax(1).item() 
        seed += i2c[pred]
        curr_seq.append(pred)
    return seed

print(generate("hello world", 50))
```
 