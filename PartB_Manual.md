## Program 7: 

```python
import torch
import torch.nn as nn 
import torch.optim as optim 
import torchvision 
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, Subset

class CNNDropout(nn.Module): 
    def __init__(self): 
        super(CNNDropout, self).__init__()

    #first block for eyeing the image 
        self.convb1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        
        #second block looks deeper same as the prev one but looks for 32 more feats 
        self.convb2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        
        self.flatten = nn.Flatten()
        #4096 layers condensed to 512 layers
        self.dense1 = nn.Linear(64*8*8, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        #final layer condenses the 512 to 10 as to what is to be predicted in CIFAR 10 Dataset
        self.dense2 = nn.Linear(512, 10)
    
    def forward(self, x): 
        x = self.convb1(x)
        x = self.convb2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x 
    
#img converted to pytorch sensors
transform = transforms.Compose([transforms.ToTensor()])
train_d = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_d = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_s = Subset(train_d, range(200))
test_s = Subset(test_d, range(50))

train_l = DataLoader(train_s, batch_size=10, shuffle= True)
test_l = DataLoader(test_s, batch_size=10, shuffle= False)

def train(model, opt, crit,  n_e, device): 
    model.to(device)
    
    for e in range(n_e): 
        model.train()
        tr_l = 0
        c_t = 0 
        t_t = 0 
        
        for d, t in train_l: 
            d, t = d.to(device), t.to(device)
            #remove all predictions 
            opt.zero_grad()
            #make new predictions
            op = model(d)
            loss = crit(op,t)
            #go backward in the nueral net to see where it went wrong 
            loss.backward()
            #adjust and make changes to optimizer 
            opt.step()
            #calculation for how many images the model got right 
            tr_l += loss.item()
            pred = torch.argmax(op.data, dim = 1)
            t_t += t.size(0)
            c_t += (pred == t).sum().item()
            
        model.eval()
        t_l = 0
        c_te = 0 
        tot_t = 0 
        
        #this no grad just tells to predict and not to update any gradients 
        with torch.no_grad(): 
            for d, t in test_l: 
                d, t = d.to(device), t.to(device)
                op = model(d)
                loss = crit(op, t)
                
                t_l += loss.item()
                tot_t += t.size(0)
                pred = torch.argmax(op.data , dim=1)
                c_te += (pred==t).sum().item()

        avg_tr_l = tr_l / len(train_l)
        tr_acc = (c_t / t_t) * 100 
        avg_te_l = t_l / len(test_l)
        tes_acc = (c_te/tot_t) * 100 
        print(f"Epoch: {e+1}, Train loss: {avg_tr_l:.4f}, Train Acc: {tr_acc:.2f}, Test Loss: {avg_te_l:.4f}, Test Acc: {tes_acc:.2f}")
            
device = torch.device("cpu")
model = CNNDropout()
crit = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr =0.001)
train(model, opt, crit, 30,device)
```

## Program 8: 

```python
import torch
import torchvision
import torch.nn as nn 
import torch.optim as optim 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, Subset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_d = torchvision.datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)
test_d = torchvision.datasets.CIFAR10(root='./data', download=True, train=False, transform=transform)

train_loader = DataLoader(train_d, batch_size=64, shuffle=True)
test_loader = DataLoader(test_d, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self): 
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        #flattening of the data without using nn.Flatten 
        x = x.view(-1, 64*8*8)
        #apply relu only to the hidden layer ie fc1 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
        

def train_E(model, crit, opt, epochs=5): 
    tr_loss, te_loss, te_accur = [],[],[]
    
    
    for e in range(epochs): 
        model.train()
        r_loss = 0 
        for im, la in train_loader: 
            opt.zero_grad()
            op = model(im)
            loss = crit(op, la)
            loss.backward()
            opt.step()
            r_loss += loss.item()
        
        model.eval()
        test_loss, corr, total = 0,0,0
        
        with torch.no_grad(): 
            for im, la in test_loader: 
                
                op = model(im)
                loss = crit(op, la)
                test_loss += loss.item()
                _, pred = torch.max(op, 1)
                corr += (pred==la).sum().item()
                total += la.size(0)
        
        avg_tr_loss = r_loss / len(train_loader)
        tr_loss.append(avg_tr_loss)
        avg_te_loss = test_loss / len(test_loader)
        te_loss.append(avg_te_loss)
        acc = corr / total
        te_accur.append(acc)
        print(f"Epoch: {e+1}, Train Loss: {avg_tr_loss:.4f}, Test Loss: {avg_te_loss:.4f}, Acc: {acc:.2f}")
    
    return tr_loss, te_loss, te_accur

m_sgd = SimpleCNN()
m_adam = SimpleCNN()

crit = nn.CrossEntropyLoss()

optim_s = optim.SGD(m_sgd.parameters(), lr=0.01, momentum=0.9)
optim_a = optim.Adam(m_adam.parameters(), lr=0.001)
print("Train with sgd")
train_s, te_s, a_s = train_E(m_sgd, crit, optim_s)
print("Train with adam")
train_a, te_a, a_a = train_E(m_adam, crit, optim_a)

epoch = range(1,6)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epoch, train_s, label='Sgd train loss')
plt.plot(epoch, te_s, label='Sgd test loss')
plt.plot(epoch, train_a, label='Adam train loss')
plt.plot(epoch, te_a, label='Adam test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('loss comp')

plt.subplot(1,2,2)
plt.plot(epoch, a_s, label='SGD Acc')
plt.plot(epoch, a_a, label='Adam Acc')
plt.title('Acc comp')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.show()
```

## Program 9: 

```python
import torch 
import torchvision
import torch.nn as nn 
import torch.optim as opt 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

class Lenet(nn.Module): 
    def __init__(self): 
        super(Lenet, self).__init__()
        #only white or black for mnist so 1 and searches for 6 basic feats 
        self.convb1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(2,2)
        self.convb2= nn.Conv2d(6, 16, kernel_size=5)
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x): 
        #using the torch relu directly instead of sep var 
        x = self.pool(torch.relu(self.convb1(x)))
        x = self.pool(torch.relu(self.convb2(x)))
        #flattening again 16*5*5 to the batch size 
        x = x.view(x.size(0), -1)
        #same as the prev one apply relu to all layers except output layer
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x         

model = Lenet()
crit = nn.CrossEntropyLoss()
opt = opt.Adam(model.parameters(), lr = 0.001)

#process only 32x32 images dataset has 28x28
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])
train_d = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=transform)
test_d = torchvision.datasets.MNIST(root='./data', download=True, train= False ,transform=transform)

train_loader = DataLoader(train_d, batch_size=64, shuffle=True)
test_loader = DataLoader(train_d, batch_size=64, shuffle=False)

for e in range(5):
    r_loss = 0 
    
    for im, la in train_loader: 
        
        opt.zero_grad()
        
        op = model(im)
        loss = crit(op, la)
        
        loss.backward()
        opt.step()
        
        r_loss += loss.item()
    
    print(f"Epoch: {e+1}, Training loss: {r_loss/len(train_loader)}")        

corr = 0 
tot = 0 

with torch.no_grad(): 
    for im, la in test_loader: 
        op = model(im) 
        _, pred = torch.max(op.data, 1)
        corr += (pred==la).sum().item()
        tot += la.size(0)

acc = 100* (corr/tot)
print(f"Test accuracy: {acc:.2f}") 
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