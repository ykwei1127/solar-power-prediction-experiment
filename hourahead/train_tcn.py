from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from utils import Dataset, energyDataset
from tcn import TCN

import pandas as pd, numpy as np, torch.nn.functional as F
import torch, os, random

base_path = os.path.dirname(os.path.abspath(__file__))

GPU           = 1
batch_size    = 64
hidden_units  = 64
dropout       = 0.1
epoches       = 300
ksize         = 3
levels        = 5
n_classes     = 1                  ### output dimensions
timesteps     = 24                 ### historic data size可以更改預測使用小時數
num_input     = 23                 ### feature dimensions
learning_rate = 0.02
seed          = 1

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.set_device(GPU)
torch.backends.cudnn.deterministic=True

base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "../data/solar")

dataset = Dataset(data_path, timesteps)
print("timesteps = ", timesteps)
# if timesteps != 24:
#     timesteps = timesteps + 1
train_data, train_target = np.array(dataset.train_data).astype(np.float32), np.array(dataset.train_target).astype(np.float32)
train_data = train_data.reshape(-1, num_input, timesteps)

trainloader = DataLoader(energyDataset(train_data, train_target), batch_size=batch_size, shuffle=True, num_workers=0)

channel_sizes = [hidden_units] * levels

model = TCN(num_input, n_classes, channel_sizes, kernel_size=ksize, dropout=dropout)
model = model.cuda()

optimizer = Adam(model.parameters(), lr=learning_rate)

pbar = tqdm(range(1, epoches+1))

model.train()

losses = list()

for epoch in pbar:
    if epoch % 100 == 0:
        learning_rate *= 0.7
        optimizer = Adam(model.parameters(), lr=learning_rate)

    train_loss = 0.
    for data, target in trainloader:
        data, target = data.cuda(), target.cuda()
        preds = model(data)
        loss = F.mse_loss(preds, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    train_loss /= len(trainloader.dataset)
    train_loss = round(np.sqrt(train_loss), 4)
    losses.append(train_loss)
    pbar.set_description(f"Epoch {epoch}, Loss: {train_loss}")

model.eval()

with torch.no_grad():
    train_data = torch.from_numpy(np.array(dataset.train_data).astype(np.float32)).float()
    train_target = np.array(dataset.train_target).astype(np.float32)
    print(train_data.shape)
    train_data = train_data.reshape(-1, num_input, timesteps)
    print(train_data.shape)
    train_data = train_data.cuda()
    train_preds = model(train_data)
    train_preds = train_preds.data.cpu().numpy()
    train_loss = round(np.sqrt(np.mean(np.square(train_target - train_preds))) * 100., 2)
    train_mae_loss = round(np.mean(np.abs(train_target - train_preds)) * 100., 2)
    test_data = torch.from_numpy(np.array(dataset.test_data).astype(np.float32)).float()
    test_target = np.array(dataset.test_target).astype(np.float32)
    test_data = test_data.reshape(-1, num_input, timesteps)
    test_data = test_data.cuda()
    test_preds = model(test_data)
    test_preds = test_preds.data.cpu().numpy()
    test_loss = round(np.sqrt(np.mean(np.square(test_target - test_preds))) * 100., 2)
    test_mae_loss = round(np.mean(np.abs(test_target - test_preds)) * 100., 2)

print (f"Train RMSE Loss: {train_loss}%, Test RMSE Loss: {test_loss}%")
print (f"Train NMAE Loss: {train_mae_loss}%, Test NMAE Loss: {test_mae_loss}%")

if not os.path.exists(os.path.join(base_path, "Output")):
    os.mkdir(os.path.join(base_path, "Output"))

pd.DataFrame({
    "loss": losses
}).plot()

plt.savefig(os.path.join(base_path, "Output/tcn_loss.png"))

print("Optimization Finished!")

pd.DataFrame({
    "predict": test_preds[:, 0],
    "target": test_target[:, 0]
}).plot()

plt.savefig(os.path.join(base_path, "Output/tcn_evaluation.png"))
