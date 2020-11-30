import pandas as pd, numpy as np
import torch, os, random

from tqdm import tqdm
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
from nbeats import NBeatsNet
from utils import Dataset, energyDataset
from matplotlib import pyplot as plt


device             = torch.device("cuda")
GPU                = 0
batch_size         = 64
hidden_units       = 64
n_blocks           = 5
epoches            = 300
output_dims        = 2           ### output dimensions
timesteps          = 24          ### historic data size
num_input          = 1           ### feature dimensions
learning_rate      = 0.02
seed               = 0

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.set_device(GPU)
torch.backends.cudnn.deterministic=True

base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "../data/solar")

dataset = Dataset(data_path, timesteps)

train_data, train_target = np.array(dataset.train_data).astype(np.float32), np.array(dataset.train_target).astype(np.float32)

trainloader = DataLoader(energyDataset(train_data[:, :, 5], np.reshape(train_target[:, 1], (-1, 1))), batch_size=batch_size, shuffle=True, num_workers=0)

model = NBeatsNet(device=device,
                  stack_types=[NBeatsNet.TREND_BLOCK, NBeatsNet.GENERIC_BLOCK, NBeatsNet.SEASONALITY_BLOCK],
                  forecast_length=1,
                  thetas_dims=[2, 8, 3],
                  nb_blocks_per_stack=n_blocks,
                  backcast_length=timesteps,
                  hidden_layer_units=hidden_units,
                  share_weights_in_stack=False,
                  nb_harmonics=None)

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
        preds = model(data)[1]
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
    train_data = torch.from_numpy(np.array(dataset.train_data).astype(np.float32)).float()[:, :, 5]
    train_target = np.array(dataset.train_target).astype(np.float32)
    train_data = train_data.reshape(-1, timesteps)
    train_data = train_data.cuda()
    train_preds = model(train_data)
    train_preds = train_preds[1].data.cpu().numpy()
    train_loss = round(np.sqrt(np.mean(np.square(train_target[:, 1] - train_preds[:, 0]))) * 100., 2)
    test_data = torch.from_numpy(np.array(dataset.test_data).astype(np.float32)).float()[:, :, 5]
    test_target = np.array(dataset.test_target).astype(np.float32)
    test_data = test_data.reshape(-1, timesteps)
    test_data = test_data.cuda()
    test_preds = model(test_data)
    test_preds = test_preds[1].data.cpu().numpy()
    test_loss = round(np.sqrt(np.mean(np.square(test_target[:, 1] - test_preds[:, 0]))) * 100., 2)

print (f"Train RMSE Loss: {train_loss}%, Test RMSE Loss: {test_loss}%")

if not os.path.exists(os.path.join(base_path, "Output")):
    os.mkdir(os.path.join(base_path, "Output"))

pd.DataFrame({
    "loss": losses
}).plot()

plt.savefig(os.path.join(base_path, "Output/nbeats_loss.png"))

print("Optimization Finished!")

dataframe = pd.DataFrame({
    "predict": test_preds[:, 0],
    "target": test_target[:, 1]
})
dataframe.to_csv("rad.csv", index=False, header=None)

pd.DataFrame({
    "predict": test_preds[:, 0],
    "target": test_target[:, 1]
}).plot()

plt.savefig(os.path.join(base_path, "Output/nbeats_evaluation.png"))