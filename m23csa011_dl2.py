# -*- coding: utf-8 -*-
"""m23csa011-DL2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vN7WdkcFQAS1W0_nHf0UgIjjC3hvxQT1
"""

from google.colab import drive
drive.mount('/content/drive')

try:
    import torchinfo
except:
    !pip install torchinfo
    import torchinfo

!pip install scikit-plot

! pip install wandb

! wandb login c07f9b9363b1d2736cf24c01a4747e245909f2cc

import os
if not os.path.exists("/content/audio"):
    ! unzip /content/drive/MyDrive/Archive.zip
else:
    print("Audio folder already exists. Skipping unzipping.")

!pip install lightning

import os
import random
import numpy as np
from pathlib import Path
import pandas as pd
import torchaudio
import zipfile
from torchaudio.transforms import Resample
import IPython.display as ipd
from matplotlib import pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import Dataset , DataLoader , random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score , roc_curve , auc
import wandb
# import scikitplot as skplt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from torchinfo import summary
import math

path = Path('/content')
df = pd.read_csv('/content/meta/esc50.csv')

wavs = list(path.glob('audio/*'))
waveform, sample_rate = torchaudio.load(wavs[5])

print("Shape of waveform {}".format(waveform.size()))

plt.figure()
plt.plot(waveform.t().numpy())

"""# Setting random seed"""

def set_random_seed(seed_val=42):
  np.random.seed(seed_val)
  random.seed(seed_val)
  torch.manual_seed(seed_val)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_val)

set_random_seed(42)

results = {
    'model_name': [],
    'train_acc': [],
    'val_acc': [],
    'test_acc': []
}

"""# Custom Dataset"""

class CustomDataset(Dataset):
  def __init__(self,dataset,**kwargs):
    self.data_directory = kwargs.get("data_directory")
    self.data_frame = kwargs.get("data_frame")
    self.validation_fold = kwargs.get("validation_fold")
    self.testing_fold = kwargs.get("testing_fold")
    self.esc_10_flag = kwargs.get("esc_10_flag")
    self.file_column = kwargs.get("file_column")
    self.label_column = kwargs.get("label_column")
    self.sampling_rate = kwargs.get("sampling_rate")
    self.new_sampling_rate = kwargs.get("new_sampling_rate")
    self.sample_length_seconds = kwargs.get("sample_length_seconds")

    if self.esc_10_flag:
      self.data_frame = self.data_frame.loc[self.data_frame['esc10']]

    if dataset == 'train':
      self.data_frame = self.data_frame.loc[(self.data_frame['fold']!=self.validation_fold)&(self.data_frame['fold']!=self.testing_fold)]
    elif dataset == 'test':
      self.data_frame = self.data_frame.loc[self.data_frame['fold']==self.testing_fold]
    elif dataset == 'val':
      self.data_frame = self.data_frame.loc[self.data_frame['fold']==self.validation_fold]

    self.categories = sorted(self.data_frame[self.label_column].unique())

    self.file_names , self.labels = [] , []

    self.category_to_index , self.index_to_category = {} , {}

    for i,category in enumerate(self.categories):
      self.category_to_index[category] = i
      self.index_to_category[i] = category

    for ind in tqdm(range(len(self.data_frame))):
      row = self.data_frame.iloc[ind]
      file_path = self.data_directory/"audio"/row[self.file_column]
      self.file_names.append(file_path)
      self.labels.append(self.category_to_index[row[self.label_column]])

    if self.sample_length_seconds != 2 :
      self.window_size = self.new_sampling_rate
      self.step_size = int(self.new_sampling_rate*0.5)
    else:
      self.window_size = self.new_sampling_rate*2
      self.step_size = int(self.new_sampling_rate*0.75)

    self.resampler = torchaudio.transforms.Resample(self.sampling_rate,self.new_sampling_rate)

  def __getitem__(self, index):
        path = self.file_names[index]
        audio_file = torchaudio.load(path, format=None, normalize=True)
        audio_tensor = self.resampler(audio_file[0])
        splits = audio_tensor.unfold(1, self.window_size, self.step_size)
        samples = splits.permute(1, 0, 2)
        return samples, self.labels[index]

  def __len__(self):
    return len(self.file_names)

"""# Custom DataLaoder"""

class CustomDataModule(pl.LightningDataModule):
  def __init__(self,**kwargs):
    super().__init__()
    self.batch_size = kwargs.get("batch_size")
    self.num_workers = kwargs.get('num_workers')
    self.data_module_kwargs = kwargs

  def setup(self,stage=None):

    if stage=='test' or stage is None:
      self.testing_dataset = CustomDataset(dataset='test',**self.data_module_kwargs)

    if stage == "fit" or stage is None:
      self.validation_dataset = CustomDataset(dataset='val',**self.data_module_kwargs)
      self.training_dataset = CustomDataset(dataset='train',**self.data_module_kwargs)

  def test_dataloader(self):
    return DataLoader(self.testing_dataset,
                      batch_size=32,
                      shuffle=False,
                      collate_fn=self.custom_collate_function,
                      num_workers=self.num_workers)

  def val_dataloader(self):
    return DataLoader(self.validation_dataset,
                      batch_size=self.batch_size,
                      shuffle=False,
                      collate_fn=self.custom_collate_function,
                      num_workers=self.num_workers)

  def train_dataloader(self):
    return DataLoader(self.training_dataset,
                      batch_size=self.batch_size,
                      shuffle=True,
                      collate_fn=self.custom_collate_function,
                      num_workers=self.num_workers)

  def custom_collate_function(self,data):
    examples,labels = zip(*data)
    examples = torch.stack(examples)
    examples = examples.reshape(examples.size(0),1,-1)
    labels = torch.flatten(torch.tensor(labels))
    return [examples,labels]

test_samp = 1
valid_samp = 2
batch_size = 32
num_workers = 2

custom_data_module = CustomDataModule(batch_size=batch_size,
                                       num_workers=num_workers,
                                       data_directory=path,
                                       data_frame=df,
                                       validation_fold=valid_samp,
                                       testing_fold=test_samp,
                                       esc_10_flag=True,
                                       file_column='filename',
                                       label_column='category',
                                       sampling_rate=44100,
                                       new_sampling_rate=16000,
                                       sample_length_seconds=1)

custom_data_module.setup()

print('Class Label: ',custom_data_module.training_dataset[0][1])
print('Shape of data sample tensor: ', custom_data_module.training_dataset[0][0].shape)

x = next(iter(custom_data_module.train_dataloader()))
features, labels = x
print("Batch of features shape:", features.shape)
print("Batch of labels shape:", labels.shape)

"""# Architecture 1 - CNN Base"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride , padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride , padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
#         x = self.batchnorm(x)
        return x

class FCBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class Conv1DNet(nn.Module):
    def __init__(self, num_classes):
        super(Conv1DNet, self).__init__()

        self.conv_block1 = ConvBlock(1, 64, kernel_size=7, stride=2 , padding=2)
        self.conv_block2 = ConvBlock(64, 32, kernel_size=5, stride=2 ,  padding=2)
        self.conv_block3 = ConvBlock(32, 16, kernel_size=3, stride=2 ,  padding=2)

        self.fc_block1 = FCBlock(36000, 128)
        self.fc_block2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.dropout(x)
        x = self.conv_block2(x)
        x = self.dropout(x)
        x = self.conv_block3(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.fc_block1(x)
        x = self.fc_block2(x)

        return x


num_classes = 10
model = Conv1DNet(num_classes)

summary(model, input_size=[32, 1, 144000])

"""# Parameters for CNN Model"""

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

print(f"Total Trainable parameters {trainable_params}")
print(f"Total Non-Trainable parameters {non_trainable_params}")

"""# Train and evaluate Function"""

def train(model, criterion, optimizer, dataloader, device,epoch):
  model.train()
  total_loss = 0.0
  total_correct = 0
  total_samples = 0

  for (inputs,targets) in tqdm(dataloader,desc=f'Epoch {epoch+1}/{epochs}'):
    inputs = inputs.to(device)
    targets = targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs,targets)
    loss.backward()
    optimizer.step()

    total_loss += loss.item() * inputs.size(0)
    _, predicted = torch.max(outputs, 1)
    total_correct += (predicted == targets).sum().item()
    total_samples += inputs.size(0)

  epoch_loss = total_loss / len(dataloader.dataset)
  epoch_acc = 100*total_correct / total_samples

  return epoch_loss, epoch_acc

def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += inputs.size(0)

    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = 100*total_correct / total_samples

    return epoch_loss, epoch_acc

def run_train_eval(model,num_classes,optimizer,criterion,custom_data_module,epochs):
  best_valid_loss = np.inf

  for epoch in range(epochs):
    train_loss, train_acc = train(model, criterion, optimizer, custom_data_module.train_dataloader(), device,epoch)
    val_loss, val_acc = validate(model, criterion, custom_data_module.val_dataloader(), device)

    wandb.log({'Epoch': epoch+1, 'Train Loss': train_loss, 'Train Accuracy': train_acc,
               'Val Loss': val_loss, 'Val Accuracy': val_acc})

    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, ' +
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

  return train_acc , val_acc

  wandb.finish()

"""# Test Function"""

def run_test(model,num_classes,custom_data_module):
  test_predictions = []
  test_labels = []
  test_probs = []

  model.eval()
  with torch.no_grad():
    for (inputs,targets) in tqdm(custom_data_module.test_dataloader()):
      inputs = inputs.to(device)
      targets = targets.to(device)
      outputs = model(inputs)
      _,preds = torch.max(outputs,dim=1)
      probs = nn.Softmax(dim=1)(outputs).cpu().numpy()
      test_predictions.extend(preds.tolist())
      test_labels.extend(targets.tolist())
      test_probs.extend(probs)

  test_predictions = np.array(test_predictions)
  test_labels = np.array(test_labels)

  accuracy = 100*accuracy_score(test_predictions,test_labels)

  conf_matrix = confusion_matrix(test_labels,test_predictions)

  f1 = f1_score(test_labels,test_predictions,average='macro')

  fpr = dict()
  tpr = dict()
  roc_auc = dict()

  auc_roc_scores = []
  for i in range(num_classes):
        class_labels = (np.array(test_labels) == i).astype(int)
        class_probs = np.array(test_probs)[:, i]
        auc_roc = roc_auc_score(class_labels, class_probs)
        auc_roc_scores.append(auc_roc)
        fpr[i], tpr[i], _ = roc_curve(class_labels, class_probs)
        roc_auc[i] = auc(fpr[i], tpr[i])

  print("")
  print(f"Overall Accuracy: {accuracy}")
  print("")
  print(f"Overall F1-Score: {f1}")
  print("")

  print("Class-wise Metrics:")
  for i, auc_roc in enumerate(auc_roc_scores):
      class_accuracy = accuracy_score(test_labels[test_labels == i], test_predictions[test_labels == i])
      class_f1 = f1_score(test_labels[test_labels == i], test_predictions[test_labels == i],average='macro')
      print(f"Class {i}: Accuracy={class_accuracy}, F1-Score={class_f1}, AUC-ROC={auc_roc}")

  print("")

  plt.figure(figsize=(10, 8))
  for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve (Class {i}) (area = {roc_auc[i]:.2f})')

  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve for all Classes')
  plt.legend(loc='lower right')
  plt.show()

  sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.title('Confusion Matrix')
  plt.show()

#   skplt.metrics.plot_roc(np.array(test_labels),np.array(test_probs))
#   plt.title('ROC Curve')

#   plt.show()

  return accuracy

"""# Train 100 epochs CNN"""

lr = 0.001
epochs = 100
batch_size = 64
num_classes = 10
num_folds = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()

dummy_input = torch.randn(1, 1, 144000)

model = model.to(device)
dummy_input = dummy_input.to(device)

output = model(dummy_input)

print("Output shape:", output.shape)

wandb.init(project='DL_assign2_CNN_task_1')

model = Conv1DNet(num_classes)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs=100
train_acc , val_acc = run_train_eval(model,num_classes,optimizer,criterion,custom_data_module,epochs)
test_acc = run_test(model,num_classes,custom_data_module)

"""# K-Fold Cross validation CNN"""

def run_KFold():
    k = 6
    epochs=100
    test_samp = 1
    all_fold_accuracies = []
    model = {}
    for fold_idx in range(2, k):
        model[fold_idx] = Conv1DNet(num_classes)
        model[fold_idx] = model[fold_idx].to(device)
        valid_samp = fold_idx
        batch_size = 32
        num_workers = 2
        optimizer = optim.Adam(model[fold_idx].parameters(), lr=lr)
        wandb.init(project=f'DL_assign2_cnn_task_2_fold_{fold_idx}')

        custom_data_module = CustomDataModule(batch_size=batch_size,
                                              num_workers=num_workers,
                                              data_directory=path,
                                              data_frame=df,
                                              validation_fold=valid_samp,
                                              testing_fold=test_samp,
                                              esc_10_flag=True,
                                              file_column='filename',
                                              label_column='category',
                                              sampling_rate=44100,
                                              new_sampling_rate=16000,
                                              sample_length_seconds=1)

        print(f"Setting up datasets for fold {fold_idx}")
        print("")
        custom_data_module.setup()
        print("")

        train_accuracy,val_accuracy = run_train_eval(model[fold_idx], num_classes, optimizer, criterion, custom_data_module,epochs)
        # test_accuracy = run_test(model, num_classes, custom_data_module)

        all_fold_accuracies.append(val_accuracy)

    avg_accuracy = sum(all_fold_accuracies) / len(all_fold_accuracies)
    print("")
    print(f"Average accuracy over {k-2} folds: {avg_accuracy}")

run_KFold()

"""# Run Hyperparameter Tuning CNN

"""

def hyperparameter_tuning_cnn():
  learning_rates = [0.001,0.01]
  batch_sizes = [32, 64]
  epochs = 100
  best_val_accuracy = 0.0
  best_hyperparameters = {}
  for lr in learning_rates:
    for batch_size in batch_sizes:
      wandb.init(project=f'DL_assign2_CNN_task_5_{lr}_{batch_size}')
      print("")
      print(f"Combination: Learning_rate : {lr} , batch_szie : {batch_size}")
      print("")
      model = Conv1DNet(num_classes)
      model = model.to(device)
      optimizer = optim.Adam(model.parameters(), lr=lr)
      custom_data_module = CustomDataModule(batch_size=batch_size,
                                          num_workers=num_workers,
                                          data_directory=path,
                                          data_frame=df,
                                          validation_fold=valid_samp,
                                          testing_fold=test_samp,
                                          esc_10_flag=True,
                                          file_column='filename',
                                          label_column='category',
                                          sampling_rate=44100,
                                          new_sampling_rate=16000,
                                          sample_length_seconds=1)
      custom_data_module.setup()
      train_acc,val_acc = run_train_eval(model,num_classes,optimizer,criterion,custom_data_module,epochs)
      test_acc = run_test(model, num_classes, custom_data_module)
      if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_hyperparameters['learning_rate'] = lr
        best_hyperparameters['batch_size'] =  batch_size

      # results['model_name'].append(f'CNN_lr_{lr}_bs_{batch_size}')
      # results['train_acc'].append(train_acc)
      # results['val_acc'].append(val_acc)
      # results['test_acc'].append(test_acc)

  return best_hyperparameters , best_val_accuracy

best_hyperparameters , best_val_accuracy = hyperparameter_tuning_cnn()
print("")
print("Best Hyperparameters",best_hyperparameters)
print("Best Validation Accuracy",best_val_accuracy)

"""# Architecture 2 - Transformer Encoder with CNN Base"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride , padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class TransConv1DNet(nn.Module):
    def __init__(self, num_classes):
        super(TransConv1DNet, self).__init__()

        self.conv_block1 = ConvBlock(1, 64, kernel_size=7, stride=2 , padding=2)
        self.conv_block2 = ConvBlock(64, 32, kernel_size=5, stride=2 ,  padding=2)
        self.conv_block3 = ConvBlock(32, 16, kernel_size=3, stride=2 ,  padding=2)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        return x

dummy_input = torch.randn(1, 1, 144000)

model = TransConv1DNet(num_classes)
model = model.to(device)

model = model.to(device)
dummy_input = dummy_input.to(device)

output = model(dummy_input)

print("Output shape:", output.shape)

def position_embedding(max_seq_len,embedding_dim):
  pos = torch.arange(0,max_seq_len,dtype=torch.float)
  pos = pos.unsqueeze(1)
  log_term = -math.log(10000.0)
  encoding_vector = torch.arange(0,embedding_dim,2).float()
  div_term = torch.exp(encoding_vector*(log_term/embedding_dim))
  positional_encodings = torch.zeros(max_seq_len,embedding_dim)
  positional_encodings[:,0::2] = torch.sin(pos*div_term)
  positional_encodings[:,1::2] = torch.cos(pos*div_term)
  positional_encodings = positional_encodings.unsqueeze(0)
  return positional_encodings

class LayerNormalization(nn.Module):
  def __init__(self,d_model,eps=1e-5):
    super(LayerNormalization,self).__init__()
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(d_model))
    self.beta = nn.Parameter(torch.ones(d_model))

  def forward(self,x):
    x = x.to(device)
    mean = x.mean(dim=-1,keepdim=True)
    std = x.std(dim=-1,keepdim=True)
    normalized = (x - mean)/(std+self.eps)
    return self.gamma*normalized + self.beta

def scaled_dot_product(query,key,values):
  d_model = query.size(-1)
  key = key.transpose(-2,-1)
  dot_product = torch.matmul(query,key)
  scaled = dot_product/math.sqrt(d_model)
  scores = F.softmax(scaled,dim=1)
  values = torch.matmul(scores,values)
  return values,scores

class MultiHeadAttention(nn.Module):
  def __init__(self,d_model,num_head):
    super(MultiHeadAttention,self).__init__()
    self.d_model = d_model
    self.num_head = num_head
    self.head_dim = d_model//num_head
    self.qkv_layer = nn.Linear(d_model,3*d_model)
    self.linear_layer = nn.Linear(d_model,d_model)

  def forward(self,x):
    batch_size ,max_sequence_length , _= x.size()
    qkv = self.qkv_layer(x)
    qkv = qkv.view(batch_size,max_sequence_length,self.num_head,3*self.head_dim)
    q,k,v = qkv.chunk(3,dim=-1)
    values,attention = scaled_dot_product(q,k,v)
    values = values.reshape(batch_size,max_sequence_length,self.num_head*self.head_dim)
    out = self.linear_layer(values)
    return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, drop_prob=0.1):

        super(PositionwiseFeedForward, self).__init__()
        self.linear_layer1 = LinearLayer(d_model, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.linear_layer2 = LinearLayer(hidden_dim, d_model)

    def forward(self, x):
        x = self.linear_layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_layer2(x)
        x = self.relu(x)
        return x

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class AttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, drop_prob):
        super(AttentionBlock, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        residual = x
        x = self.multihead_attention(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, drop_prob):
        super(FeedForwardBlock, self).__init__()
        self.positionwise_ffn = PositionwiseFeedForward(d_model, hidden_dim, drop_prob)
        self.layer_norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        residual = x
        x = self.positionwise_ffn(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, drop_prob):
        super(TransformerEncoderBlock, self).__init__()
        self.attention_block1 = AttentionBlock(d_model, num_heads, drop_prob)
        self.attention_block2 = AttentionBlock(d_model, num_heads, drop_prob)
        self.attention_block3 = AttentionBlock(d_model, num_heads, drop_prob)
        self.attention_block4 = AttentionBlock(d_model, num_heads, drop_prob)
        self.feedforward_block = FeedForwardBlock(d_model, hidden_dim, drop_prob)

    def forward(self, x):
        x = self.attention_block1(x)
        x = self.attention_block2(x)
        x = self.attention_block3(x)
        x = self.attention_block4(x)
        x = self.feedforward_block(x)
        return x

class TransformerEncoder(nn.Module):
  def __init__(self,num_classes,d_model,num_heads,hidden_dim,drop_prob,max_sequence_length):
    super(TransformerEncoder,self).__init__()
    self.positional_embedding = position_embedding(max_sequence_length,d_model)
    self.layer = TransformerEncoderBlock(d_model,num_heads,hidden_dim,drop_prob)
    self.cls_token = nn.Parameter(torch.randn(1,1,d_model))
    self.linear_layer = nn.Linear(d_model,num_classes)

  def forward(self,x):
    x = x.permute(0,2,1)
    x = x.to(device)
    cls_token = self.cls_token.repeat(x.size(0),1,1)
    cls_token = cls_token.to(device)
    x = torch.cat([cls_token,x],dim=1)
    positional_embedding = self.positional_embedding.repeat(x.size(0),1,1)
    positional_embedding = positional_embedding.to(device)
    x = x + positional_embedding
    x = self.layer(x)
    cls_output = x[:,0,:]
    logits = self.linear_layer(cls_output)
    # mean_output = torch.mean(x, dim=1)
    # logits = self.linear_layer(mean_output)

    return logits

class CombinedModel(nn.Module):
  def __init__(self,d_model,cnn_model,transformer_model):
    super(CombinedModel,self).__init__()
    self.cnn_model = cnn_model
    self.transformer_model = transformer_model

  def forward(self,x):
    cnn_output = self.cnn_model(x)
    transformer_output = self.transformer_model(cnn_output)
    return transformer_output

"""# Train Transformer for 100 epochs for Num_Heads = 1,2,4"""

d_model = 16
num_heads = 4
hidden_dim = 1024
drop_prob = 0.1
num_classes = 10
max_sequence_length = 2251
lr = 0.001
epochs = 100
batch_size = 32
num_classes = 10
num_folds = 4

cnn_model = TransConv1DNet(num_classes)
cnn_model = cnn_model.to(device)
transformer_model = TransformerEncoder(num_classes,d_model,num_heads,hidden_dim,drop_prob,max_sequence_length)
transformer_model = transformer_model.to(device)
combined_model = CombinedModel(d_model,cnn_model,transformer_model)
combined_model = combined_model.to(device)

summary(combined_model, input_size=[32, 1, 144000])

"""# Parameters of Transformer Architecture"""

total_params = sum(p.numel() for p in combined_model.parameters())
trainable_params = sum(p.numel() for p in combined_model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

print(f"Total Trainable parameters {trainable_params}")
print(f"Total Non-Trainable parameters {non_trainable_params}")

num_heads = [1,2,4]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()

test_samp = 1
valid_samp = 2
custom_data_module = CustomDataModule(batch_size=batch_size,
                                          num_workers=num_workers,
                                          data_directory=path,
                                          data_frame=df,
                                          validation_fold=valid_samp,
                                          testing_fold=test_samp,
                                          esc_10_flag=True,
                                          file_column='filename',
                                          label_column='category',
                                          sampling_rate=44100,
                                          new_sampling_rate=16000,
                                          sample_length_seconds=1)

custom_data_module.setup()

for num_head in num_heads:
  wandb.init(project=f'DL_assign2_transformer_task_1_num_head{num_head}')
  print("")
  print(f"Running for Num_Head : {num_head}")
  print("")
  cnn_model = TransConv1DNet(num_classes)
  cnn_model = cnn_model.to(device)
  transformer_model = TransformerEncoder(num_classes,d_model,num_head,hidden_dim,drop_prob,max_sequence_length)
  transformer_model = transformer_model.to(device)
  combined_model = CombinedModel(d_model,cnn_model,transformer_model)
  combined_model = combined_model.to(device)
  optimizer_transformer = optim.Adam(combined_model.parameters(),lr=lr)
  train_acc , val_acc = run_train_eval(combined_model,num_classes,optimizer_transformer,criterion,custom_data_module,epochs)
  test_acc = run_test(combined_model,num_classes,custom_data_module)
#   results['model_name'].append(f'Transformer_NumHead_{num_head}_lr_{lr}_bs_{batch_size}_hidden_dim_{hidden_dim}')
#   results['train_acc'].append(train_acc)
#   results['val_acc'].append(val_acc)
#   results['test_acc'].append(test_acc)

"""# Run K-Fold validation Transformer"""

def run_KFold_transformer():
    k = 6
    test_samp = 1
    all_fold_accuracies = []
    num_heads = [1,2,4]
    criterion =  nn.CrossEntropyLoss()
    for num_head in num_heads:
      for fold_idx in range(2, k):
          cnn_model = TransConv1DNet(num_classes)
          cnn_model = cnn_model.to(device)
          transformer_model = TransformerEncoder(num_classes,d_model,num_head,hidden_dim,drop_prob,max_sequence_length)
          transformer_model = transformer_model.to(device)
          combined_model = CombinedModel(d_model,cnn_model,transformer_model)
          combined_model = combined_model.to(device)
          optimizer_transformer = optim.Adam(combined_model.parameters(),lr=lr)
          valid_samp = fold_idx
          batch_size = 32
          num_workers = 2
          wandb.init(project=f'DL_assign2_transformer_task_2_fold_{fold_idx}')

          custom_data_module = CustomDataModule(batch_size=batch_size,
                                                num_workers=num_workers,
                                                data_directory=path,
                                                data_frame=df,
                                                validation_fold=valid_samp,
                                                testing_fold=test_samp,
                                                esc_10_flag=True,
                                                file_column='filename',
                                                label_column='category',
                                                sampling_rate=44100,
                                                new_sampling_rate=16000,
                                                sample_length_seconds=1)

          print(f"Setting up datasets for fold {fold_idx}")
          print("")
          custom_data_module.setup()
          print("")

          train_accuracy , val_accuracy = run_train_eval(combined_model, num_classes, optimizer_transformer, criterion, custom_data_module , epochs)
          # test_accuracy = run_test(combined_model, num_classes, custom_data_module)

          all_fold_accuracies.append(val_accuracy)

      avg_accuracy = sum(all_fold_accuracies) / len(all_fold_accuracies)
      print("")
      print(f"Average accuracy for Num_head {num_head} over {k-2} folds: {avg_accuracy}")
      print("")

run_KFold_transformer()

"""# Run HyperParameter Tuning Transformer"""

def hyperparameter_tuning():
  hidden_dims = [512,1024]
  num_heads = [1,2,4]
  best_val_accuracy = 0.0
  best_hyperparameters = {}
  lr = 0.001
  batch_size=32
  criterion = nn.CrossEntropyLoss()
  for num_head in num_heads:
      for hidden_dim in hidden_dims :
        wandb.init(project=f'DL_assign2_transformer_{num_head}_{hidden_dim}')
        cnn_model = TransConv1DNet(num_classes)
        cnn_model = cnn_model.to(device)
        transformer_model = TransformerEncoder(num_classes,d_model,num_head,hidden_dim,drop_prob,max_sequence_length)
        transformer_model = transformer_model.to(device)
        combined_model = CombinedModel(d_model,cnn_model,transformer_model)
        combined_model = combined_model.to(device)
        optimizer_transformer = optim.Adam(combined_model.parameters(),lr=lr)
        custom_data_module = CustomDataModule(batch_size=batch_size,
                                            num_workers=num_workers,
                                            data_directory=path,
                                            data_frame=df,
                                            validation_fold=valid_samp,
                                            testing_fold=test_samp,
                                            esc_10_flag=True,
                                            file_column='filename',
                                            label_column='category',
                                            sampling_rate=44100,
                                            new_sampling_rate=16000,
                                            sample_length_seconds=1)
        custom_data_module.setup()
        train_acc,val_acc = run_train_eval(combined_model,num_classes,optimizer_transformer,criterion,custom_data_module,epochs)
        test_acc = run_test(combined_model, num_classes, custom_data_module)
        if val_acc > best_val_accuracy:
          best_val_accuracy = val_acc
          best_hyperparameters['num_head'] = num_head
          best_hyperparameters['hidden_dim'] =  hidden_dim

#         results['model_name'].append(f'Transformer_num_head_{num_head}_hidden_dim_{hidden_dim}')
#         results['train_acc'].append(train_acc)
#         results['val_acc'].append(val_acc)
#         results['test_acc'].append(test_acc)

  return best_hyperparameters , best_val_accuracy

best_hyperparameters , best_val_accuracy =  hyperparameter_tuning()
print("")
print("Best Hyperparameters",best_hyperparameters)
print("Best Validation Accuracy",best_val_accuracy)

results={}
results['Model Name'] = [
'CNN_lr_0.001_bs_32',
'CNN_lr_0.001_bs_64',
'CNN_lr_0.01_bs_32',
'CNN_lr_0.01_bs_64',
'Transformer_num_head_1_hidden_dim_512',
'Transformer_num_head_1_hidden_dim_1024',
'Transformer_num_head_2_hidden_dim_512',
'Transformer_num_head_2_hidden_dim_1024',
'Transformer_num_head_4_hidden_dim_512',
'Transformer_num_head_4_hidden_dim_1024']

results['Test Accuracy'] = [42.5,47.5,41.25,10,42.50,35.5,51.24,51.20,57.40,58.75]
results['Val Accuracy'] = [47.50,40,36.25,10,37.25,30.0,32.35,48.25,62.5,62.5]
results['F1-Score'] = [0.4,0.46,0.37,0.018,0.4,0.3,0.46,0.48,0.56,0.58]

"""# Comparative table"""

df_results = pd.DataFrame(results)
df_results

df_results.set_index('Model Name', inplace=True)

num_models = len(df_results)
bar_width = 0.2
index = np.arange(num_models)
plt.figure(figsize=(12, 6))

plt.bar(index + 2 * bar_width, df_results['Test Accuracy'], bar_width, label='Test Accuracy')

# Adding text above each bar
for i in range(num_models):
    plt.text(index[i] + 2 * bar_width, df_results['Test Accuracy'].iloc[i] + 0.01,
             str(round(df_results['Test Accuracy'].iloc[i], 2)), ha='center')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Metrics')
plt.xticks(index + bar_width, df_results.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

num_models = len(df_results)
bar_width = 0.2
index = np.arange(num_models)
plt.figure(figsize=(12, 6))

plt.bar(index + 2 * bar_width, df_results['Val Accuracy'], bar_width, label='Val Accuracy')

# Adding text above each bar
for i in range(num_models):
    plt.text(index[i] + 2 * bar_width, df_results['Val Accuracy'].iloc[i] + 0.01,
             str(round(df_results['Val Accuracy'].iloc[i], 2)), ha='center')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Metrics')
plt.xticks(index + bar_width, df_results.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

num_models = len(df_results)
bar_width = 0.2
index = np.arange(num_models)
plt.figure(figsize=(12, 6))

plt.bar(index + 2 * bar_width, df_results['F1-Score'], bar_width, label='F1-Score')

# Adding text above each bar
for i in range(num_models):
    plt.text(index[i] + 2 * bar_width, df_results['F1-Score'].iloc[i] + 0.01,
             str(round(df_results['F1-Score'].iloc[i], 2)), ha='center')

plt.xlabel('Model')
plt.ylabel('F1-Score')
plt.title('Model Metrics')
plt.xticks(index + bar_width, df_results.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

