import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score

#This is a sophisticated twin-channel architecture (often used for protein-protein interaction) that
# incorporates Self-Attention and Batch Normalization.
#To convert this to PyTorch, we need to define a custom nn.Module. Since the two "Channels" are
# identical in structure, we can create a reusable ProteinBranch class to keep the code clean.
#1. The Self-Attention Module
#In PyTorch, we need to explicitly define the Self_Attention layer that your Keras code uses. I have
#implemented a standard version below that matches the Keras behavior of applying attention over a sequence.

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (Batch, Seq_len, Dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Scale dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v)
        return out

#2. The Full Model Architecture
#Here is the PyTorch equivalent of your define_model(). I have grouped the repeated logic into a ProteinBranch
# to mirror your "Channel-1" and "Channel-2".
class ProteinBranch(nn.Module):
    def __init__(self, input_dim=573, dr=0.2):
        super(ProteinBranch, self).__init__()

        # Block 1: Feature Extraction
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dr),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dr),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dr)
        )

        # Attention Mechanism
        self.attention = SelfAttention(32)
        self.attn_bn = nn.BatchNorm1d(8)  # Applied after reshape (8, 32)
        self.attn_dr = nn.Dropout(0.1)

        # Block 2: Post-Attention
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dr),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dr),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dr)
        )

    def forward(self, x):
        x = self.block1(x)

        # Reshape to (Batch, 8, 32) for Attention
        x = x.view(-1, 8, 32)
        x = self.attention(x)
        x = self.attn_bn(x)
        x = self.attn_dr(x)

        # Flatten back for Dense layers
        x = x.view(-1, 256)
        x = self.block2(x)
        return x


class ProteinInteractionModel(nn.Module):
    def __init__(self, dr=0.2):
        super(ProteinInteractionModel, self).__init__()
        self.branch_a = ProteinBranch(dr=dr)
        self.branch_b = ProteinBranch(dr=dr)

        # Prediction Module
        self.merge_dense = nn.Linear(64, 32)  # 32 (Branch A) + 32 (Branch B)
        self.merge_attn = SelfAttention(32)
        self.merge_bn = nn.BatchNorm1d(1)
        self.merge_dr = nn.Dropout(0.1)

        self.final_layers = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, pro_a, pro_b):
        feat_a = self.branch_a(pro_a)
        feat_b = self.branch_b(pro_b)

        # Concatenate
        merged = torch.cat((feat_a, feat_b), dim=1)

        # Head Attention Block
        x = F.relu(self.merge_dense(merged))
        x = x.view(-1, 1, 32)  # Reshape(1, 32)
        x = self.merge_attn(x)
        x = self.merge_bn(x)
        x = self.merge_dr(x)

        x = x.view(-1, 32)  # Flatten
        return self.final_layers(x)

#3. Key Conversion Differences
#When moving this specific model from Keras to PyTorch, keep these three points in mind:
#L2 Regularization: In your Keras code, you use kernel_regularizer=l2(l2c). In PyTorch, you don't define
# this in the layer. Instead, you pass weight_decay=l2c to your optimizer (e.g., torch.optim.SGD).
#Initializers: Keras uses glorot_normal. PyTorch layers use kaiming_uniform by default. To match Keras exactly,
# you should apply initialization:
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

model = ProteinInteractionModel()
model.apply(init_weights)
#Optimizer: Your Keras SGD has a manual learning rate decay. In PyTorch, it's cleaner to
# use a Learning Rate Scheduler (torch.optim.lr_scheduler).

#4. Setup Training
model = ProteinInteractionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
criterion = nn.BCELoss() # Binary Crossentropy

#Since PyTorch doesn't have a built-in .fit() method like Keras, we have to be a bit more explicit. Think of it as switching from an automatic transmission to a manual—you have more control, but you have to shift the gears yourself.
#Here is the complete pipeline to get your protein interaction model running.
#1. Dataset Preparation
#We’ll use torch.utils.data.Dataset to handle the twin inputs (Protein A and Protein B).
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ProteinDataset(Dataset):
    def __init__(self, features_a, features_b, labels=None):
        """
        features_a/b: Numpy arrays or tensors of shape (N, 573)
        labels: Numpy array of shape (N, 1) or None for prediction
        """
        self.features_a = torch.tensor(features_a, dtype=torch.float32)
        self.features_b = torch.tensor(features_b, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.features_a)

    def __getitem__(self, idx):
        data_a = self.features_a[idx]
        data_b = self.features_b[idx]

        if self.labels is not None:
            return data_a, data_b, self.labels[idx]
        return data_a, data_b
# Example: Creating dummy data for demonstration
# Xa_train = np.random.rand(1000, 573)
# Xb_train = np.random.rand(1000, 573)
# y_train = np.random.randint(0, 2, (1000, 1))

# train_ds = ProteinDataset(Xa_train, Xb_train, y_train)
# train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

#2. Training Script
#This script handles the optimization loop, loss calculation, and backpropagation.
def train_model(model, train_loader, val_loader=None, epochs=50, lr=0.01, modelname=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Matching your Keras SGD setup
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.BCELoss()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_a, batch_b, labels in train_loader:
            batch_a, batch_b, labels = batch_a.to(device), batch_b.to(device), labels.to(device)
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_a, batch_b)
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Basic Validation Logic
        #model.eval()
        #val_loss = 0
        #with torch.no_grad():
        #    for va, vb, vl in val_loader:
        #        va, vb, vl = va.to(device), vb.to(device), vl.to(device)
        #        v_out = model(va, vb)
        #        val_loss += criterion(v_out, vl).item()

        if epoch > 30:
            print(
                #f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss / len(train_loader):.4f} - Val Loss: {val_loss / len(val_loader):.4f}")
                f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss / len(train_loader):.4f}")

    # Save the weights
    torch.save(model.state_dict(), "results/" + modelname + "_protein_model_Human.pth")
    print("Model saved!")
#3. Prediction Script
#Once the model is trained, use this to run inference on new protein pairs.
def predict(model, test_features_a, test_features_b, weight_path="protein_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load weights
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    # Prepare data
    tensor_a = torch.tensor(test_features_a, dtype=torch.float32).to(device)
    tensor_b = torch.tensor(test_features_b, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(tensor_a, tensor_b)

    return predictions.cpu().numpy()
# Usage:
# results = predict(my_model, new_protein_a_data, new_protein_b_data)
# print(f"Interaction Probability: {results}")

def calculate_performace(y_test_1, y_score_1):
    auc = metrics.roc_auc_score(y_test_1, y_score_1)
    for i in range(0, len(y_score_1)):
        if (y_score_1[i] > 0.5):
            y_score_1[i] = 1
        else:
            y_score_1[i] = 0
    cm1 = confusion_matrix(y_test_1, y_score_1)
    acc1 = accuracy_score(y_test_1, y_score_1, sample_weight=None)
    spec1 = (cm1[0, 0]) / (cm1[0, 0] + cm1[0, 1])
    sens1 = recall_score(y_test_1, y_score_1, sample_weight=None)
    prec1 = precision_score(y_test_1, y_score_1, sample_weight=None)
    return auc, cm1, acc1, spec1, sens1, prec1

df_pos_A_AC = pd.read_csv('Data/Human/P_Protein_A_AC.csv')
df_pos_A_AC = df_pos_A_AC.drop(df_pos_A_AC.columns[0], axis=1)
df_pos_B_AC = pd.read_csv('Data/Human/P_Protein_B_AC.csv')
df_pos_B_AC = df_pos_B_AC.drop(df_pos_B_AC.columns[0], axis=1)
df_pos_A_ACC = pd.read_csv('Data/Human/P_Protein_A_ACC.csv')
df_pos_A_ACC = df_pos_A_ACC.drop(df_pos_A_ACC.columns[0], axis=1)
df_pos_B_ACC = pd.read_csv('Data/Human/P_Protein_B_ACC.csv')
df_pos_B_ACC = df_pos_B_ACC.drop(df_pos_B_ACC.columns[0], axis=1)
df_pos_A_conjoint = pd.read_csv('Data/Human/P_Protein_A_conjoint.csv')
df_pos_A_conjoint = df_pos_A_conjoint.drop(df_pos_A_conjoint.columns[0], axis=1)
df_pos_B_conjoint = pd.read_csv('Data/Human/P_Protein_B_conjoint.csv')
df_pos_B_conjoint = df_pos_B_conjoint.drop(df_pos_B_conjoint.columns[0], axis=1)
# Concatenate horizontally
df_pos = pd.concat([df_pos_A_AC, df_pos_A_ACC, df_pos_A_conjoint
                    ,df_pos_B_AC, df_pos_B_ACC, df_pos_B_conjoint], axis=1)
print(df_pos.shape)

df_neg_A_AC = pd.read_csv('Data/Human/N_Protein_A_AC.csv')
df_neg_A_AC =  df_neg_A_AC.drop( df_neg_A_AC.columns[0], axis=1)
df_neg_B_AC = pd.read_csv('Data/Human/N_Protein_B_AC.csv')
df_neg_B_AC =  df_neg_B_AC.drop( df_neg_B_AC.columns[0], axis=1)
df_neg_A_ACC = pd.read_csv('Data/Human/N_Protein_A_ACC.csv')
df_neg_A_ACC =  df_neg_A_ACC.drop( df_neg_A_ACC.columns[0], axis=1)
df_neg_B_ACC = pd.read_csv('Data/Human/N_Protein_B_ACC.csv')
df_neg_B_ACC =  df_neg_B_ACC.drop( df_neg_B_ACC.columns[0], axis=1)
df_neg_A_conjoint = pd.read_csv('Data/Human/N_Protein_A_conjoint.csv')
df_neg_A_conjoint =  df_neg_A_conjoint.drop( df_neg_A_conjoint.columns[0], axis=1)
df_neg_B_conjoint = pd.read_csv('Data/Human/N_Protein_B_conjoint.csv')
df_neg_B_conjoint =  df_neg_B_conjoint.drop( df_neg_B_conjoint.columns[0], axis=1)
# Concatenate horizontally
df_neg = pd.concat([ df_neg_A_AC,  df_neg_A_ACC,  df_neg_A_conjoint
                    , df_neg_B_AC,  df_neg_B_ACC,  df_neg_B_conjoint], axis=1)
print( df_neg.shape)

df_neg['Status'] = 0
df_pos['Status'] = 1
df_neg = df_neg.sample(n=len(df_pos))

df = pd.concat([df_pos, df_neg])
df = df.reset_index()
df = df.sample(frac=1)
df = df.iloc[:, 1:]

X = df.iloc[:, 0:1146].values
y = df.iloc[:, 1146:].values
Trainlabels = y
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

##Let us try 10 times
resultStr = ""
for seedrandom in range(2):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,       # 33% of the data will be used for testing
        random_state=seedrandom,     # Ensures the split is the same every time for reproducibility
        shuffle=True         # Shuffles the data before splitting (default is True)
    )
    X1_train = X_train[:, :573]
    X2_train = X_train[:, 573:]
    X1_test = X_test[:, :573]
    X2_test = X_test[:, 573:]
    train_ds = ProteinDataset(X1_train, X2_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    train_model(model, train_loader,modelname="seed-" + str(seedrandom))
    y_score = predict(model, X1_test, X2_test, "results/seed-" + str(seedrandom) + "_protein_model_Human.pth")
    auc, cm1, acc1, spec1, sens1, prec1 = calculate_performace(y_test, y_score)
    print("Seed:" + str(seedrandom) + "\tAUC:" + str(auc) + "\tACC:" + str(acc1) + "\tsens:" + str(sens1))
    resultStr = resultStr + "\nSeed:" + str(seedrandom) + "\tAUC:" + str(auc) + "\tACC:" + str(acc1) + "\tsens:" + str(sens1)
    model.apply(init_weights) #reinit
with open('results/result_Human.txt', 'w') as f:
    f.write(resultStr)
print(resultStr)
print("Hello World")