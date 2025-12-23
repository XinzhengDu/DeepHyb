import numpy as np
import torch
import json
import os
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from tqdm import tqdm
import pandas as pd
import time

start_time = time.time()

# Using GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(
    f"#################### Your GPU usage is {torch.cuda.is_available()}! ########################\n")

# 指定文件夹路径
train_folder = '/mnt/disk2/tang/3_latest_site_pattern_counter_script/10w-13spes-773/output_jsons1001-1500'
test_folder = '../no35-10w-1/test/'
suffix = "15+256+15m5+64m4d"

epochs = 300000
batch_size = 512
maxloss = 0.001
lr = 0.001

# Parameters
input_dim = 15+256+75+256
num_classes_1 = 2
num_classes_3 = 64

model_save_path = f"./model/1225_concate_CNN_model_{suffix}.pth"
loss_save_path = f"./loss_history/1226_concate_CNN_model_{suffix}_avg_loss.json"

os.makedirs("./model", exist_ok= True)
os.makedirs("./loss_history", exist_ok= True)

# Print parameters
print("=" * 40)
print("MODEL PARAMETERS".center(40))
print("=" * 40)
print(f"{'Train Folder:':<20} {train_folder}")
print(f"{'Test Folder:':<20} {test_folder}")
print(f"{'Epochs:':<20} {epochs}")
print(f"{'Batch Size:':<20} {batch_size}")
print(f"{'Maximum Loss:':<20} {maxloss}")
print(f"{'Learning Rate:':<20} {lr}")
print(f"{'Model Save Path:':<20} {model_save_path}")
print(f"{'Loss Save Path:':<20} {loss_save_path}")
print("=" * 40)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64 * (input_dim // 8), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_1 = nn.Linear(64, num_classes_1)
        self.fc3_3 = nn.Linear(64, num_classes_3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, 2)
        
        x = x.view(-1, 64 * (input_dim // 8))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x1 = self.fc3_1(x)
        x2 = self.fc3_3(x)
        
        return x1, x2
        # CrossEntropyLoss expect raw logits as input
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

def load_data_from_json(folder_path):
    X = []
    X_labels_1 = []
    X_labels_2 = []
    X_labels_3 = []
    X_labels_8 = []

    y = []
    target4 = []
    file_names = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    onedim = np.concatenate([
                    np.array(data['labels_1']), # Hyde 15
                    np.array(data['labels_2']), # 256
                    np.array(data['labels_3']).flatten(), # vertical kmer 5 groups
                    np.array(data['labels_8']).flatten(), # horizontal kmer 4 groups
                ])

                    if len(data['labels_1']) == 15:
                        X.append(onedim)
                        X_labels_1.append(data['labels_1'])  # 1x15
                        X_labels_2.append(data['labels_2'])  # 1x256
                        X_labels_3.append(data['labels_3'])  
                        X_labels_8.append(data['labels_8']) 

                        ged_mapping = {2: 1, 8: 0}
                        y.append(ged_mapping.get(data['ged'], -1))  # Default to -1 for unexpected values

                        target4.append(data['target4'])  # 或者其他合适的默认值

                        file_names.append(filename.replace('.json', ''))
                    else:
                        print(f"Warning: Inconsistent shape in file {filename}, skipped.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {filename}: {e}")
            except KeyError as e:
                print(f"Key error in file {filename}: {e}")
            except Exception as e:
                print(f"An error occurred in file {filename}: {e}")

    return (np.array(X_labels_1, dtype=np.float32), #1
            np.array(X_labels_2, dtype=np.float32), #2
            np.array(X_labels_3, dtype=np.float32), #3
            np.array(X_labels_8, dtype=np.float32), #4
            np.array(y, dtype=np.int32), #5
            np.array(target4, dtype=np.int32), #6
            file_names, #7
            np.array(X, dtype=np.float32)) #8


# 加载 a 文件夹的所有数据
_, _, _, _, y_train_a, target4_train_a, _, X_train = load_data_from_json(train_folder)
_, _, _, _, y_test_a, target4_test, file_names_b, X_test = load_data_from_json(test_folder)


X_train_tensor = torch.tensor(X_train).unsqueeze(1)
y_train_tensor = torch.tensor(y_train_a, dtype=torch.long)
target4_train_tensor = torch.tensor(target4_train_a, dtype=torch.long)

X_test_tensor = torch.tensor(X_test).unsqueeze(1)
y_test_tensor = torch.tensor(y_test_a, dtype=torch.long)
target4_test_tensor = torch.tensor(target4_test, dtype=torch.long)

# 数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, target4_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    
class MultiTargetLoss(nn.Module):
    def __init__(self):
        super(MultiTargetLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, ged_pred, target4_pred, ged_true, target4_true):
        # 计算每个目标的损失
        loss_ged = self.cross_entropy(ged_pred, ged_true)
        loss_target4 = self.cross_entropy(target4_pred, target4_true)
        total_loss = loss_ged + loss_target4
        return total_loss

# 初始化模型、损失函数和优化器
model = CNNModel().to(device)
criterion = MultiTargetLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 用于保存每个周期的损失
loss_history = []

# 早停法参数
patience = 30000  # 早停法的耐心值
min_val_loss = float('inf')  # 初始化最小验证损失
early_stopping_counter = 0  # 早停计数器

# 训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for X_batch, y_batch, target4_batch in train_loader:
            optimizer.zero_grad()

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            target4_batch = target4_batch.to(device)

            ged_pred, target4_pred = model(X_batch)
            loss = criterion(ged_pred, target4_pred, y_batch, target4_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    average_loss = total_loss / len(train_loader)
    loss_history.append(average_loss)
    print(f"Epoch {epoch+1} finished with average loss: {average_loss}, total loss: {total_loss}")

    # 评估验证集上的性能
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch, target4_batch in train_loader:  # 这里使用验证集加载器
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            target4_batch = target4_batch.to(device)

            ged_pred, target4_pred = model(X_batch)
            loss = criterion(ged_pred, target4_pred, y_batch, target4_batch)
            val_loss += loss.item()

    val_loss /= len(train_loader)  # 计算平均验证损失
    print(f"Validation loss: {val_loss}")

    # 早停法逻辑
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        early_stopping_counter = 0  # 重置计数器
        torch.save(model.state_dict(), model_save_path)  # 保存最佳模型
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break  # 结束训练循环

# 保存损失历史到 JSON 文件
with open(loss_save_path, 'w') as f:
    json.dump(loss_history, f)

def evaluate(model, y_true, target4_true, file_names, X):
    model.eval()
    predictions = []
    target4_predictions = []

    with torch.no_grad():
        X_tensor = torch.tensor(X).unsqueeze(1).to(device)
        ged_pred, target4_pred= model(X_tensor)

        ged_prob = F.softmax(ged_pred, dim = 1)
        _, ged_prediction = torch.max(ged_prob, 1)

        target4_prob = F.softmax(target4_pred, dim = 1)
        _, target4_prediction = torch.max(target4_prob, 1)

        predictions.extend(ged_prediction.cpu().numpy())
        target4_predictions.extend(target4_prediction.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, predictions)

    target4_accuracy = accuracy_score(target4_true, target4_predictions)
    cm_3 = confusion_matrix(target4_true, target4_predictions)

    return (predictions, y_true, accuracy, precision,
            recall, f1, cm, cm_3, target4_predictions,
            target4_true, target4_accuracy)

# 评估模型并获取预测值、实际值、文件名以及评估指标
predictions, actuals, accuracy, precision, recall, f1, cm, cm_3, target4_predictions, target4_actuals, target4_accuracy= evaluate(model, y_test_a, target4_test, file_names_b, X_test)
unique_target4 = np.unique(list(target4_actuals) + list(target4_predictions))
# 打印预测值、实际值和文件名
#for file_name, prediction, actual, target4_predictions, target4_actuals in zip(file_names_b, predictions, actuals, target4_predictions, target4_actuals):
#    print(f'File: {file_name}, hyb Predicted: {prediction}, hyb Actual: {actual}, target4_predictions:{target4_predictions}, target4_actuals:{target4_actuals}')

cm1_df = pd.DataFrame(cm, index=["no-Hybrid", "Hybrid"], columns=["Predicted no-Hybrid", "Predicted Hybrid"])
cm3_df = pd.DataFrame(cm_3, index = unique_target4,#["0", '1', '2', '4', '7', '8', '13', '22', '25', '26'], 
                           columns= unique_target4)#["p0", 'p1', 'p2', 'p4', 'p7', 'p8', 'p13', 'p22', 'p25', 'p26'])
# 打印评估指标
print(f'\nAccuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('\nGed Confusion Matrix:')
print(cm1_df)


print(f'\nTarget 3 Accuracy: {target4_accuracy}')
print("\nTarget3 Confusion Matrix")
print(cm3_df)


end_time = time.time()
duration = end_time - start_time

print(f"\nModel Training and Testing time: {duration}")
