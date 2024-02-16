import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from joblib import dump, load

# 定义一个简单的PyTorch模型
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__() 
        
        # 定义隐藏层
        self.hidden1 = nn.Linear(8, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 32)
    
        # 定义输出层
        self.output = nn.Linear(32, 2)
        
        # 定义丢失层
        self.dropout = nn.Dropout(0.1)
        # 定义激活函数
        self.relu = nn.ReLU()
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout(x)
        x = self.relu(self.hidden3(x))
        x = self.dropout(x)
        x = self.softmax(self.output(x))
        return x
# 定义损失函数（Focal Loss）
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (self.alpha * (1 - pt) ** self.gamma).detach()
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()
    
# 将PyTorch模型包装为sklearn风格的估计器
class TorchDNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,model,criterion,optimizer,epochs=50, lr=0.001):
        self.epochs = epochs
        self.lr = lr
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, X, y):
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        with torch.no_grad():
            outputs = self.model(torch.FloatTensor(X))
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

    def predict_proba(self, X):
        with torch.no_grad():
            outputs = torch.softmax(self.model(torch.FloatTensor(X)), dim=1)
        return outputs.numpy()


def dnnkflod(path,outname):
    df12=pd.read_csv(path)  
    # 创建RepeatedStratifiedKFold对象
    n_splits = 5  # 5折交叉验证
    n_repeats = 10  # 重复10次
    random_state = 42  # 随机种子，可选
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    X=df12[['NuM4','NfiltM','base','NuM2','NuW1','NuM3','NhitM','NuM1']]
    y=df12['lable']
    # 初始化StandardScaler
    scaler = StandardScaler()
    
    # 对数据框进行标准化
    standardized_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X = np.asarray(standardized_X)
    y = np.asarray(y)
    
  
    acc=[]
    f=[]
    pre=[]
    rec=[]
    a=[]
    repeat_models = []  # 用于存储每次迭代的模型
    repeat_results = []
    # 循环进行交叉验证
    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        print(f"第 {i+1} 次开始")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
         # 初始化DNN模型
        torch_model = DNN()
        criterion = FocalLoss()
        optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.001)
 
        # 创建模型
        model = TorchDNNClassifier(torch_model, criterion, optimizer, epochs=50,lr=0.001)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测标签
        y_pred = model.predict(X_test)
        
        # 预测概率
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 计算性能指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        print('Accuracy:', accuracy,'F1:', f1,'Precision:', precision,'Recall:', recall,'AUC:',auc)
        # 保存性能指标和预测结果
        # 将评价指标保存到DataFrame中
        acc.append(accuracy)
        f.append(f1)
        pre.append(precision)
        rec.append(recall)
        a.append(auc)
        
        
        repeat_results.append((accuracy, y_test, y_pred, y_prob))
        # 在存储评价指标后，存储模型
        repeat_models.append((accuracy, model))   
        # 每五次交叉验证结束，选择准确率最高的那次
        if (i+1) % n_splits == 0:
            best_result = max(repeat_results, key=lambda x: x[0])
            best_y_test, best_y_pred, best_y_prob = best_result[1:]
            best_model = max(repeat_models, key=lambda x: x[0])[1]
            # 保存最佳模型
            dump(best_model, f'/home/abcdlj/Gam-p/final/models/DNN_{outname}_best_model_{int((i+1)/n_splits)}.joblib')             
            df_best = pd.DataFrame({'y_test': best_y_test, 'y_pred': best_y_pred, 'y_prob': best_y_prob})
            df_best.to_csv(f'/home/abcdlj/Gam-p/final/predict/DNN_{outname}_best_{int((i+1)/n_splits)}.csv', index=False)
            repeat_results = []
            repeat_models = []
                       
    # 保存评价指标到CSV文件
    df_scores = pd.DataFrame({
        'Accuracy': acc,
        'F1': f,
        'Precision': pre,
        'Recall': rec,
        'AUC': a
    })
    df_scores.to_csv('/home/abcdlj/Gam-p/final/predict/DNN_{}_results.csv'.format(outname), index=False)   
    
if __name__=="__main__":
    """path1='/home/abcdlj/Gam-p/final/data/df15rec.csv'
    dnnkflod(path1,'df15')
    path2='/home/abcdlj/Gam-p/final/data/df14rec.csv'
    dnnkflod(path2,'df14')
    path3='/home/abcdlj/Gam-p/final/data/df13rec.csv'
    dnnkflod(path3,'df13')"""
    path4='/home/abcdlj/Gam-p/final/data/df12rec.csv'
    dnnkflod(path4,'df12')