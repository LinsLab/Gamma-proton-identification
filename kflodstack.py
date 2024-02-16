import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold,cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
import torch.nn.functional as F
from joblib import dump

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
        self.classes_ = np.unique(y)
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
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

        
def stack(datapath,outpath):
    df12=pd.read_csv(datapath)  
    X=df12[['NuM4','NfiltM','base','NuM2','NuW1','NuM3','NhitM','NuM1']]
    y=df12['lable']
    
    scaler = StandardScaler()
    
    # 对数据框进行标准化
    standardized_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X = np.asarray(standardized_X)
    y = np.asarray(y)   
    # 初始化DNN模型
    torch_model = DNN()
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.001)
    ratio = float(np.sum(y== 0)) / np.sum(y == 1) 
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=y)
    weights = {0: class_weights[0], 1: class_weights[1]}  
    # 定义基模型
    base_learners = [
        ('lr', LogisticRegression(penalty='l2',C=0.1,solver='liblinear',max_iter=1000,class_weight='balanced')),
        ('svm', SVC(probability=True,kernel='linear',class_weight='balanced')),
        ('dt', DecisionTreeClassifier(criterion='gini', splitter= 'best', max_depth=30, min_samples_split= 2, min_samples_leaf=1,class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators= 100, max_depth=30, min_samples_split=1, min_samples_leaf=0.2, bootstrap= False,random_state=42)),
        ('xgb', XGBClassifier(learning_rate=0.1,n_estimators=500,max_depth=8,min_child_weight=0.01,scale_pos_weight=ratio)),
        ('catboost', CatBoostClassifier(iterations=300,depth=10,learning_rate=0.1,random_strength=10,bagging_temperature=1,od_type='Iter',
                                   od_wait=50,class_weights=weights, verbose=0, eval_metric='Accuracy')),
        ('dnn', TorchDNNClassifier(torch_model, criterion, optimizer, epochs=50,lr=0.001))
    ]
    repeat_results=[]
    repeat_models=[]
    # 使用LogisticRegression作为元模型
    stack = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())

    # 定义评分指标
    scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc']
    # 使用RepeatedStratifiedKFold进行交叉验证
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    scores = cross_validate(stack, X, y, cv=cv, scoring=scoring,return_train_score=False,return_estimator=True,n_jobs=-1)

    # 打印每次交叉验证的评价指标结果
    for i, (test_accuracy, test_f1, test_precision, test_recall, test_auc,estimator) in enumerate(zip(scores['test_accuracy'], scores['test_f1_macro'], scores['test_precision_macro'], scores['test_recall_macro'], scores['test_roc_auc'],scores['estimator'])):
        print(f"Fold {i+1} - Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, AUC: {test_auc:.4f}")
        
        y_pred = estimator.predict(X)
        y_prob = estimator.predict_proba(X)[:, 1]
        repeat_results.append((test_accuracy, y, y_pred, y_prob))
        repeat_models.append((test_accuracy,estimator))

        # 每五次交叉验证结束，选择准确率最高的那次
        if (i+1) % 5 == 0:
            best_result = max(repeat_results, key=lambda x: x[0])
            best_y, best_y_pred, best_y_prob = best_result[1:]
            best_model = max(repeat_models, key=lambda x: x[0])[1]
            # 保存最佳模型
            dump(best_model, f'/home/abcdlj/Gam-p/final/models/{outpath}_stack_best_model_{int((i+1)/5)}.joblib')
            df_best = pd.DataFrame({'y_true': best_y, 'y_pred': best_y_pred, 'y_prob': best_y_prob})
            df_best.to_csv(f'/home/abcdlj/Gam-p/final/predict/{outpath}_stack_best_{int((i+1)/5)}.csv', index=False)
            repeat_results = []
            repeat_models = []
            
    df_scores = pd.DataFrame({
        'Accuracy': scores['test_accuracy'],
        'F1': scores['test_f1_macro'],
        'Precision': scores['test_precision_macro'],
        'Recall': scores['test_recall_macro'],
        'AUC': scores['test_roc_auc']
    })

    # 保存DataFrame到CSV文件
    df_scores.to_csv(f'/home/abcdlj/Gam-p/final/predict/{outpath}_stack.csv', index=False)
    
   
if __name__=="__main__":
    """path1='/home/abcdlj/Gam-p/final/data/df15rec.csv'
    out1='df15'
    stack(path1,out1)
    path2='/home/abcdlj/Gam-p/final/data/df14rec.csv'
    out2='df14'
    stack(path2,out2)
    path3='/home/abcdlj/Gam-p/final/data/df13rec.csv'
    out3='df13'
    stack(path3,out3)"""
    path4='/home/abcdlj/Gam-p/final/data/df12rec.csv'
    out4='df12'
    stack(path4,out4)
    
