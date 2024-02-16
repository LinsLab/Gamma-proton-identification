from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from joblib import dump, load

def frkflod(path,outname):
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
        
        # 创建模型
        model = RandomForestClassifier(n_estimators= 100, max_depth=30, min_samples_split=1, min_samples_leaf=0.2, bootstrap= False,random_state=42)
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
            dump(best_model, f'/home/abcdlj/Gam-p/final/models/RF_{outname}_best_model_{int((i+1)/n_splits)}.joblib')             
            df_best = pd.DataFrame({'y_test': best_y_test, 'y_pred': best_y_pred, 'y_prob': best_y_prob})
            df_best.to_csv(f'/home/abcdlj/Gam-p/final/predict/RF_{outname}_best_{int((i+1)/n_splits)}.csv', index=False)
            repeat_results = []
            repeat_models = []
    # 保存评价指标到CSV文件
    # 保存评价指标到CSV文件
    df_scores = pd.DataFrame({
        'Accuracy': acc,
        'F1': f,
        'Precision': pre,
        'Recall': rec,
        'AUC': a
    })
    df_scores.to_csv('/home/abcdlj/Gam-p/final/predict/RF_{}_results.csv'.format(outname), index=False)             

if __name__=="__main__":
    path1='/home/abcdlj/Gam-p/final/data/df15rec.csv'
    frkflod(path1,'df15')
    path2='/home/abcdlj/Gam-p/final/data/df14rec.csv'
    frkflod(path2,'df14')
    path3='/home/abcdlj/Gam-p/final/data/df13rec.csv'
    frkflod(path3,'df13')
    path4='/home/abcdlj/Gam-p/final/data/df12rec.csv'
    frkflod(path4,'df12')