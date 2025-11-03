"""
XGBoost模型模块
包含模型训练、预测、评估、保存/加载等功能
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from typing import Dict, List, Tuple, Optional, Union
import joblib
import json
from datetime import datetime


class XGBoostModel:
    """XGBoost模型类"""
    
    def __init__(self, 
                 task_type: str = 'classification',
                 params: Dict = None):
        """
        初始化XGBoost模型
        
        Args:
            task_type: 任务类型 ('classification' 或 'regression')
            params: 模型参数字典
        """
        self.task_type = task_type
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.training_history = []
        
        # 默认参数
        default_params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        self.params = default_params
        
        # 初始化模型
        if task_type == 'classification':
            self.params['objective'] = 'binary:logistic'
            self.params['eval_metric'] = 'auc'
            self.model = XGBClassifier(**self.params)
        elif task_type == 'regression':
            self.params['objective'] = 'reg:squarederror'
            self.params['eval_metric'] = 'rmse'
            self.model = XGBRegressor(**self.params)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        print(f"XGBoostModel initialized: task_type={task_type}")
        print(f"Parameters: {self.params}")
    
    def train(self,
             X_train: pd.DataFrame,
             y_train: pd.Series,
             X_val: pd.DataFrame = None,
             y_val: pd.Series = None,
             sample_weight: np.ndarray = None,
             early_stopping_rounds: int = 10,
             verbose: bool = True) -> 'XGBoostModel':
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            sample_weight: 样本权重
            early_stopping_rounds: 早停轮数
            verbose: 是否打印训练过程
            
        Returns:
            self
        """
        print(f"\nTraining model...")
        print(f"Training samples: {len(X_train)}")
        if X_val is not None:
            print(f"Validation samples: {len(X_val)}")
        
        self.feature_names = X_train.columns.tolist()
        
        # 准备训练参数
        fit_params = {
            'verbose': verbose
        }
        
        # 添加验证集
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_train, y_train), (X_val, y_val)]
            fit_params['early_stopping_rounds'] = early_stopping_rounds
        
        # 添加样本权重
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        
        # 训练
        self.model.fit(X_train, y_train, **fit_params)
        
        # 保存特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_,
            'importance_type': 'gain'
        }).sort_values('importance', ascending=False)
        
        # 评估
        train_metrics = self.evaluate(X_train, y_train)
        print(f"\nTraining metrics: {train_metrics}")
        
        if X_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            print(f"Validation metrics: {val_metrics}")
        
        # 记录训练历史
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'train_size': len(X_train),
            'val_size': len(X_val) if X_val is not None else 0,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics if X_val is not None else None
        })
        
        print("Training completed!")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.task_type == 'classification':
            # 返回正类概率
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率（仅分类任务）
        
        Args:
            X: 特征数据
            
        Returns:
            预测概率
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        return self.model.predict_proba(X)
    
    def predict_class(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        预测类别（仅分类任务）
        
        Args:
            X: 特征数据
            threshold: 分类阈值
            
        Returns:
            预测类别
        """
        if self.task_type != 'classification':
            raise ValueError("predict_class only available for classification tasks")
        
        proba = self.predict(X)
        return (proba >= threshold).astype(int)
    
    def predict_rank(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测并排名（返回百分位排名）
        
        Args:
            X: 特征数据
            
        Returns:
            百分位排名（0-1之间）
        """
        predictions = self.predict(X)
        ranks = pd.Series(predictions).rank(pct=True).values
        return ranks
    
    def evaluate(self,
                X: pd.DataFrame,
                y: pd.Series) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            X: 特征数据
            y: 真实标签
            
        Returns:
            评估指标字典
        """
        predictions = self.predict(X)
        
        metrics = {}
        
        if self.task_type == 'classification':
            # 分类指标
            y_pred_class = (predictions > 0.5).astype(int)
            
            metrics['accuracy'] = accuracy_score(y, y_pred_class)
            metrics['precision'] = precision_score(y, y_pred_class, zero_division=0)
            metrics['recall'] = recall_score(y, y_pred_class, zero_division=0)
            metrics['f1'] = f1_score(y, y_pred_class, zero_division=0)
            
            try:
                metrics['auc'] = roc_auc_score(y, predictions)
            except:
                metrics['auc'] = np.nan
        
        else:
            # 回归指标
            metrics['rmse'] = np.sqrt(mean_squared_error(y, predictions))
            metrics['mae'] = mean_absolute_error(y, predictions)
            metrics['r2'] = r2_score(y, predictions)
            
            # IC (相关系数)
            from scipy.stats import spearmanr, pearsonr
            metrics['ic_spearman'], _ = spearmanr(y, predictions)
            metrics['ic_pearson'], _ = pearsonr(y, predictions)
        
        return metrics
    
    def get_feature_importance(self, 
                              top_n: int = 20,
                              importance_type: str = 'gain') -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            top_n: 返回前N个重要特征
            importance_type: 重要性类型 ('gain', 'weight', 'cover')
            
        Returns:
            特征重要性DataFrame
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet")
        
        # 如果需要不同类型的重要性
        if importance_type != 'gain':
            booster = self.model.get_booster()
            importance_dict = booster.get_score(importance_type=importance_type)
            
            importance_df = pd.DataFrame({
                'feature': list(importance_dict.keys()),
                'importance': list(importance_dict.values()),
                'importance_type': importance_type
            }).sort_values('importance', ascending=False)
        else:
            importance_df = self.feature_importance.copy()
        
        return importance_df.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """
        绘制特征重要性图
        
        Args:
            top_n: 显示前N个特征
            figsize: 图片大小
        """
        import matplotlib.pyplot as plt
        
        importance_df = self.get_feature_importance(top_n=top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def get_training_history(self) -> List[Dict]:
        """获取训练历史"""
        return self.training_history
    
    def save_model(self, filepath: str, save_history: bool = True):
        """
        保存模型
        
        Args:
            filepath: 保存路径
            save_history: 是否保存训练历史
        """
        # 保存模型
        joblib.dump(self.model, filepath)
        
        # 保存元数据
        metadata = {
            'task_type': self.task_type,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None,
            'training_history': self.training_history if save_history else None
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {filepath}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        # 加载模型
        self.model = joblib.load(filepath)
        
        # 加载元数据
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.task_type = metadata.get('task_type')
            self.params = metadata.get('params')
            self.feature_names = metadata.get('feature_names')
            
            if metadata.get('feature_importance'):
                self.feature_importance = pd.DataFrame(metadata['feature_importance'])
            
            if metadata.get('training_history'):
                self.training_history = metadata['training_history']
            
            print(f"Model loaded from {filepath}")
        except FileNotFoundError:
            print(f"Warning: Metadata file not found at {metadata_path}")
    
    def get_model_summary(self) -> Dict:
        """
        获取模型摘要信息
        
        Returns:
            模型摘要字典
        """
        summary = {
            'task_type': self.task_type,
            'params': self.params,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'n_training_rounds': len(self.training_history),
            'is_trained': self.model is not None
        }
        
        if self.feature_importance is not None:
            summary['top_5_features'] = self.feature_importance.head(5)['feature'].tolist()
        
        return summary


def test_xgboost_model():
    """测试XGBoost模型功能"""
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    
    print("="*60)
    print("Test 1: Classification Model")
    print("="*60)
    
    # 创建分类数据
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 训练分类模型
    clf_model = XGBoostModel(task_type='classification')
    clf_model.train(X_train, y_train, X_val, y_val, early_stopping_rounds=10, verbose=False)
    
    # 预测
    predictions = clf_model.predict(X_test)
    print(f"\nTest predictions (first 10): {predictions[:10]}")
    
    # 评估
    test_metrics = clf_model.evaluate(X_test, y_test)
    print(f"\nTest metrics: {test_metrics}")
    
    # 特征重要性
    print("\nTop 10 Features:")
    print(clf_model.get_feature_importance(top_n=10))
    
    # 保存模型
    clf_model.save_model('test_clf_model.pkl')
    
    print("\n" + "="*60)
    print("Test 2: Regression Model")
    print("="*60)
    
    # 创建回归数据
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练回归模型
    reg_model = XGBoostModel(task_type='regression')
    reg_model.train(X_train, y_train, verbose=False)
    
    # 预测
    predictions = reg_model.predict(X_test)
    print(f"\nTest predictions (first 10): {predictions[:10]}")
    
    # 评估
    test_metrics = reg_model.evaluate(X_test, y_test)
    print(f"\nTest metrics: {test_metrics}")
    
    # 模型摘要
    print("\nModel Summary:")
    print(reg_model.get_model_summary())


if __name__ == "__main__":
    test_xgboost_model()
