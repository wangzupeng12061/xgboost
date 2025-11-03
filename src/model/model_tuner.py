"""
超参数优化模块
支持网格搜索、随机搜索和贝叶斯优化
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelTuner:
    """模型调参类"""
    
    def __init__(self, 
                 task_type: str = 'classification',
                 cv_splits: int = 5,
                 scoring: str = None,
                 random_state: int = 42,
                 num_class: int = None):
        """
        初始化调参器
        
        Args:
            task_type: 任务类型 ('classification' 或 'regression')
            cv_splits: 交叉验证折数
            scoring: 评分指标
            random_state: 随机种子
            num_class: 类别数（用于多分类）
        """
        self.task_type = task_type
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.num_class = num_class
        
        # 设置默认评分指标
        if scoring is None:
            if task_type == 'classification':
                # 多分类使用accuracy，二分类使用roc_auc
                if num_class is not None and num_class > 2:
                    self.scoring = 'accuracy'
                else:
                    self.scoring = 'roc_auc'
            else:
                self.scoring = 'neg_mean_squared_error'
        else:
            self.scoring = scoring
        
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.cv_results = None
        
        print(f"ModelTuner initialized: task_type={task_type}, cv_splits={cv_splits}")
        if num_class is not None:
            print(f"  Multi-class with {num_class} classes, scoring={self.scoring}")
    
    def grid_search(self,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   param_grid: Dict = None,
                   n_jobs: int = -1,
                   verbose: int = 1) -> Dict:
        """
        网格搜索
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数网格
            n_jobs: 并行任务数
            verbose: 详细程度
            
        Returns:
            最佳参数
        """
        print("\n" + "="*60)
        print("开始网格搜索...")
        print("="*60)
        
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        print(f"参数空间大小: {self._count_param_combinations(param_grid)}")
        
        # 创建基础模型
        base_model = self._create_base_model()
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        
        # 网格搜索
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring=self.scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        # 保存结果
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.best_model = grid_search.best_estimator_
        self.cv_results = pd.DataFrame(grid_search.cv_results_)
        
        print("\n" + "="*60)
        print("网格搜索完成！")
        print("="*60)
        print(f"最佳参数: {self.best_params}")
        print(f"最佳得分: {self.best_score:.4f}")
        
        return self.best_params
    
    def random_search(self,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     param_distributions: Dict = None,
                     n_iter: int = 50,
                     n_jobs: int = -1,
                     verbose: int = 1) -> Dict:
        """
        随机搜索
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_distributions: 参数分布
            n_iter: 迭代次数
            n_jobs: 并行任务数
            verbose: 详细程度
            
        Returns:
            最佳参数
        """
        print("\n" + "="*60)
        print("开始随机搜索...")
        print("="*60)
        
        if param_distributions is None:
            param_distributions = self._get_default_param_distributions()
        
        print(f"随机采样次数: {n_iter}")
        
        # 创建基础模型
        base_model = self._create_base_model()
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        
        # 随机搜索
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=tscv,
            scoring=self.scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=self.random_state,
            return_train_score=True
        )
        
        random_search.fit(X_train, y_train)
        
        # 保存结果
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.best_model = random_search.best_estimator_
        self.cv_results = pd.DataFrame(random_search.cv_results_)
        
        print("\n" + "="*60)
        print("随机搜索完成！")
        print("="*60)
        print(f"最佳参数: {self.best_params}")
        print(f"最佳得分: {self.best_score:.4f}")
        
        return self.best_params
    
    def bayesian_optimization(self,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             param_bounds: Dict = None,
                             n_iter: int = 50,
                             init_points: int = 5) -> Dict:
        """
        贝叶斯优化（需要安装bayesian-optimization库）
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_bounds: 参数边界
            n_iter: 迭代次数
            init_points: 初始探索点数
            
        Returns:
            最佳参数
        """
        try:
            from bayes_opt import BayesianOptimization
        except ImportError:
            print("警告: bayesian-optimization未安装")
            print("请运行: pip install bayesian-optimization")
            return None
        
        print("\n" + "="*60)
        print("开始贝叶斯优化...")
        print("="*60)
        
        if param_bounds is None:
            param_bounds = self._get_default_param_bounds()
        
        # 定义目标函数
        def xgb_evaluate(**params):
            # 转换参数类型
            params_copy = params.copy()
            params_copy['max_depth'] = int(params_copy['max_depth'])
            params_copy['n_estimators'] = int(params_copy['n_estimators'])
            params_copy['min_child_weight'] = int(params_copy['min_child_weight'])
            
            # 创建模型
            if self.task_type == 'classification':
                model = XGBClassifier(
                    **params_copy,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                model = XGBRegressor(
                    **params_copy,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            
            # 交叉验证
            from sklearn.model_selection import cross_val_score
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            
            scores = cross_val_score(
                model, X_train, y_train,
                cv=tscv,
                scoring=self.scoring,
                n_jobs=-1
            )
            
            return scores.mean()
        
        # 贝叶斯优化
        optimizer = BayesianOptimization(
            f=xgb_evaluate,
            pbounds=param_bounds,
            random_state=self.random_state,
            verbose=2
        )
        
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter
        )
        
        # 获取最佳参数
        best_params = optimizer.max['params']
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['min_child_weight'] = int(best_params['min_child_weight'])
        
        self.best_params = best_params
        self.best_score = optimizer.max['target']
        
        print("\n" + "="*60)
        print("贝叶斯优化完成！")
        print("="*60)
        print(f"最佳参数: {self.best_params}")
        print(f"最佳得分: {self.best_score:.4f}")
        
        return self.best_params
    
    def _create_base_model(self):
        """创建基础模型"""
        if self.task_type == 'classification':
            if self.num_class is not None and self.num_class > 2:
                # 多分类
                return XGBClassifier(
                    objective='multi:softprob',
                    num_class=self.num_class,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                # 二分类
                return XGBClassifier(random_state=self.random_state, n_jobs=-1)
        else:
            return XGBRegressor(random_state=self.random_state, n_jobs=-1)
    
    def _get_default_param_grid(self) -> Dict:
        """获取默认参数网格（用于网格搜索）"""
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 200],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        return param_grid
    
    def _get_default_param_distributions(self) -> Dict:
        """获取默认参数分布（用于随机搜索）"""
        from scipy.stats import randint, uniform
        
        param_distributions = {
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.2),
            'n_estimators': randint(50, 300),
            'min_child_weight': randint(1, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 2)
        }
        return param_distributions
    
    def _get_default_param_bounds(self) -> Dict:
        """获取默认参数边界（用于贝叶斯优化）"""
        param_bounds = {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (50, 300),
            'min_child_weight': (1, 10),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'gamma': (0, 0.5)
        }
        return param_bounds
    
    def _count_param_combinations(self, param_grid: Dict) -> int:
        """计算参数组合数量"""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count
    
    def get_cv_results(self) -> pd.DataFrame:
        """
        获取交叉验证结果
        
        Returns:
            CV结果DataFrame
        """
        if self.cv_results is None:
            raise ValueError("尚未进行调参")
        
        return self.cv_results
    
    def plot_cv_results(self, param_name: str, figsize: Tuple[int, int] = (10, 6)):
        """
        绘制参数-得分曲线
        
        Args:
            param_name: 参数名称
            figsize: 图片大小
        """
        if self.cv_results is None:
            raise ValueError("尚未进行调参")
        
        import matplotlib.pyplot as plt
        
        results = self.cv_results.copy()
        param_col = f'param_{param_name}'
        
        if param_col not in results.columns:
            print(f"警告: 参数 {param_name} 不在结果中")
            return
        
        # 按参数分组
        grouped = results.groupby(param_col).agg({
            'mean_test_score': ['mean', 'std'],
            'mean_train_score': ['mean', 'std']
        })
        
        plt.figure(figsize=figsize)
        
        x = grouped.index
        
        # 测试集得分
        plt.plot(x, grouped['mean_test_score']['mean'], 
                'o-', label='Test Score', linewidth=2)
        plt.fill_between(x,
                        grouped['mean_test_score']['mean'] - grouped['mean_test_score']['std'],
                        grouped['mean_test_score']['mean'] + grouped['mean_test_score']['std'],
                        alpha=0.2)
        
        # 训练集得分
        plt.plot(x, grouped['mean_train_score']['mean'],
                's--', label='Train Score', linewidth=2, alpha=0.7)
        
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(f'Score vs {param_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_best_model(self):
        """获取最佳模型"""
        if self.best_model is None:
            raise ValueError("尚未进行调参")
        return self.best_model
    
    def tune_and_train(self,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      method: str = 'random',
                      **kwargs):
        """
        调参并训练最终模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            method: 调参方法 ('grid', 'random', 'bayesian')
            **kwargs: 其他参数
            
        Returns:
            最佳模型
        """
        # 调参
        if method == 'grid':
            self.grid_search(X_train, y_train, **kwargs)
        elif method == 'random':
            self.random_search(X_train, y_train, **kwargs)
        elif method == 'bayesian':
            self.bayesian_optimization(X_train, y_train, **kwargs)
        else:
            raise ValueError(f"未知的调参方法: {method}")
        
        # 使用最佳参数训练最终模型
        if self.task_type == 'classification':
            final_model = XGBClassifier(**self.best_params, random_state=self.random_state)
        else:
            final_model = XGBRegressor(**self.best_params, random_state=self.random_state)
        
        final_model.fit(X_train, y_train)
        self.best_model = final_model
        
        return final_model


def test_model_tuner():
    """测试超参数优化功能"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("="*60)
    print("测试超参数优化")
    print("="*60)
    
    # 创建测试数据
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 初始化调参器
    tuner = ModelTuner(task_type='classification', cv_splits=3)
    
    # 测试1: 网格搜索（小参数空间）
    print("\n测试1: 网格搜索")
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.1],
        'n_estimators': [50, 100]
    }
    best_params = tuner.grid_search(X_train, y_train, param_grid=param_grid, verbose=0)
    print(f"最佳参数: {best_params}")
    
    # 测试2: 随机搜索
    print("\n测试2: 随机搜索")
    tuner2 = ModelTuner(task_type='classification', cv_splits=3)
    best_params = tuner2.random_search(X_train, y_train, n_iter=10, verbose=0)
    print(f"最佳参数: {best_params}")
    
    # 评估最佳模型
    print("\n评估最佳模型:")
    best_model = tuner2.get_best_model()
    test_score = best_model.score(X_test, y_test)
    print(f"测试集准确率: {test_score:.4f}")
    
    # 查看CV结果
    print("\nCV结果汇总:")
    cv_results = tuner2.get_cv_results()
    print(cv_results[['mean_test_score', 'std_test_score']].describe())


if __name__ == "__main__":
    test_model_tuner()
