# 当前最好（越低越好，更新请及时修改）

0.12269->0.12183->0.11289

# 代码简述

当前最优模型：stacking方案，岭回归学习器学习XGBoost、LightGBM、Lasso、ElasticNet、Ridge五种模型的最优组合

Neighborhood 目标编码：用 5 折交叉验证将每个邻域的平均对数价格作为新特征加入，提升地理位置信息的表达能力。

Stacking 元模型升级：将元模型从线性 Ridge 替换为 XGBoost（非线性），使集成能学习基模型预测间的复杂交互。

# 运行代码

库下载  
pip install pandas numpy scikit-learn xgboost lightgbm catboost mlxtend

进入项目根目录  
cd kaggle-prediction

运行预测代码  
python src/predict.py

stacking方案：运行stacking\_method/house\_prices\_solution.py

enhanced\_stacking方案：python enhanced\_stacking/enhanced\_stacking.py



# 协作方式

1、Fork目标项目  
在 GitHub 上把它fork到你自己的账号下  
2、将项目Clone到自己电脑  
3、进入项目并创建新分支  
4、保存并提交（Commit）你的修改  
5、推送到（Push）你的 GitHub 仓库  
6、发起 Pull Request (PR)，把修改请求发到我的仓库来

