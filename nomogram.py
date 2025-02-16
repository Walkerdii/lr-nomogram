import pandas as pd
import numpy as np
import shap
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import plotly.graph_objects as go

# ==============================
# 1️⃣ 读取数据和加载模型
# ==============================

# 使用 GitHub 上的文件
train_data_url = "https://raw.githubusercontent.com/Walkerdii/lr-nomogram/main/trainData_6.xlsx"
validation_data_url = "https://raw.githubusercontent.com/Walkerdii/lr-nomogram/main/valData_6.xlsx"

# 加载数据
df_train = pd.read_excel(train_data_url)
df_val = pd.read_excel(validation_data_url)

# 定义目标变量和特征变量
target = "Prognosis"
continuous_vars = ['AGE', 'Number_of_diagnoses']
binary_vars = ["Excision_of_lesion_in_spinal_canal", "Lumbar_puncture"]
multiclass_vars = ['Patient_source', 'Timing_of_surgery']
all_features = continuous_vars + binary_vars + multiclass_vars

# ==============================
# 2️⃣ 数据预处理（标准化）
# ==============================

# 标准化连续变量
scaler = StandardScaler()
df_train[continuous_vars] = scaler.fit_transform(df_train[continuous_vars])
df_val[continuous_vars] = scaler.transform(df_val[continuous_vars])  # 验证集应用训练集的标准化

# 分离特征和标签
X_train = df_train[all_features]
y_train = df_train[target]
X_val = df_val[all_features]
y_val = df_val[target]

# ==============================
# 3️⃣ 定义并调参 Logistic 回归模型
# ==============================

# 使用管道创建Logistic回归模型
pipe = Pipeline([
    ('lr', LogisticRegression(max_iter=5000))
])

# 定义超参数网格
param_grid = {
    'lr__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'lr__penalty': ['l1', 'l2'],
    'lr__solver': ['liblinear', 'saga']
}

# 网格搜索 + 交叉验证
grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=10, scoring='roc_auc', verbose=1, n_jobs=-1, refit=True)
grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_

# 微调参数
fine_tuned_params = {
    'lr__C': [best_params['lr__C'] / 2, best_params['lr__C'], best_params['lr__C'] * 2],
    'lr__penalty': [best_params['lr__penalty']],
    'lr__solver': [best_params['lr__solver']]
}

# 微调网格搜索
fine_tuned_grid_search = GridSearchCV(pipe, param_grid=fine_tuned_params, cv=10, scoring='roc_auc', verbose=1,
                                      n_jobs=-1, refit=True)
fine_tuned_grid_search.fit(X_train, y_train)

# 获取最终模型
best_model = fine_tuned_grid_search.best_estimator_

# 训练最终模型
best_model.fit(X_train, y_train)

# ==============================
# 4️⃣ 计算 SHAP 解释
# ==============================

# 使用 SHAP 计算模型的 SHAP 值
explainer = shap.Explainer(best_model['lr'], X_train)
shap_values = explainer(X_val)


# ==============================
# 5️⃣ 创建列线图
# ==============================

def create_nomogram(coefficients, intercept, feature_names):
    # 创建空的列线图
    fig = go.Figure()

    # 添加每个特征的条形图
    for i, coef in enumerate(coefficients):
        feature_name = feature_names[i]
        fig.add_trace(go.Bar(
            x=[coef],
            y=[feature_name],
            orientation='h',
            name=feature_name
        ))

    # 添加截距线
    fig.add_trace(go.Scatter(
        x=[intercept],
        y=['Intercept'],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Intercept'
    ))

    fig.update_layout(
        title="Logistic Regression Nomogram",
        xaxis_title="Coefficient Value",
        yaxis_title="Features",
        showlegend=False
    )

    return fig


# ==============================
# 6️⃣ Streamlit 网页界面
# ==============================

st.title("Logistic 回归模型列线图")

# 用户输入
with st.form("input_form"):
    st.subheader("输入患者信息")

    # 连续变量
    age = st.number_input("AGE", min_value=0, max_value=120, value=50)
    num_diagnoses = st.number_input("Number of Diagnoses", min_value=0, max_value=20, value=5)

    # 二分类变量
    excision = st.selectbox("Excision of lesion in spinal canal", options=[0, 1],
                            format_func=lambda x: "Yes" if x else "No")
    lumbar = st.selectbox("Lumbar puncture", options=[0, 1], format_func=lambda x: "Yes" if x else "No")

    # 多分类变量
    patient_source = st.selectbox("Patient Source", options=[1, 2, 3, 4], format_func=lambda x: {
        1: "Erban core", 2: "Erban fringe", 3: "County", 4: "Countryside"
    }[x])
    timing_surgery = st.selectbox("Timing of Surgery", options=[0, 1, 2], format_func=lambda x: {
        0: "Non-surgery", 1: "Surgery within 48h", 2: "Surgery over 48h"
    }[x])

    submitted = st.form_submit_button("预测")

# ==============================
# 7️⃣ 生成预测与列线图
# ==============================

if submitted:
    # 构造输入数据
    input_data = pd.DataFrame([[age, num_diagnoses, excision, lumbar, patient_source, timing_surgery]],
                              columns=all_features)

    # 对输入数据进行标准化
    input_data[continuous_vars] = scaler.transform(input_data[continuous_vars])

    # 将 Timing_of_surgery 转换为模型计算所需的 0,1,2 格式
    input_data['Timing_of_surgery'] = input_data['Timing_of_surgery'].map({0: 0, 1: 1, 2: 2})

    # 进行预测
    prediction_prob = best_model.predict_proba(input_data)[:, 1][0]
    st.write(f"预测预后不良的概率: {prediction_prob:.2%}")

    # 创建列线图
    fig = create_nomogram(best_model['lr'].coef_[0], best_model['lr'].intercept_[0], all_features)
    st.plotly_chart(fig)
