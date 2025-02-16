import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# ==============================
# 1️⃣ 读取数据和加载模型
# ==============================

train_data_url = "https://raw.githubusercontent.com/Walkerdii/lr-nomogram/main/trainData_6.xlsx"
validation_data_url = "https://raw.githubusercontent.com/Walkerdii/lr-nomogram/main/valData_6.xlsx"

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

scaler = StandardScaler()
df_train[continuous_vars] = scaler.fit_transform(df_train[continuous_vars])
df_val[continuous_vars] = scaler.transform(df_val[continuous_vars])

X_train = df_train[all_features]
y_train = df_train[target]
X_val = df_val[all_features]
y_val = df_val[target]

# ==============================
# 3️⃣ 定义并调参 Logistic 回归模型
# ==============================

pipe = Pipeline([
    ('lr', LogisticRegression(max_iter=5000))
])

param_grid = {
    'lr__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'lr__penalty': ['l1', 'l2'],
    'lr__solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=10, scoring='roc_auc', verbose=1, n_jobs=-1, refit=True)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

fine_tuned_params = {
    'lr__C': [best_params['lr__C'] / 2, best_params['lr__C'], best_params['lr__C'] * 2],
    'lr__penalty': [best_params['lr__penalty']],
    'lr__solver': [best_params['lr__solver']]
}

fine_tuned_grid_search = GridSearchCV(pipe, param_grid=fine_tuned_params, cv=10, scoring='roc_auc', verbose=1,
                                      n_jobs=-1, refit=True)
fine_tuned_grid_search.fit(X_train, y_train)

best_model = fine_tuned_grid_search.best_estimator_
best_model.fit(X_train, y_train)

# ==============================
# 4️⃣ 绘制交互式列线图
# ==============================

def create_interactive_nomogram(coefficients, intercept, feature_names):
    # 计算每个特征的分数
    scores = coefficients * np.array([1, 1, 1, 1, 1, 1])  # 请根据具体数据和系数计算
    scores = scores + intercept
    
    # 创建 plotly 图
    fig = go.Figure()

    # 添加每个特征的条形图
    for i, score in enumerate(scores):
        feature_name = feature_names[i]
        fig.add_trace(go.Bar(
            x=[score],
            y=[feature_name],
            orientation='h',
            name=feature_name,
            marker=dict(color='royalblue')
        ))

    # 添加截距项
    fig.add_trace(go.Scatter(
        x=[intercept],
        y=['Intercept'],
        mode='markers',
        marker=dict(color='red', size=12),
        name='Intercept'
    ))

    # 调整布局和显示
    fig.update_layout(
        title="Interactive Logistic Regression Nomogram",
        xaxis_title="Score",
        yaxis_title="Features",
        showlegend=False,
        height=600
    )

    return fig

# ==============================
# 5️⃣ Streamlit 网页界面
# ==============================

st.title("Interactive Logistic Regression Nomogram")

# 用户输入
with st.form("input_form"):
    st.subheader("输入患者信息")
    age = st.number_input("AGE", min_value=0, max_value=120, value=50)
    num_diagnoses = st.number_input("Number of Diagnoses", min_value=0, max_value=20, value=5)
    excision = st.selectbox("Excision of lesion in spinal canal", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
    lumbar = st.selectbox("Lumbar puncture", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
    patient_source = st.selectbox("Patient Source", options=[1, 2, 3, 4], format_func=lambda x: f"Source {x}")
    timing_surgery = st.selectbox("Timing of Surgery", options=[1, 2, 3], format_func=lambda x: f"Surgery {x} hours")

    submitted = st.form_submit_button("预测")

# ==============================
# 6️⃣ 计算预测并显示列线图
# ==============================

if submitted:
    input_data = pd.DataFrame([[age, num_diagnoses, excision, lumbar, patient_source, timing_surgery]], columns=all_features)
    input_data[continuous_vars] = scaler.transform(input_data[continuous_vars])
    input_data['Timing_of_surgery'] = input_data['Timing_of_surgery'].map({1: 0, 2: 1, 3: 2})

    prediction_prob = best_model.predict_proba(input_data)[:, 1][0]
    st.write(f"预测预后不良的概率: {prediction_prob:.2%}")

    fig = create_interactive_nomogram(best_model['lr'].coef_[0], best_model['lr'].intercept_[0], all_features)
    st.plotly_chart(fig)
