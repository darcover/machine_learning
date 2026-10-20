import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error


df = pd.read_csv('lab2_1_DS.csv')
print("Первые 5 строк датасета:")
print(df.head())
print("\nКорреляция признаков:")
print(df.corr())

X = df[['experience']]
y = df['income']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=10.0),
    'Lasso': Lasso(alpha=100.0),
    'Elastic Net': ElasticNet(alpha=2.0, l1_ratio=0.5)
}

colors = {
    'Linear Regression': 'blue',
    'Ridge': 'red',
    'Lasso': 'green',
    'Elastic Net': 'purple'
}

linestyles = {
    'Linear Regression': '-',
    'Ridge': '--',
    'Lasso': '-.',
    'Elastic Net': ':'
}

metrics_list = []


plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='black', label='Обучающая выборка (Train)', alpha=0.7, s=50)
plt.scatter(X_test, y_test, color='orange', label='Тестовая выборка (Test)', edgecolor='black', s=70, zorder=5)

x_range = pd.DataFrame({'experience': np.linspace(X['experience'].min(), X['experience'].max(), 100)})

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_line = model.predict(x_range)
    plt.plot(x_range['experience'], y_line, color=colors[name], linestyle=linestyles[name], linewidth=2, label=name)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    metrics_list.append({
        'Модель': name,
        'Intercept (w0)': round(model.intercept_, 2),
        'Coef (w1)': round(model.coef_[0], 2),
        'MAE': round(mae, 2),
        'R2': round(r2, 4),
        'MAPE': round(mape, 4)
    })

plt.title('Линейная регрессия: зависимость дохода (income) от опыта работы (experience)')
plt.xlabel('Опыт работы (experience), лет')
plt.ylabel('Доход (income)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()


plt.savefig('regression_plot_1d.png', dpi=150)
plt.show()


results_df = pd.DataFrame(metrics_list)
print("\nСводная таблица параметров и метрик моделей на тестовой выборке:")
print(results_df.to_string(index=False))


print("\n  Множественная регрессия по двум признакам (age и experience)  ")
X_multi = df[['age', 'experience']]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.3, random_state=42)

multi_metrics = []
for name, model in models.items():
    model.fit(X_train_m, y_train_m)
    y_pred_m = model.predict(X_test_m)
    multi_metrics.append({
        'Модель': name,
        'Intercept': round(model.intercept_, 2),
        'Coef (age)': round(model.coef_[0], 2),
        'Coef (experience)': round(model.coef_[1], 2),
        'MAE': round(mean_absolute_error(y_test_m, y_pred_m), 2),
        'R2': round(r2_score(y_test_m, y_pred_m), 4),
        'MAPE': round(mean_absolute_percentage_error(y_test_m, y_pred_m), 4)
    })

print(pd.DataFrame(multi_metrics).to_string(index=False))