import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def main():
    default_file = 'lab1_DS.csv'
    file_path = input(f"Введите путь к CSV файлу (нажмите Enter для использования '{default_file}'): ").strip()
    if not file_path:
        file_path = default_file

    if not os.path.exists(file_path):
        print(f"Ошибка: файл '{file_path}' не найден.")
        return

    df = pd.read_csv(file_path)
    columns = list(df.columns)

    print("\nДоступные столбцы в файле:")
    for idx, col in enumerate(columns):
        print(f"  {idx}: {col}")

    x_input = input(f"\nВведите номер или название столбца для X (Enter — '{columns[0]}'): ").strip()
    if not x_input:
        x_col = columns[0]
    elif x_input.isdigit() and int(x_input) < len(columns):
        x_col = columns[int(x_input)]
    else:
        x_col = x_input

    default_y_idx = 1 if len(columns) > 1 else 0
    y_input = input(f"Введите номер или название столбца для Y (Enter — '{columns[default_y_idx]}'): ").strip()
    if not y_input:
        y_col = columns[default_y_idx]
    elif y_input.isdigit() and int(y_input) < len(columns):
        y_col = columns[int(y_input)]
    else:
        y_col = y_input

    print("\n    Статистическая информация о данных    ")
    for col in [x_col, y_col]:
        col_data = df[col].dropna()
        print(f"Столбец '{col}':")
        print(f"  Количество (count): {len(col_data)}")
        print(f"  Минимум (min):      {col_data.min():.4f}")
        print(f"  Максимум (max):     {col_data.max():.4f}")
        print(f"  Среднее (mean):     {col_data.mean():.4f}")
        print(f"  Медиана (median):   {col_data.median():.4f}")
        print(f"  Ст. отклонение:     {col_data.std():.4f}\n")

    x = df[x_col].values
    y = df[y_col].values

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    k = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - k * x_mean

    print("Параметры регрессионной прямой (МНК)   ")
    print(f"Угловой коэффициент (k): {k:.4f}")
    print(f"Свободный член (b):      {b:.4f}")
    print(f"Уравнение прямой:        y = {k:.4f} * x + {b:.4f}\n")

    y_pred = k * x + b

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.canvas.manager.set_window_title('Лабораторная работа №1 — Линейная регрессия')

    axes[0].scatter(x, y, color='blue', label='Исходные данные')
    axes[0].set_title('1. Исходные точки')
    axes[0].set_xlabel(x_col)
    axes[0].set_ylabel(y_col)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].scatter(x, y, color='blue', label='Исходные данные')
    axes[1].plot(x, y_pred, color='red', linewidth=2, label=f'y = {k:.2f}x + {b:.2f}')
    axes[1].set_title('2. Линейная регрессия')
    axes[1].set_xlabel(x_col)
    axes[1].set_ylabel(y_col)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].scatter(x, y, color='blue', label='Исходные данные', zorder=4)
    axes[2].plot(x, y_pred, color='red', linewidth=2, label='Линия регрессии', zorder=3)
    axes[2].set_title('3. Квадраты ошибок')
    axes[2].set_xlabel(x_col)
    axes[2].set_ylabel(y_col)
    axes[2].grid(True, linestyle='--', alpha=0.6)

    x_margin = (np.max(x) - np.min(x)) * 0.15
    y_margin = (np.max(y) - np.min(y)) * 0.15
    axes[2].set_xlim(np.min(x) - x_margin, np.max(x) + x_margin)
    axes[2].set_ylim(np.min(y) - y_margin, np.max(y) + y_margin)

    x_range = axes[2].get_xlim()[1] - axes[2].get_xlim()[0]
    y_range = axes[2].get_ylim()[1] - axes[2].get_ylim()[0]
    scale_factor = x_range / y_range

    for xi, yi, ypi in zip(x, y, y_pred):
        err_y = abs(yi - ypi)
        err_x = err_y * scale_factor
        bottom_y = min(yi, ypi)
        rect = Rectangle(
            (xi, bottom_y), err_x, err_y,
            facecolor='green', edgecolor='darkgreen',
            alpha=0.25, hatch='//', zorder=2
        )
        axes[2].add_patch(rect)
        axes[2].plot([xi, xi], [yi, ypi], color='darkgreen', linestyle=':', linewidth=1, zorder=3)

    axes[2].add_patch(Rectangle((0, 0), 0, 0, facecolor='green', edgecolor='darkgreen', alpha=0.25, hatch='//', label='Квадраты ошибок'))
    axes[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()