import json
import matplotlib.pyplot as plt
from numpy import average
from scipy.stats import linregress, kendalltau

# 加載數據
with open("./saved_model/v1/predictions.json", "r") as f:
    full_data = json.load(f)
    data = full_data

    # 計算損失
    true_list = list(map(lambda x: x['true_answer']**2 - x['predicted_score']**2, data))
    loss = average(true_list)
    print(f"avg loss {loss}")

    # 提取真實值和預測值
    true = [d['true_answer'] for d in full_data]
    predicted = [d['predicted_score'] for d in full_data]

    # 執行線性回歸
    regression = linregress(true, predicted)
    slope, intercept, r_value, p_value, std_err = regression
    print(f"Regression Results:\nSlope: {slope}\nIntercept: {intercept}\nR-squared: {r_value**2}\nP-value: {p_value}")

    # 計算 Kendall Tau
    tau, tau_p_value = kendalltau(true, predicted)
    print(f"Kendall Tau: {tau}")

    # 繪製散點圖和回歸線
    plt.scatter(x=true, y=predicted, label="True vs Predicted", alpha=0.6)
    plt.plot(true, [slope * x + intercept for x in true], color="red", label=f"Regression Line (R²={r_value**2:.2f})")

    plt.xlabel('True Answer')
    plt.ylabel('Predicted Score')
    plt.title('True Answer vs Predicted Score with Regression Line')
    plt.legend()
    plt.grid(True)
    plt.savefig("./saved_model/v1/prediction_scatter_with_regression.png")
    plt.show()
