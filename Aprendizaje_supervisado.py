import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

print("Coeficiente de determinación (R²):", r2)

y_pred = model.predict(X_test)

x_range = np.arange(len(y_test))

plt.plot(x_range, y_test, label='Valores reales')
plt.plot(x_range, y_pred, label='Predicciones')
plt.xlabel('Índice de muestra')
plt.ylabel('Progresión de la enfermedad')
plt.title('Comparación de valores reales y predicciones')
plt.legend()
plt.show()
