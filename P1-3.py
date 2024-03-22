import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Definición de la función objetivo
def f(x1, x2):
    return 10 - np.exp(-(x1**2 + 3*x2**2))

# Definición del gradiente de la función objetivo
def gradient(x1, x2):
    df_dx1 = 2 * x1 * np.exp(-(x1**2 + 3*x2**2))
    df_dx2 = 6 * x2 * np.exp(-(x1**2 + 3*x2**2))
    return np.array([df_dx1, df_dx2])

# Función de descenso del gradiente
def gradient_descent(lr, iterations, x_init):
    x = x_init # Usar valores iniciales dados
    error_history = [] # Lista para almacenar el historial de errores
    for _ in range(iterations):
        grad = gradient(x[0], x[1]) # Calcular el gradiente en el punto actual
        x -= lr * grad # Actualizar el punto usando el descenso del gradiente
        # Limitar los valores de x dentro del rango [-1, 1]
        x = np.clip(x, -1, 1)
        # Calcular el error (función objetivo) en el nuevo punto y agregarlo al historial
        error = f(x[0], x[1])
        error_history.append(error)
    return x, error_history

# Función de actualización para el slider de la tasa de aprendizaje
def update(val):
    global lr
    lr = slider.val # Actualizar la tasa de aprendizaje global con el valor del slider
    x_init = np.random.uniform(-1, 1, size=2) # Generar nuevos valores iniciales aleatorios
    # Ejecutar el descenso del gradiente con la nueva tasa de aprendizaje y los nuevos valores iniciales
    optimal_point, error_history = gradient_descent(lr, iterations, x_init)
    # Limpiar el eje y volver a trazar el gráfico con el nuevo historial de errores
    ax.clear()
    ax.plot(range(iterations), error_history)
    ax.set_title('Convergencia del Error')
    ax.set_xlabel('Iteración')
    ax.set_ylabel('Valor de la Función Objetivo')
    plt.draw()

# Parámetros
iterations = 1000  # Número de iteraciones
lr_init = np.random.uniform(0.01, 0.1)  # Tasa de aprendizaje inicial aleatoria
x_init = np.random.uniform(-1, 1, size=2)  # Valores iniciales aleatorios

# Optimización inicial
optimal_point, error_history = gradient_descent(lr_init, iterations, x_init)
optimal_value = f(*optimal_point)

# Imprimir la mejor solución y el valor óptimo encontrado
print("Mejor solución (x1, x2):", optimal_point)
print("Valor óptimo encontrado:", optimal_value)

# Crear la figura y el eje para el gráfico
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
plt.title('Convergencia del Error')
plt.xlabel('Iteración')
plt.ylabel('Valor de la Función Objetivo')

# Crear slider para ajustar lr
ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Learning Rate', 0.01, 0.1, valinit=lr_init)
slider.on_changed(update)

# Mostrar gráfico
plt.plot(range(iterations), error_history)
plt.show()
