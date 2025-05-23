import pandas as pd
import matplotlib.pyplot as plt

def graficador_bar_pie(dataframe,column, dir):
    #columnas_c_faltantes = []

    valores = dataframe[column].value_counts()
    valores_faltantes = dataframe[column].isna().sum()
    name_column_aux = column.replace("/", "_")

    if valores_faltantes > 0:
        valores["Faltantes"] = valores_faltantes
        #columnas_c_faltantes.append(column, valores_faltantes)
    
    print(f'valores en la columna "{column}": {dataframe[column].unique()}\n')
    plt.figure(figsize=(17, 8)) 
    colores = plt.cm.Paired.colors
    valores.plot(kind='bar', color=colores)
    plt.title(column)
    plt.xlabel(column)
    plt.xticks(rotation=0) 
    plt.tight_layout() 
    plt.savefig(f'graficos/{dir}/{name_column_aux}_bar.png')
    #plt.show()

    plt.figure(figsize=(17, 8)) 
    valores.plot(kind='pie')
    plt.title(column)
    plt.savefig(f'graficos/{dir}/{name_column_aux}_pie.png')
    plt.close()

    if valores_faltantes > 0 : return column, valores_faltantes

def graficador_hist(dataframe,column, dir):
    plt.hist(dataframe[column], bins=30, density=True, alpha=0.6, color='skyblue')
    plt.title("Distribución")
    plt.xlabel("Valor") 
    plt.ylabel("Densidad")
    plt.grid(True)
    plt.title(column)
    plt.savefig(f'graficos/{dir}/{column}_hist.png')
    plt.close()



def plot_history(trainning, metrics:list, objetivo):
    history = trainning.history

    # Graficar la pérdida
    plt.figure(figsize=(12, 6))

    # Pérdida de entrenamiento y validación
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history['val_loss'], label='Pérdida de validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Graficar precisión
    plt.subplot(1, 2, 2)
    plt.plot(history[f'{metrics[0][0]}'], label='Precisión de entrenamiento')
    plt.plot(history[f'val_{metrics[0][0]}'], label='Precisión de validación')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    # Mostrar las gráficas
    plt.tight_layout()
    plt.savefig(f'graficos/model_performance/historial_{objetivo}.png')
    plt.show()
    plt.close()