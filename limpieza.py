import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Exploracion inicial de los datos
df = pd.read_csv("ventas_con_problemas_v3.csv")
campos_numericos = ["cantidad", "precio_unitario", "total"]


def exploracion(df):
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.columns)
    print(df.isnull().sum())

    print(df['fecha'].value_counts())
    print(df['producto'].value_counts())
    print(df['cantidad'].value_counts())
    print(df['precio_unitario'].value_counts())
    print(df['total'].value_counts())


def iniciar(df):
    # exploracion(df)
    global outliers
    df=limpieza_null(df)
    df=limpieza_Fechas(df)
    df=limpiar_negativos(df)
    df=limpieza_inconsistencia(df)
    #Detectar outliers en campos numéricos
    for campo in campos_numericos:
        outliers = limpieza_valores_atipicos(df, campo)
        print(f"Outliers en {campo}:")
        print(outliers)

    df = df[~df.index.isin(outliers.index)]
    df.loc[:, 'cantidad']= df['cantidad'].clip(lower=0)

    df=limpieza_duplicados(df)
    csv_limpio(df)


def limpiar_negativos(df):
    negativos = df[df[campos_numericos].lt(0).any(axis=1)]
    print("Filas con valores negativos:")
    print(negativos)
    df = df[~df[campos_numericos].lt(0).any(axis=1)]
    df.loc[:, campos_numericos] = df[campos_numericos].abs()
    return df


def limpieza_valores_atipicos(df, columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]


def limpieza_inconsistencia(df):
    df['total_calculado'] = df['cantidad'] * df['precio_unitario']

    # Identificar discrepancias
    discrepancias = df[df['total'] != df['total_calculado']]
    print("Filas con discrepancias en el total:")
    print(discrepancias)
    df['total'] = df['total_calculado']
    return df


def limpieza_duplicados(df):
    duplicados = df[df.duplicated()]
    print("Filas duplicadas:")
    print(duplicados)
    df = df.drop_duplicates()
    return df


def limpieza_null(df):
    df = df.dropna()
    print("Filas restantes después de eliminar valores nulos:", len(df))
    return df


def limpieza_Fechas(df):
    # Convertir la columna de fecha al formato datetime
    df.loc[:, 'fecha'] = pd.to_datetime(df['fecha'], errors='coerce') #df['fecha']

    # Identificar filas con fechas no válidas (NaT)
    fechas_invalidas = df[df['fecha'].isnull()]
    print("Filas con fechas inválidas:")
    print(fechas_invalidas)

    # Manejar fechas no válidas
    # a. Eliminar filas con fechas inválidas:
    df = df.dropna(subset=['fecha'])

    # b. Rellenar fechas inválidas con una fecha predeterminada:
    # df['fecha'] = df['fecha'].fillna(pd.Timestamp("2000-01-01"))

    # Verificar fechas válidas
    print(df['fecha'].head())
    return df


def limpieza_Total(df):
    # Calcular el total esperado
    df['total_calculado'] = df['cantidad'] * df['precio_unitario']

    # Identificar discrepancias
    discrepancias = df[df['total'] != df['total_calculado']]
    print("Filas con discrepancias en el total:")
    print(discrepancias)

    # Actualizar el total con el cálculo correcto
    df['total'] = df['total_calculado']
    return df


# Creacion del CSV
def csv_limpio(df):
    # Guardar el DataFrame limpio en un archivo nuevo
    df.to_csv("archivo_limpio.csv", index=False)
    print("Archivo limpio guardado como 'archivo_limpio.csv'")

iniciar(df)

# Simulando que ya tenemos el DataFrame limpio y el original
# (carga los datos iniciales y finales según corresponda)
df_original = pd.read_csv("ventas_con_problemas_v3.csv")
df_limpio = pd.read_csv("archivo_limpio.csv")

def informe(df_original,df_limpio):



    # 1. Número de filas originales vs. después de la limpieza
    filas_originales = len(df_original)
    filas_limpias = len(df_limpio)

    # 2. Problemas identificados y corregidos
    # a) Valores nulos en campos críticos
    valores_nulos_originales = df_original.isnull().sum()
    valores_nulos_criticos = valores_nulos_originales[["fecha", "producto", "cantidad", "precio_unitario", "total"]]
    valores_nulos_corregidos = valores_nulos_criticos.sum()

    # b) Fechas inválidas
    fechas_invalidas_originales = len(df_original[pd.to_datetime(df_original["fecha"], errors="coerce").isnull()])
    fechas_invalidas_corregidas = fechas_invalidas_originales - len(
        df_limpio[pd.to_datetime(df_limpio["fecha"], errors="coerce").isnull()])

    # c) Valores negativos
    valores_negativos_originales = np.sum(df_original[["cantidad", "precio_unitario", "total"]].lt(0).values)
    valores_negativos_corregidos = valores_negativos_originales

    # d) Inconsistencias en el cálculo del total
    df_original["total_calculado"] = df_original["cantidad"] * df_original["precio_unitario"]
    inconsistencias_originales = len(df_original[df_original["total"] != df_original["total_calculado"]])
    inconsistencias_corregidas = inconsistencias_originales


    # e) Valores atípicos (outliers)
    def contar_outliers(df, columna):
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        return np.sum((df[columna] < limite_inferior) | (df[columna] > limite_superior))


    outliers_originales = sum([contar_outliers(df_original, col) for col in ["cantidad", "precio_unitario", "total"]])
    outliers_corregidos = outliers_originales

    # f) Duplicados
    duplicados_originales = df_original.duplicated().sum()
    duplicados_corregidos = duplicados_originales

    # 3. Estadísticas descriptivas de los datos limpios utilizando numpy
    estadisticas = df_limpio[["cantidad", "precio_unitario", "total"]].describe()
    media = np.mean(df_limpio[["cantidad", "precio_unitario", "total"]], axis=0)
   # mediana = np.median(df_limpio[["cantidad", "precio_unitario", "total"]], axis=0)
    mediana = df_limpio[["cantidad", "precio_unitario", "total"]].median()
    desviacion = np.std(df_limpio[["cantidad", "precio_unitario", "total"]], axis=0)

    # Generar el informe
    informe = f"""
    Informe de Limpieza de Datos:
    
    1. Número de filas:
       - Originales: {filas_originales}
       - Después de la limpieza: {filas_limpias}
    
    2. Problemas identificados y corregidos:
       - Valores nulos en campos críticos: {valores_nulos_corregidos}
       - Fechas inválidas: {fechas_invalidas_corregidas}
       - Valores negativos: {valores_negativos_corregidos}
       - Inconsistencias en el cálculo del total: {inconsistencias_corregidas}
       - Outliers en campos numéricos: {outliers_corregidos}
       - Filas duplicadas: {duplicados_corregidos}
    
3. Estadísticas descriptivas de los datos limpios:
   - Media:
     - Cantidad: {media.iloc[0]:.2f}
     - Precio Unitario: {media.iloc[1]:.2f}
     - Total: {media.iloc[2]:.2f}
   
   - Mediana:
     - Cantidad: {mediana["cantidad"]:.2f}
     - Precio Unitario: {mediana["precio_unitario"]:.2f}
     - Total: {mediana["total"]:.2f}
   
   - Desviación Estándar:
     - Cantidad: {desviacion.iloc[0]:.2f}
     - Precio Unitario: {desviacion.iloc[1]:.2f}
     - Total: {desviacion.iloc[2]:.2f}
    
    {estadisticas}
    """

    print(informe)

    # Guardar el informe en un archivo de texto
    with open("informe_limpieza.txt", "w") as archivo:
        archivo.write(informe)

informe(df_original,df_limpio)


def visualizar_datos(df_original, df_limpio):
    # Configurar el estilo de seaborn
    sns.set(style="whitegrid")

    # Crear una figura con múltiples subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Distribución de Datos Originales vs Limpios", fontsize=16)

    columnas = ["cantidad", "precio_unitario", "total"]

    for i, columna in enumerate(columnas):
        # Gráfico de distribución para los datos originales
        sns.histplot(df_original[columna], kde=True, ax=axes[i, 0], color="blue", alpha=0.6, label="Original")
        axes[i, 0].set_title(f"Distribución Original: {columna.capitalize()}")
        axes[i, 0].set_xlabel(columna.capitalize())
        axes[i, 0].set_ylabel("Frecuencia")
        axes[i, 0].legend()

        # Gráfico de distribución para los datos limpios
        sns.histplot(df_limpio[columna], kde=True, ax=axes[i, 1], color="green", alpha=0.6, label="Limpio")
        axes[i, 1].set_title(f"Distribución Limpia: {columna.capitalize()}")
        axes[i, 1].set_xlabel(columna.capitalize())
        axes[i, 1].set_ylabel("Frecuencia")
        axes[i, 1].legend()

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()

    # Comparación del número de filas
    filas_originales = len(df_original)
    filas_limpias = len(df_limpio)
    plt.figure(figsize=(6, 6))
    plt.bar(["Originales", "Limpios"], [filas_originales, filas_limpias], color=["blue", "green"], alpha=0.7)
    plt.title("Comparación del Número de Filas")
    plt.ylabel("Número de Filas")
    plt.show()

    # Boxplots para detectar outliers antes y después de la limpieza
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Boxplots de Datos Originales vs Limpios", fontsize=16)

    sns.boxplot(data=df_original[columnas], ax=axes[0], palette="Blues")
    axes[0].set_title("Datos Originales")
    axes[0].set_ylabel("Valores")
    axes[0].set_xlabel("Columnas")

    sns.boxplot(data=df_limpio[columnas], ax=axes[1], palette="Greens")
    axes[1].set_title("Datos Limpios")
    axes[1].set_ylabel("Valores")
    axes[1].set_xlabel("Columnas")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


# Llamar a la función con los datos
visualizar_datos(df_original, df_limpio)
