import numpy as np
import pandas as pd
import unidecode

import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import missingno as msngo

from pandas.plotting import scatter_matrix #matriz diagrama de dispersion con histograma en la diagonal principal
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score



################################################ EDA: ANÁLISIS EXPLORATORIO DE DATOS

# 1. ESTANDARIZACIÓN DEL NOMBRE DE LAS COLUMNAS DEL DATAFRAME.

# De esta forma se evita la pérdida de tiempo buscando el nombre original de la columna, si tiene espacio - mayúsculas, etc.

def remove_accents(a): 
    """Esta función reemplaza el nombre de las columnas del dataframe si tienen grado, apóstrofe o un punto por un guión bajo.
         Args: 
            columns: df.columns
         Returs:
            Una transformación que estandariza el nombre de las columnas del dataframe.  
    """
    a = a.replace('°',  '')
    a = a.replace("'",  '')
    a = a.replace(".",  '_')
    return unidecode.unidecode(a)  

def process_cols(columns):
    """Esta función transforma el nombre de las columnas del dataframe sin mayúsculas, reemplaza los espacios blanco con guion bajo y el guion medio con guion bajo.
         Args: 
            columns: df.columns
         Returs:
            Una transformación que estandariza el nombre de las columnas del dataframe.  
    """
    columns = columns.str.lower()   
    columns = columns.str.strip()   
    columns = columns.str.strip('.')   
    columns = columns.str.replace(' ', '_')
    columns = columns.str.replace('-', '_') 
    columns = [remove_accents(x) for x in columns]
    return columns



# 2. VALORES PERDIDOS

def datos_faltantes(df):
    """ Esta función muestra la cantidad de datos faltantes en el dataframe.
         Args: 
             data: dataframe con el que se está trabajando. (df)
         Returs:
             El total de valores pedidos por columna.
             El porcentaje de valores perdidos.
             El tipo de datos.
    """ 
    total = df.isnull().sum()
    porcentaje = round((total/df.shape[0])*100,2)
    tt = pd.concat([total, porcentaje], axis=1, keys=['Total', 'Porcentaje'])
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    tt['Tipo de Dato'] = types
    tt = tt[tt['Total'] != 0]
    tt =tt.sort_values('Total',ascending=False) 
    display(tt)
 
    print(color.azul +                                                     "MATRIZ DE VALORES PERDIDOS"+color.fin)
    plt.figure(figsize = (4,4))
    msngo.matrix(df.replace([8, 9], [np.nan, np.nan]))
    plt.show()
    print(color.azul +                                                     "CANTIDAD DE DATOS PERDIDOS POR VARIABLE"+color.fin)
    msngo.bar(df, color = "dodgerblue", sort = "ascending", fontsize = 12)
    plt.show()
    print(color.azul +                                                     "GRADO DE ASOCIACION ENTRE DATOS PERDIDOS POR VARIABLES"+color.fin)
    msngo.heatmap(df)
    plt.show()
    
    df_1 = df.dropna()
    perdida_de_muestra = ((df.shape[0] - df_1.shape[0])/df.shape[0])* 100
    print(f"\n\nSí eliminamos los datos faltantes perdemos el {round(perdida_de_muestra,2)}% de la muestra") 
    
    
    
    
# 3. RECODIFICACIÓN DE VARIABLES

## Recodificación de una variable
def recodificacion(df, vble, renombrar_por, reemplazar, reemplazar_por):
    """ Esta función recodifica una variable.
         Args: 
             df: dataframe con el que se está trabajando. (df)
             vble: la variable que se quiere recodificar.(string)
             renombrar_por: la variable (string)
             reemplazar: los valores que se quieren modificar (list)
             reemplazar_por: valor/nombre que se quiere considerar (list)
         Returs:
             Nuevo df con las modificaciones buscadas.
    """
    print(f"Los valores que asume la variable {vble}/{renombrar_por} originalmente son: \n")
    print(df[vble].value_counts())
    print("                                 ") 
    df = df.rename(columns={vble:renombrar_por})
    df[renombrar_por] = df[renombrar_por].replace(reemplazar, reemplazar_por)
    print(f"La variable transformada {vble}/{renombrar_por} asume los nuevos valores de: \n")
    print(df[renombrar_por].value_counts())
    print("----------------------------------------------------------------------------------- ")  
    return df 
   
## Recodificación binaria de una variable    
def recodificacion_binaria(df, vbles):
    """ Esta función recodifica una variable binaria asignandole el criterio de 1 a aquellas categorías minoritarias.
         Args: 
             df: dataframe con el que se está trabajando. (df)
             vbles: las variables que se quiere recodificar.(string)
         Returs:
             Nuevo df con variables recodificada de forma binaria.
    """ 
    for i in vbles:
        print(f"Los valores que asume la variable {i} originalmente son: \n")
        print(df[i].value_counts())
        print("                                 ") 
        df[i] = np.where(df[i] == df[i].value_counts().index[0], 0, 1)
        print(f"La variable transformada {i} asume los nuevos valores de: \n")
        print(df[i].value_counts())
        print("----------------------------------------------------------------------------------- ")
    return df 

# Transformación numérica
def transformacion_numerica(df, vble):
    """ Esta función transforma a números una columna numérica que esta considerada bajo una estructura de datos genérica.
         Args: 
             df: dataframe con el que se está trabajando. (df)
             vble: es la variable que esta como string que queremos transformar a número
         Returs:
             La variable transformada de object a float
            
    """    
    df[vble] = df[vble].str.replace('"','')
    df[vble] = pd.to_numeric(df[vble])
    return df 

################################################ MEDIDAS DESCRIPTIVAS DE LOS DATOS


# Diferenciar entre variables cualitativas y cuantitativas
def medidas_descriptivas(df):
    """ Esta función presenta las medidas descriptivas de cada variable.
         Args: 
             df: dataframe con el que se está trabajando. (df)
         Returs:
             Medidas descriptivas
    """ 
    print(color.azul + f"                                 VARIABLES CUANTITATIVAS \n                        "+ color.fin)
    display(round(df.describe().transpose(),2))
    print(color.azul + f"                                 VARIABLES CUALITATIVAS \n                        "+ color.fin)
    for col in df.columns:
        if df[col].dtype == "object":
            print(f"{col.upper()}")
            print("----------")
            print(round(df[col].value_counts(),2),"\n") 
            print("----------")
    
            
           
        
################################################ GRAFICOS DE LAS VARIABLES

def grafico_vbles(df):
    """ Esta función gráfica las variables según sean cualitativas o cuantitativas.
         Args: 
             df: dataframe con el que se está trabajando. (df)
         Returs:
             Gráficos con los comportamientos de las variables.
    """ 
    for n, i in enumerate(df):
        j = n + 1
        plt.subplots_adjust(left = 3,right = 7,bottom = 7, top = 25,wspace = 0.2, hspace = 0.2)
        plt.subplot(17, 2 , j)
        #plt.figure(figsize=(5,3))
        
        # Graficos de barras para las categórica
        if type(df[i][2]) == str:
            sns.countplot(y = df[i].dropna(), palette="pastel")
            plt.title(i.upper(), fontsize= 25)
            plt.xlabel("")
            
        else:
        # Histograma para las cuantitativas
            if len(df[i].value_counts()) > 2:
                sns.distplot(df[i].dropna())
                plt.title(i.upper(), fontsize= 25)
                plt.xlabel("")
            else:
                sns.countplot(y = df[i].dropna(), palette="Set3")
                plt.title(i.upper(), fontsize= 25)
                plt.xlabel("")


################################################ CORRELACION 

def correlacion(df, y_pred, correlacion):
    """ Esta función presenta las correlaciones entre todas las variables del dataframe. De la misma podemos identificar que variables están correlacionadas fuertemente y quitarlas del análisis para no obtener problemas de autocorrelacion.
     También podemos calcular las correlaciones con las variables que queremos proyectar.
         Args: 
             df: dataframe con el que se está trabajando. (df)
         Returs:
         - La correlación de la variable objetivos con las variables exógenas.
         - La correlación de la variable objetivo con las variables exógenas de mayor asociación lineal.
         - Gráfico de calor de todas las correlaciones
         
            
    """  
    # Coeficiente de Pearson
    pearson = round(pd.DataFrame(df.corr(method="pearson")),2)
    pearson_1 = pearson[y_pred]
    print(color.azul + f"                                   CORRELACION ENTRE {y_pred.upper()} Y LAS VARIABLES \n                        "+ color.fin)
    print(f"{pearson_1}\n")
    print(color.azul + f"                                 CONSIDERANDO ÚNICAMENTE LAS CORRELACIONES MAYORES A {round(correlacion,2)}\n"+ color.fin)
    mayores_a = pearson_1[np.abs(pearson_1) > correlacion]
    print(f"{mayores_a}\n\n")
    
    
    pearson_2 = pearson[np.abs(pearson) > correlacion].fillna(" ")
    print(color.azul + f"                                FUERZA DE ASOCIACIÓN ENTRE LAS VARIABLES MAYOR A {correlacion}                         \n" + color.fin)
    display(pearson_2)
    print(color.celeste +"Excluir del análisis la que explique menos el comportamiento de la variable a predecir. Debido a que, de los contrario, podría existir un fuerte problema de autocorrelacion en la predicción.\n\n"+ color.fin)
    
    
    
    # Gráfico de calor de todas las correlaciones    
    plt.figure(figsize = (20, 20))
    corr = sns.heatmap(round(df.corr(),2), cmap='Blues', annot=True, annot_kws={"size": 13})
    print(color.azul + "                                             GRÁFICO DE CALOR                                        " + color.fin)
    plt.show()
    return corr 
    
################################################ MODELOS ECONOMÉTRICOS
###################################################################### REGRESION LINEAL
def relacion_lineal(df, y_pred):
    """ Esta función presenta la relación existente entre 'y' predicha y las variables explicativas. Como estamos en un modelo de regresión lineal, la relación existente debe serlo también. De lo contrario se debe transformar la variable.
         Args: 
             df: dataframe con el que se está trabajando. (df)
             y_pred: es la variable que buscamos predecir
         Returs:
            Un gráfico de regresión lineal
    """ 
    for i in df.columns:
        sns.regplot(f"{i}", y_pred , data = df, color='darkviolet',marker="+")
    return plt.show()
  
def regresion_lineal(df, y_pred, X_drop, n, p_value, cte=False, correccion = False, plot = False):
    
    """ Esta función presenta
         Args: 
             df: dataframe con el que se está trabajando. (df)
             y_pred: es la variable que buscamos predecir.
             X_drop: lista con y_pred y las variables que no son significativas.
             n: 0 = Indicadores OLS 
                1 = Indicadores de las variables
                2 = todos los indicadores
             cte = ordenada al origen
             correccion = variables no significativas.
             
         Returs:
              Regresion Lineal
    """ 
    y = df[y_pred]
    X = df.drop(X_drop, axis = 1)
    
    if cte == True:
        X = sm.add_constant(X)

    model = sm.OLS(y,X)
    model = model.fit()
    results= model.summary()
    tabla = model.summary2().tables[1]
    coeficientes = pd.DataFrame(tabla[['Coef.']]).transpose()
    filtrar = tabla[tabla['P>|t|'] > p_value].sort_values('P>|t|',ascending =False)
    lista = list(filtrar.index)   
    
    if n == 2:
        display(results)
    else:
        display(results.tables[n])
    
    if correccion == True:
        print(color.violeta + "                  VARIABLES NO SIGNIFICATIVAS" + color.fin)
        display(filtrar)
        print(color.violeta + f"\nVariables a excluir:  {lista}\n\n" + color.fin)   
        
    #Limpieza de variables no significativas
        lista_resultados = []
        for i in lista:
            drop = X_drop
            drop.append(i)
            y = df[y_pred]
            X = df.drop(drop, axis = 1)
            if cte == True:
                X = sm.add_constant(X)

            model_1= sm.OLS(y,X).fit()
            print(f"\n\nSí eliminamos la variable {i.upper()} nos quedan como:\n")
            print(color.violeta + "                  VARIABLES NO SIGNIFICATIVAS" + color.fin)
            tabla_1 = model_1.summary2().tables[1]
            filtrar_1 = tabla_1[tabla_1['P>|t|'] > p_value].sort_values('P>|t|',ascending =False)
            display(filtrar_1)
    
    
            lista_resultados.append([i, round(model_1.rsquared_adj,4), round(model_1.aic,2), round(model_1.bic,2)])

    
        df_mrl = pd.DataFrame(lista_resultados)
        print("\n\nSÍ CON LA ELIMINACIÓN DE CADA VARIABLE NO SIGNIFICATIVA")
        print(color.rojo + "DISMINUYE EL CRITERIO DE AIC-BIC Y SE INCREMENTA EL R2 ADJ " + color.fin)
        print("           ESTAMOS FRENTE A UN MEJOR MODELO !!!")
        df_mrl.columns = ['Vble Eliminada','R Cuadrado Ajustado', 'AIC', 'BIC']
        display(df_mrl)
        
        
    if plot == True:
        y_hat = model.predict(df.drop(X_drop, axis = 1))
        print(color.violeta + "\n\n                                           REAL VS PREDICCIÓN" + color.fin)
        plt.figure(figsize=(18,7))
        plt.plot(y, color='indigo', label = "Real" )
        plt.plot(y_hat, color='orange', label = "Predicción")
        plt.legend()
        plt.show()
        
        df_pred = df[[y_pred]]
        df_pred["Modelo"] = y_hat
        df_pred["Desvios"] = y - df_pred["Modelo"]
        df_pred["%Desvios"] = (y / df_pred["Modelo"]- 1)*100

        display(df_pred.head(5)) 
        print(f"\n\nEl modelo tiene un desvio promedio de: {round(np.abs(df_pred['%Desvios']).mean(),2)}")
        display(coeficientes)
    
    return model
   
###################################################################### REGRESION LOGISTICA
def concise_summary(df, modelo, y_pred, X_drop, p_value, print_fit = True, correccion = True):
    """ Esta función presenta
         Args: 
             df: dataframe con el que se está trabajando. (df)
             modelo: modelo que se esta utilizando.
             y_pred: es la variable que buscamos predecir.
             X_drop: lista con y_pred y las variables que no son significativas.
             p_value: p value buscado.
             correccion = variables no significativas.
             print_fit: estadisticas
             
         Returs:
             Parámetros asociados a estadísticas de ajuste
             Parámetros estimados por cada regresor.
             Estadisticas de Bondad de Ajuste
             Estimación puntual
             Corección del modelo
         
    """
          
    fit = pd.DataFrame({'Statistics': modelo.summary2().tables[0][2][2:],
                        'Value': modelo.summary2().tables[0][3][2:]})
    estimates = round(pd.DataFrame(modelo.summary2().tables[1].loc[:, 'Coef.': 'Std.Err.']),2)
    estimates["Puntaje_Z"] = round(estimates["Coef."] / estimates["Std.Err."],2)
    # imprimir fit es opcional
    if print_fit is True:
        print(color.violeta +"\n                                  ESTADÍSTICAS DE BONDAD DE AJUSTE\n"+ color.fin)
        display(fit)
    print(color.violeta +"                                      ESTIMACIÓN PUNTUAL\n\n"+color.fin)
    display(estimates)
    
    
    print(color.violeta + " \n\n                                  VARIABLES NO SIGNIFICATIVAS" + color.fin)
    _sumary = modelo.summary2().tables[1]
    _sumary = _sumary[_sumary['P>|z|']> p_value].sort_values('P>|z|',ascending =False) 
    display(_sumary)
    
    lista = list(_sumary.index)  
    
    if correccion == True:
        print(color.violeta + f"\nVariables a excluir:  {lista}\n\n" + color.fin)   
        
def formula(df, vble_y , drop):
    """ Esta función presenta
         Args: 
             df: dataframe con el que se está trabajando. (df)
             y_pred: es la variable que buscamos predecir.
             drop: lista con y_pred y las variables que no son significativas.
             
         Returs:
             Formula del modelo
         
      """
    
    preffix = vble_y +" ~ "
    formula = ""

    for i in df.drop(columns=drop).columns:
        formula +=f"{i} + "
    formula = formula.strip(" + ")

    return preffix + formula
        
def filtrado(modelo, trae_mayor_a):
    rtado = modelo.summary2().tables[1]
    display(rtado)
    
    filtrar = rtado[rtado['P>|z|']>trae_mayor_a].sort_values(by = 'P>|z|',ascending= False)
    print(f"\nLas variables a quitar son: {list(filtrar.index)}")
    return filtrar  
  
def inverse_logit(log_odds):
    return np.exp(log_odds) / ( 1 + np.exp(log_odds))

################################################ MODELOS DE MACHINE LEARNING

###################################################################### REGRESION LINEAL
def regresion_linealML(df,y_pred, test_size=.33,random_state=2054):
    
    """ Esta función presenta
         Args: 
             df: dataframe con el que se está trabajando. (df)
             y_pred: es la variable que buscamos predecir.

             
         Returs:
            
    """ 
    X, y = df.drop([y_pred], axis = 1), df[y_pred]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=random_state)

    model_1 = LinearRegression(fit_intercept = True, normalize = True).fit(X_train, y_train)  # Y = a + bX 
    model_2 = LinearRegression(fit_intercept = False, normalize = True).fit(X_train, y_train) # Y = bX
    model_3 = LinearRegression(fit_intercept = False, normalize = False).fit(X_train, y_train)

    modelos = ["REGRESION LINEAL 1(c/i y normalizado)","REGRESION LINEAL 2(s/i y normalizado)","REGRESION LINEAL 3(s/i y sin normalizar)"]


    for i, model in enumerate([model_1, model_2, model_3]):
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        print(color.underline + "MODELO" + color.fin)
        print(color.azul + f"                              {modelos[i]}"+ color.fin)
        
    # Regression metrics
        r2 = metrics.r2_score(y_test, y_test_pred)
        mean_absolute_error = metrics.mean_absolute_error(y_test, y_test_pred) 
        mse = metrics.mean_squared_error(y_test, y_test_pred) 
        #mean_squared_log_error = metrics.mean_squared_log_error(y_test, y_test_pred)
        #median_absolute_error = metrics.median_absolute_error(y_test, y_test_pred)
        #explained_variance = metrics.explained_variance_score(y_test, y_test_pred)
        

        #print("-------------------------------")
        print('R2: ', round(r2,3))
        print('MAE: ', round(mean_absolute_error,3))
        print('MSE: ', round(mse,3))
        #print('RMSE: ', round(np.sqrt(mse),2))
        #print("-------------------------------")   
    
        plt.figure(figsize = (10,5))
        plt.subplot(1,2,2)
        sns.distplot(y_train - y_train_pred, bins = 20, label = 'train')
        sns.distplot(y_test - y_test_pred, bins = 20, label = 'test')
        plt.xlabel('Errores')
        plt.legend()
        
        ax = plt.subplot(1,2,1)
        ax.scatter(y_test,y_test_pred, s =40)
        lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes]
        ]  
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        plt.xlabel('y (test)')
        plt.ylabel('y_pred (test)')   
        plt.tight_layout()
        plt.show()
    
    # COEFICIENTES
    print(color.azul +"\n\nEl valor que asumen los coeficientes en cada uno de los modelos es:"+ color.fin)
    coef = pd.DataFrame(df.columns).drop(0)
    coef = coef.rename(columns = {0:"Variables"})
    coef["Coeficientes_1"] = model_1.coef_
    coef["Coeficientes_2"] = model_2.coef_
    coef["Coeficientes_3"] = model_3.coef_
    display(coef)
    
    # INTERCEPTO
    print(color.azul +"\nEl valor que asumen el intercepto en cada uno de los modelos es:"+ color.fin)
    print("\nEl intercepto del modelo 1 es: ", round(model_1.intercept_ ,2))
    print("El intercepto del modelo 2 es: ", round(model_2.intercept_ ,2))
    print("El intercepto del modelo 3 es: ", round(model_3.intercept_ ,2))
    
    
    y_hat_ML1 = model_1.predict(X_train)
    y_hat_ML2 = model_2.predict(X_train)
    y_hat_ML3 = model_3.predict(X_train)
    
    modelo = pd.DataFrame(y_train)
    modelo["Modelo1_ML"] = y_hat_ML1
    modelo["Modelo2_ML"] = y_hat_ML2
    modelo["Modelo3_ML"] = y_hat_ML3
    
    modelo["Desvio1"] = modelo[y_pred] - modelo["Modelo1_ML"]
    modelo["Desvio2"] = modelo[y_pred] - modelo["Modelo2_ML"]
    modelo["Desvio3"] = modelo[y_pred] - modelo["Modelo3_ML"]
    
    modelo["%Desvios1"] = ( modelo[y_pred] / modelo["Modelo1_ML"]- 1 )*100
    modelo["%Desvios2"] = ( modelo[y_pred] / modelo["Modelo2_ML"]- 1 )*100
    modelo["%Desvios3"] = ( modelo[y_pred] / modelo["Modelo3_ML"]- 1 )*100
    
    #display(modelo.head(2))
    print(color.azul +"\n\n                                          DESVIO PROMEDIO\n"+ color.fin)
    print(f"El modelo 1 tiene un desvio promedio de: {round(np.abs(modelo['%Desvios1']).mean(),3)}")
    print(f"El modelo 2 tiene un desvio promedio de: {round(np.abs(modelo['%Desvios2']).mean(),3)}")
    print(f"El modelo 3 tiene un desvio promedio de: {round(np.abs(modelo['%Desvios3']).mean(),3)}")

    
###################################################################### REGRESION LOGISTICA   

    


class color:
    violeta= '\033[95m'
    celeste = '\033[96m'
    azul = '\033[94m'
    verde = '\033[92m'
    amarillo = '\033[93m'
    rojo = '\033[91m'
    negrita = '\033[1m'
    underline = '\033[4m'
    fin = '\033[0m'

   

    
    
    
    
    
    
    
    