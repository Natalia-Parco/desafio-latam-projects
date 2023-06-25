import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def grafica_hist(df, var, sample_mean = False, true_mean = False, df_full =[], bins = 10):
    
    df[var].hist(bins=bins, color="lightblue")
    
    if sample_mean is True:
        plt.axvline(df[var].mean(), color="indigo", linestyle = "--", label = "Media submuestral")
    
    if true_mean is True:
        plt.axvline(df_full[var].mean(), color="red", linestyle = "--", label = "Media muestral")
    
    if np.mean(df[var]) > np.mean(df_full[var]):
        print(f"En la variable {var} la media submuestral es mayor a la media muestral:")
    else:
        print(f"En la variable {var} la media muestral es mayor a la media submuestral:")
    
    plt.title(f"Histograma {var}")
    plt.legend()
    
    
    

def dotplot(df, plot_var, plot_by, global_state = False, df_completo =[], statistic="mean", color="tomato", use_zscore = False):
    plt.figure(figsize=(20,10))
    
    aux = df.copy()
    col = plot_var
    if use_zscore is True:
        aux["z"] = (aux[plot_var] - aux[plot_var].mean()) / aux[plot_var].std()
        col ="z"
    
    if statistic == "mean":
        tmp = aux.groupby(plot_by)[col].agg(np.mean)
    elif statistic == "median":
        tmp = aux.groupby(plot_by)[col].agg(np.median)
        
        
    if global_state is True:
        aux_completo = df_completo.copy()
        
        if use_zscore is True:
            aux_completo["z"] = (aux_completo[plot_var] - aux_completo[plot_var].mean()) / aux_completo[plot_var].std()
        if statistic == "mean":
            plt.axhline(aux_completo[col].mean(), label = f"Media global de {col}", color=color)
        elif statistic =="median":
            plt.axhline(aux_completo[col].median(), label = f"Mediana global de {col}", color=color)

    plt.plot(tmp, marker='^', linewidth = 0,color=color, markersize=14)  
    plt.legend()
    
    
    
    
def reporte_perdidos(df, var, print_list = False):
    variable = []
    perdidos = []
    porc_datos_perdidos = []
    
    for i in var:
        perdidos_1 = df[df[i].isnull()]
        
        variable.append(i)
        perdidos.append(len(perdidos_1))
        porc_datos_perdidos.append(round((len(perdidos_1)/len(df)),2))
        
        if print_list is True:
            print(f"Lista de observaciones perdidas en la variable: {i.upper()}")
            display(perdidos_1)
                   
    tmp = pd.DataFrame({'variable': variable ,
                        'perdidos': perdidos,
                        '% de datos perdidos':porc_datos_perdidos }).sort_values(by = "perdidos", ascending = False)
    return tmp