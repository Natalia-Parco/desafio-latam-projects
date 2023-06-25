print("Programa de Letras")
import math

n = int(input("Ingrese un número mayor a 2:\n"))

while n < 3:
    n = int(input("Ingrese un numero mayor a 2:\n"))

# Imprime la letra I. Diferencia entre si el número ingresado es par o impar.
def letra_i(n): 
    for i in range( 1, n + 1):
   
        linea_vertical = " " * (math.floor(n/2)) + "*" + "\n"
        linea_horizontal_impar = "*" * n + "\n"
        linea_horizontal_par = "*" * int(n/2) + "*" + "*" * int(n/2) + "\n"
   
        if n % 2 != 0:
            return linea_horizontal_impar +  linea_vertical * (n-2) + linea_horizontal_impar   
        else:
            return linea_horizontal_par + linea_vertical * (n-2) + linea_horizontal_par

print(letra_i(n))           