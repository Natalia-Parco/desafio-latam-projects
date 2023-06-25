print("Programa de Letras")
n = int(input("Ingrese un número mayor a 2:\n"))

while n < 3:
    n = int(input("Ingrese un número mayor a 2:\n"))

# Imprime la letra "O".
def letra_o(n):
    for i in range(1, n+1):
        if i == 1:
            a = "*" * n + "\n"
        elif i < n:
            b = ("*" + (" " * (n-2)) + "*") + "\n"
        else:
            c = ("*" * n) + "\n"
    return a + (b*(n-2)) + c 

print(letra_o(n))