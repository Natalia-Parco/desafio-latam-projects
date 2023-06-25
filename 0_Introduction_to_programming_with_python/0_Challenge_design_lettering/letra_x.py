import math
n = int(input("Ingrese un número mayor a 2:\n"))

while n < 3:
    n = int(input("Ingrese un numero mayor a 2:\n"))

# Imprime la letra "X"
def letra_x(n):
    contain = ""
    j = 1
    # En caso de que sea par:
    if n % 2 == 0:  
        for i in range(n + 1):
            if i < (n/2):
                contain += " " * i + "*" + " " * (n-j) + "*" + "\n"
            elif i == (n/2):
                contain += " " * i + "*" + "\n"
            else:
                contain += " " * (n-i) + "*" + " " * (j-n-2) + "*" + "\n"  
            j += 2         
        return contain
    # En caso de que sea impar:    
    else:
        for i in range(n):
            if i < (n/2):
                contain += " " * i + "*" + " " * (n-j) + "*" + "\n"
            else:
                contain += " " * (n-(i+1)) + "*" + " " * (j-n) + "*" + "\n"  
            j += 2         
        return contain

print(letra_x(n))