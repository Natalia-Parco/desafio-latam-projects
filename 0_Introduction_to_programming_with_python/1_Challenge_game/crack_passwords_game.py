# Muestra cuántos intentos se requieren para hackear un password por fuerza bruta
#from string import printable
from string import ascii_lowercase

password = input("Ingresa una contraseña sin la letra ñ: \n").lower()
abcd =  ascii_lowercase

# Cuenta la cantidad de intentos necesarios para hackear un password
lista = 0
for i in password:
    for j in abcd:
        if i == j :
            print(f"Elemento encontrado: {i}")
            lista += 1
            break
        else :
            lista += 1
 
print (f"Se descifro en {lista} intentos")

# contraseña = input("Ingresa una contraseña sin ñ: \n")
# abc =  printable

# # Cuenta la cantidad de intentos necesarios para hackear un password
# lista = 0
# for i in contraseña:
#     for j in abc:
#         if i == j :
#             print(f"Elemento encontrado: {i}")
#             lista += 1
#             break
#         else :
#             lista += 1
 
# print (f"Se descifro en {lista} intentos")

#Otros para ver:
#string.ascii_letters = abc minuscula y el abc en mayuscula
#string.digits = numeros del 0 al 9
#string.hexdigits = alfanumerico
