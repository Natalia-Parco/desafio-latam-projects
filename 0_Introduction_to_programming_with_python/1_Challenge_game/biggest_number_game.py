print("Juego: El mayor de tres números")

# Pide al usuario el ingreso de tres números
N1 = float (input ("Ingrese un número: "))
N2 = float (input ("Ingrese un segundo número: "))
N3 = float (input ("Ingrese un tercer número: "))

# Muestra el mayor número ingresado
if N1 >= N2  and N1 >= N3 :
    print("El mayor número ingresado es: ", N1)
elif N2 > N1 and N2 >= N3 :
    print("El mayor número ingresado es: ", N2)
else :
    print ("El mayor número ingresado es: ", N3)
