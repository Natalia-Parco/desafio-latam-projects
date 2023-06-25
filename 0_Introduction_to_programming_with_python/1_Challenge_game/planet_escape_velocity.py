import math as m

#Demanda la gravedad del planeta del individuo
g = float (input ("Ingresa la gravedad de tu planeta en mts sobre segundos: "))

#Demanda el radio del planeta del individuo
r = float (input ("Ingresa el radio de tu planeta en metros: "))*1000

#Muestra los datos ingresados
print(f"La gravedad de su planeta es: {g} mts/seg^2")
print(f"El radio de su planeta es: {r} kms")

#Calcula la velocidad de escape del planeta
Ve = m.sqrt ( 2 * g * r )
print (f"La velocidad de escape de su planeta es de: {Ve}")