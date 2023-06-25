print("Juego: Piedra, Papel o Tijera")
import random as r

# Se pide al usuario elegir entre piedra, papel o tijera
Jugador = input("Elija Piedra, Papel o Tijera: ").lower()

# Repregunta si el ingreso que realiza el individuo es incorrecto
while Jugador != "piedra" and Jugador != "papel" and Jugador != "tijera":
    print("Argumento inválido. Debe ser: piedra, papel o tijera.")
    Jugador = input("Elija Piedra, Papel o Tijera: ").lower()

print(f"Jugaste: {Jugador}")

# La compu juega
Compu = r.randint(1,3)
if Compu == 1 :
   print("La compu jugó: piedra")
elif Compu == 2 :
   print("La compu jugó: papel")
else :
    print("La compu jugó: tijera")

# Resolución del juego
if Jugador == "piedra" and Compu == 1 or Jugador == "papel" and Compu == 2 or Jugador == "tijera" and Compu == 3 :
    print("EMPATARON")
elif Jugador == "piedra" and Compu == 2 or Jugador == "papel" and Compu == 3 or Jugador == "tijera" and Compu == 1 :
    print("PERDISTE")
else :
    print("GANASTE")

