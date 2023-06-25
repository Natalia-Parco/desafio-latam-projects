print("Concatenando Letras")
from string import ascii_lowercase
import math

g = int(input("Ingrese un número: \n"))

def gen(g):
    repeticion = math.floor( g / len(ascii_lowercase))
    resto = g % len(ascii_lowercase) 
    abc = ascii_lowercase[:g] * repeticion + ascii_lowercase [:resto]

    print(abc)
    
gen(g)


