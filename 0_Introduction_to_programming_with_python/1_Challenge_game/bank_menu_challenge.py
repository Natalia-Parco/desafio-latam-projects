def mostrar_menu(saldo = 0, nombre_del_banco = "Banco Amigo") :
    menu = f"Bienvenido al portal del {nombre_del_banco}.\nEscoja una opcion:\n\t1. Consultar saldo \n\t2. Hacer depósito \n\t3. Realizar giro\n\t4. Salir"

    while True:
        print(menu)
        opcion = int(input("Opcion: "))
        if opcion == 1:
            mostrar_saldo(saldo)
        elif opcion == 2:
            cantidad_a_depositar = int(input("Ingrese la cantidad a depositar:$ "))
            saldo = depositar(saldo, cantidad_a_depositar)
            mostrar_saldo(saldo)
        elif opcion == 3:
            if saldo == 0:
                print("No tiene saldo para realizar la transacción.")
            else:
                saldo_aux = False 
                cantidad_a_girar = int(input("Ingrese la cantidad que quiere girar:$ "))
                while saldo_aux is False:
                    print("No tienes saldo suficiente para realizar esta operación")
                    mostrar_saldo(saldo)
                    cantidad_a_girar = int(input("Ingrese la cantidad que quiere girar:$ "))
                    saldo_aux = girar(saldo, cantidad_a_girar)
                saldo = saldo_aux
                mostrar_saldo(saldo)
        elif opcion == 4:
            break
        else:
            print("Opción inválida. Por favor ingrese una de las opciones del menu.")
        opcion = 0
        print("\n")
    print(f"Gracias por elegirnos")
            

def depositar(saldo, cantidad):
    return saldo + cantidad

def mostrar_saldo(saldo):
    print(f"Tu saldo es de: ${saldo}")

def girar(saldo, cantidad):
    if cantidad > saldo:
        return False
    return saldo - cantidad

if __name__ == "__main__": 
    mostrar_menu()