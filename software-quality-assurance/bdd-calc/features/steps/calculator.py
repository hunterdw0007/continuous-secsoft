def add(a, b):
    try:
        return int(a) + int(b)
    except ValueError:
        return "Error: Invalid Input"

def sub(a, b):
    try:
        return int(a) - int(b)
    except ValueError:
        return "Error: Invalid Input"

def mult(a, b):
    try:
        return int(a) * int(b)
    except ValueError:
        return "Error: Invalid Input"

def div(a, b):
    try:
        return int(a) / int(b)
    except ZeroDivisionError:
        return 'Error: Division By Zero'
    except ValueError:
        return 'Error: Invalid Input'