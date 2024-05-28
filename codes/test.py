from typing import Callable, Concatenate, TypeVar

# Définir les variables de type
# P = ParamSpec('P')
R = TypeVar('R')  # Type de retour de la fonction cible

# Définir le wrapper
# def wrapper(func: Callable[[bool, *P.args, **P.kwargs], R]) -> Callable[[bool, *P.args, **P.kwargs], R]:
#     def wrapped_func(flag: bool, *args: P.args, **kwargs: P.kwargs) -> R:
#         # Ici, vous pouvez ajouter toute logique utilisant le flag avant d'appeler la fonction originale
#         print(f"Flag is {flag}")
#         # Appeler la fonction originale avec les arguments restants
#         return func(flag, *args, **kwargs)
#     return wrapped_func

def wrapper[**P, R](func: Callable[Concatenate[bool, P], R]) -> Callable[Concatenate[bool, bool, P], R]:
    def wrapped_func(flag1: bool, flag2: bool, /, *args: P.args, **kwargs: P.kwargs) -> R:
        # Ici, vous pouvez ajouter toute logique utilisant le flag avant d'appeler la fonction originale
        print(f"Flag is {flag1}")
        print(f"Flag is {flag2}")
        # Appeler la fonction originale avec les arguments restants
        return func(flag1, *args, **kwargs)
    return wrapped_func



@wrapper
def ma_fonction(flag: bool, a: int, b: int, c: int = 0) -> int:
    if flag:
        return a + b
    else:
        return a + b + c
    
ma_fonction(True, True, 1, 2, c=3)