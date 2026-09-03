"""
Ejecutor de pruebas automatizadas para el proyecto Traductor en Tiempo Real.
Ejecuta todas las pruebas unitarias y de integración mostrando un informe visual claro.
"""

import sys
import unittest
import time
from colorama import init, Fore, Style

init(autoreset=True)


def run_all_tests():
    print(Fore.CYAN + "=" * 65)
    print(Fore.CYAN + Style.BRIGHT + "   EJECUTOR DE PRUEBAS DEL SISTEMA (ETAPA 1)")
    print(Fore.CYAN + "=" * 65)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")

    start_time = time.perf_counter()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    duration = time.perf_counter() - start_time

    print(Fore.CYAN + "-" * 65)
    print(Fore.WHITE + Style.BRIGHT + f"Pruebas ejecutadas : {result.testsRun}")
    print(Fore.GREEN + f"Exitosas           : {result.testsRun - len(result.failures) - len(result.errors)}")

    if result.failures:
        print(Fore.RED + f"Fallidas           : {len(result.failures)}")
    if result.errors:
        print(Fore.RED + f"Errores            : {len(result.errors)}")

    print(Fore.WHITE + f"Tiempo total       : {duration:.2f} segundos")
    print(Fore.CYAN + "=" * 65)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if result.wasSuccessful():
        print(Fore.GREEN + Style.BRIGHT + " [OK] TODAS LAS PRUEBAS PASARON CORRECTAMENTE\n")
        sys.exit(0)
    else:
        print(Fore.RED + Style.BRIGHT + " [X] SE DETECTARON FALLOS EN LAS PRUEBAS\n")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
