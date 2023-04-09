import math
import multiprocessing
import os
import sys
import platform
from concurrent.futures import ProcessPoolExecutor

from qgis._core import QgsProcessingFeedback

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]


def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True


def parallelize():
    if platform.system() == 'Windows':
        multiprocessing.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))
    with ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            feedback.pushInfo('%d is prime: %s' % (number, prime))


