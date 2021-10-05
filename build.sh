#! /bin/bash

echo === Building project... ===

gcc -Wall -Wextra -Werror -pedantic-errors spkmeans.c -lm -o spkmeans
python3 setup.py build_ext --inplace

echo === Done ===