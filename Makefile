build:
	gcc -Wall -o matrix_lib_test matrix_lib_test.c -mfma 

run:
	./matrix_lib_test 2 3 3 3 3