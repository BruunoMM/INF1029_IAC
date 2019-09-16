build:
	gcc -o matrix_lib_test matrix_lib_test.c matrix_lib.c -mfma 

run:
	./matrix_lib_test 2 8 8 8 8 floats_256_2.0f.dat floats_256_5.0f.dat result1.dat result2.dat