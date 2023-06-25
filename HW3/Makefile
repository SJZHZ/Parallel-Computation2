CC = g++ -std=c++11
OBJ = *.o
EXE = gaussian_blur gaussian_blur_try

all:${EXE}

gaussian_blur: gaussian_blur.cpp
	$(CC) -o $@ $^ $(FLAGS)
gaussian_blur_try: gaussian_blur_try.cpp
	$(CC) -o $@ $^ $(FLAGS) -fopenmp -Ofast -march=native

clean:
	rm -f $(OBJ) $(EXE)