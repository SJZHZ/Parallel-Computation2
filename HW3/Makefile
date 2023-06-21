CC = g++ -std=c++11
OBJ = *.o
EXE = gaussian_blur

all:${EXE}

gaussian_blur: gaussian_blur.cpp
	$(CC) -o $@ $^ $(FLAGS) 

clean:
	rm -f $(OBJ) $(EXE)