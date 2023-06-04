CC = g++ -std=c++11
OBJ = *.o
EXE = serial destination cache persistent pathway pathway_cache

all:${EXE}

serial: serial.cpp
	$(CC) -o $@ $^ $(FLAGS)
destination: destination.cpp
	$(CC) -o $@ $^ $(FLAGS) -fopenmp
cache: cache.cpp
	$(CC) -o $@ $^ $(FLAGS) -fopenmp
persistent: persistent.cpp
	$(CC) -o $@ $^ $(FLAGS) -fopenmp
pathway: pathway.cpp
	$(CC) -o $@ $^ $(FLAGS) -fopenmp
pathway_cache: pathway_cache.cpp
	$(CC) -o $@ $^ $(FLAGS) -fopenmp

clean:
	rm -f $(OBJ) $(EXE)