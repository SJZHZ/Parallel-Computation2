CC = g++ -std=c++11
OBJ = *.o
EXE = serial persistent_T check

all:${EXE}

serial: serial.cpp
	$(CC) -o $@ $^ $(FLAGS)
destination: destination.cpp
	$(CC) -o $@ $^ $(FLAGS) -Ofast -fopenmp
collapse: collapse.cpp
	$(CC) -o $@ $^ $(FLAGS) -Ofast -fopenmp
cache: cache.cpp
	$(CC) -o $@ $^ $(FLAGS) -Ofast -fopenmp

	
pathway: pathway.cpp
	$(CC) -o $@ $^ $(FLAGS)  -fopenmp
pathway_cache: pathway_cache.cpp
	$(CC) -o $@ $^ $(FLAGS)  -fopenmp
persistent: persistent.cpp
	$(CC) -o $@ $^ $(FLAGS) -fopenmp
persistent_O: persistent.cpp
	$(CC) -o $@ $^ $(FLAGS) -Ofast -fopenmp



persistent_T_without: persistent_T.cpp
	$(CC) -o $@ $^ $(FLAGS) -fopenmp
persistent_T: persistent_T.cpp
	$(CC) -o $@ $^ $(FLAGS) -Ofast -fopenmp
check: check.cpp
	$(CC) -o $@ $^ $(FLAGS)
clean:
	rm -f $(OBJ) $(EXE)