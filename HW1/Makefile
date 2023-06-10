CC = g++ -std=c++11
OBJ = *.o
EXE = serial destination persistent pathway pathway_cache collapse final check persistent_T

all:${EXE}

serial: serial.cpp
	$(CC) -o $@ $^ $(FLAGS)
destination: destination.cpp
	$(CC) -o $@ $^ $(FLAGS) -Ofast -fopenmp
pathway: pathway.cpp
	$(CC) -o $@ $^ $(FLAGS) -Ofast -fopenmp
collapse: collapse.cpp
	$(CC) -o $@ $^ $(FLAGS) -Ofast -fopenmp
pathway_cache: pathway_cache.cpp
	$(CC) -o $@ $^ $(FLAGS) -Ofast -fopenmp
persistent: persistent.cpp
	$(CC) -o $@ $^ $(FLAGS) -fopenmp
persistent_T: persistent_T.cpp
	$(CC) -o $@ $^ $(FLAGS) -Ofast -fopenmp
final: persistent.cpp
	$(CC) -o $@ $^ $(FLAGS) -Ofast -fopenmp
check: check.cpp
	$(CC) -o $@ $^ $(FLAGS)
clean:
	rm -f $(OBJ) $(EXE)