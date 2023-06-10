CC = mpic++ -std=c++11
OBJ = *.o
EXE = attention

all:${EXE}

attention: attention.cpp
	$(CC) -o $@ $^ $(FLAGS) 

clean:
	rm -f $(OBJ) $(EXE)