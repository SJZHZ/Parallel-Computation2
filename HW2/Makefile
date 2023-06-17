CC = mpic++ -std=c++11
OBJ = *.o
EXE = attention attention_try

all:${EXE}

attention: attention.cpp
	$(CC) -o $@ $^ $(FLAGS) 
attention_try: attention_try.cpp
	$(CC) -o $@ $^ $(FLAGS) -Ofast

clean:
	rm -f $(OBJ) $(EXE)