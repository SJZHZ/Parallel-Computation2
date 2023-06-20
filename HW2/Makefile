CC = mpic++ -std=c++11
OBJ = *.o
EXE = attention attention_try attention_test

all:${EXE}

attention: attention.cpp
	$(CC) -o $@ $^ $(FLAGS) 
attention_try: attention_try.cpp
	$(CC) -o $@ $^ $(FLAGS) -Ofast -fopenmp
attention_test: attention_test.cpp
	$(CC) -o $@ $^ $(FLAGS) -Ofast -fopenmp
	
# attention_Of: attention.cpp
# 	$(CC) -o $@ $^ $(FLAGS) -Ofast 
# attention_try_no: attention_try.cpp
# 	$(CC) -o $@ $^ $(FLAGS) -fopenmp

clean:
	rm -f $(OBJ) $(EXE)