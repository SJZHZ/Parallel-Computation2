#include <cstdio>
#include <mpi.h>
#include <tuple>
#include <cmath>
#include <string>
#include <string.h>

/*
    You can modify the following namespace.
*/
namespace attention {
    struct Matrix {
        int row, col;
        double **data;
        double* array2d;

        Matrix(int row, int col) : row(row), col(col)
        {
            data = new double*[row];        //不是经典的二维数组
            array2d = new double[row * col];
            for (int i = 0; i < row; ++i)
            {
                data[i] = array2d + col * i;
            }
        }

        ~Matrix()
        {
            // for (int i = 0; i < row; ++i)
            // {
            //     delete[] data[i];
            // }
            delete[] array2d;
            delete[] data;
        }
    };

    // 连续取n行
    Matrix* getlines(Matrix* a, int n, int pos)
    {
        Matrix *c = new Matrix(n, a->col);
        memcpy(c->array2d, &(a->data[pos][0]), n * a->col * sizeof(double));
        return c;
    }

    // 取转置
    Matrix* transpose(Matrix* a) {
        Matrix *c = new Matrix(a->col, a->row);
        for (int i = 0; i < a->row; ++i) {
            for (int j = 0; j < a->col; ++j) {
                c->data[j][i] = a->data[i][j];
            }
        }
        return c;
    }

    // 矩阵乘
    Matrix* matmul(Matrix *a, Matrix *b) {
        if (a->col != b->row) {
            return nullptr;
        }
        Matrix *c = new Matrix(a->row, b->col);
        for (int i = 0; i < a->row; ++i) {
            for (int j = 0; j < b->col; ++j) {
                c->data[i][j] = 0;
                for (int k = 0; k < a->col; ++k) {
                    c->data[i][j] += a->data[i][k] * b->data[k][j];
                }
            }
        }
        return c;
    }

    // 尺度变换
    Matrix* scale(Matrix *a, double s) {
        Matrix *c = new Matrix(a->row, a->col);
        for (int i = 0; i < a->row; ++i) {
            for (int j = 0; j < a->col; ++j) {
                c->data[i][j] = a->data[i][j] * s;
            }
        }
        return c;
    }

    // 按行
    Matrix* softmax(Matrix *a) {
        Matrix *c = new Matrix(a->row, a->col);
        for (int i = 0; i < a->row; ++i) {
            double sum = 0;
            for (int j = 0; j < a->col; ++j) {
                sum += exp(a->data[i][j]);
            }
            for (int j = 0; j < a->col; ++j) {
                c->data[i][j] = exp(a->data[i][j]) / sum;
            }
        }
        return c;
    }

    Matrix* attention(Matrix *q, Matrix *k, Matrix *v)
    {
        int size, rank, begin, chunk, n;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        chunk = (q->row + size - 1) / size;
        begin = chunk * rank;
        n = chunk < (q->row - begin) ? chunk : (q->row - begin);
        Matrix *qq = getlines(q, n, begin);
        
        Matrix *kt = transpose(k);
        Matrix *qk = matmul(qq, kt);
        Matrix *qk_s = scale(qk, 1.0 / sqrt(k->col));
        Matrix *qk_s_s = softmax(qk_s);
        Matrix *qkv = matmul(qk_s_s, v);

        Matrix *ans = new Matrix(q->row, v->col);

        // if (rank == 0)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         memcpy(ans->data[j], qkv->data[j], sizeof(double) * qkv->col);
        //     }

        //     for (int i = 1; i < size; i++)
        //         for (int j = 0; j < n; j++)
        //         {
        //             MPI_Recv(ans->data[i * n + j], qkv->col, MPI_DOUBLE, i, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //         }
            
        //     for (int j = 0; j < qkv->row; j++)
        //     {
        //         // MPI_Bcast(ans->data[j], qkv->col, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //     }
        // }
        // else
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         MPI_Send(qkv->data[j], qkv->col, MPI_DOUBLE, 0, j, MPI_COMM_WORLD);
        //     }
        // }
        return qkv;
    }
}

attention::Matrix* read_matrix(FILE *f) {
    int row, col;
    fscanf(f, "%d %d", &row, &col);
    attention::Matrix *m = new attention::Matrix(row, col);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; j++) {
            fscanf(f, "%lf", &m->data[i][j]);
        }
    }
    return m;
}


std::tuple<attention::Matrix*, attention::Matrix*, attention::Matrix*, double> prepare(std::string filename) {
    auto f = fopen(filename.c_str(), "r");
    int row, col;
    double ans;
    attention::Matrix *q = read_matrix(f);
    attention::Matrix *k = read_matrix(f);
    attention::Matrix *v = read_matrix(f);
    fscanf(f, "%lf", &ans);
    fclose(f);
    return std::make_tuple(q, k, v, ans);
}

/*
    You CANNOT modify the following function.
    This function reduces the sum of the matrix.
*/
double reduce_the_sum(attention::Matrix* qkv) {
    double sum = 0;
    for(int i = 0; i < qkv->row; i++){
        for(int j = 0; j < qkv->col; j++){
            sum += qkv->data[i][j];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sum;
}

/*
    You CANNOT modify the following function.
    This function checks the answer.
*/
bool check(double qkv_ans, double ans) {
    return fabs(qkv_ans - ans) < 1e-6;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Read the filename.
    std::string filename(argv[1]);

    // Prepare for matrices and the ans.
    attention::Matrix *q, *k, *v;
    double ans;
    std::tie(q, k, v, ans) = prepare(filename);
    if (rank == 0) {
        printf("Matrix size: %d x %d\n", q->row, q->col);
        printf("Ans: %.10lf\n", ans);
    }

    // Start attention.
    auto start = MPI_Wtime();
    auto qkv = attention::attention(q, k, v);
        MPI_Barrier(MPI_COMM_WORLD);
    auto end = MPI_Wtime();

    // Reduce the answer
    double qkv_ans = reduce_the_sum(qkv);
    if (rank == 0) {
        printf("Your answer: %.10lf\n", qkv_ans);

        // Check the answer.
        bool correct = check(qkv_ans, ans);

        // Output the result.
        if (correct) {
            printf("Correct! Time: %.10lf\n", end - start);
        } else {
            printf("Wrong!\n");
        }
    }
    MPI_Finalize();

    return 0;
}