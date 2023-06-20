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
            data = new double*[row];
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
        memcpy(c->array2d, &(a->array2d[pos * a->col]), n * a->col * sizeof(double));
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

    Matrix* matmul_T_cache(Matrix *a, Matrix *b)
    {
        if (a->col != b->col)
            return nullptr;
        int nn = a->col;
        int mm = b->row;
        Matrix *c = new Matrix(a->row, b->row);
        for (int j = 0; j < mm - 7; j += 8)
        {
            for (int i = 0; i < a->row; ++i)
            {
                c->data[i][j] = 0;
                c->data[i][j + 1] = 0;
                c->data[i][j + 2] = 0;
                c->data[i][j + 3] = 0;
                c->data[i][j + 4] = 0;
                c->data[i][j + 5] = 0;
                c->data[i][j + 6] = 0;
                c->data[i][j + 7] = 0;
                // int ci = i * mm + j, ai = i * nn, bi = j * nn;
                // c->array2d[ci] = 0;
                // c->array2d[ci + 1] = 0;
                for (int k = 0; k < nn; ++k)
                {
                    double aik = a->data[i][k];
                    c->data[i][j] += aik * b->data[j][k];
                    c->data[i][j + 1] += aik * b->data[j + 1][k];
                    c->data[i][j + 2] += aik * b->data[j + 2][k];
                    c->data[i][j + 3] += aik * b->data[j + 3][k];
                    c->data[i][j + 4] += aik * b->data[j + 4][k];
                    c->data[i][j + 5] += aik * b->data[j + 5][k];
                    c->data[i][j + 6] += aik * b->data[j + 6][k];
                    c->data[i][j + 7] += aik * b->data[j + 7][k];
                    // c->array2d[ci] += a->array2d[ai + k] * b->array2d[bi + k];
                    // c->array2d[ci + 1] += a->array2d[ai + k] * b->array2d[bi + nn + k];
                }
            }
        }
        //TODO: not devided by 8
        for (int j = std::max(mm - 8, 0); j < mm; ++j)
        {
            for (int i = 0; i < a->row; ++i)
            {
                c->data[i][j] = 0;
                // c->array2d[i * mm + j] = 0;
                for (int k = 0; k < a->col; ++k)
                {
                    c->data[i][j] += a->data[i][k] * b->data[j][k];
                    // c->array2d[i * mm + j] += a->array2d[i * nn + k] * b->array2d[j * nn + k];
                }
            }
        }
        return c;
    }

    Matrix* matmul_T(Matrix *a, Matrix *b)
    {
        if (a->col != b->col)
            return nullptr;
        int nn = a->col;
        int mm = b->row;
        Matrix *c = new Matrix(a->row, b->row);
#pragma omp parallel for
        for (int j = 0; j < b->row; ++j)
        {
            for (int i = 0; i < a->row; ++i)
            {
                c->data[i][j] = 0;
                // c->array2d[i * mm + j] = 0;
                for (int k = 0; k < a->col; ++k)
                {
                    c->data[i][j] += a->data[i][k] * b->data[j][k];
                    // c->array2d[i * mm + j] += a->array2d[i * nn + k] * b->array2d[j * nn + k];
                }
            }
        }
        return c;
    }
    
    Matrix* scale(Matrix *a, double s) {
        Matrix *c = new Matrix(a->row, a->col);
#pragma omp parallel for
        for (int i = 0; i < a->row; ++i) {
            for (int j = 0; j < a->col; ++j) {
                c->data[i][j] = a->data[i][j] * s;
            }
        }
        return c;
    }
    Matrix* scale_in_place(Matrix *a, double s) {
#pragma omp parallel for
        for (int i = 0; i < a->row * a->col; ++i)
            a->array2d[i] = a->array2d[i] * s;
        return a;
    }
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
    Matrix* softmax_in_place(Matrix *a)
    {
#pragma omp parallel for
        for (int i = 0; i < a->row; ++i)
        {
            double sum = 0;
            for (int j = 0; j < a->col; ++j) {
                sum += exp(a->data[i][j]);
            }
            for (int j = 0; j < a->col; ++j) {
                a->data[i][j] = exp(a->data[i][j]) / sum;
            }
        }
        return a;
    }

    Matrix* attention(Matrix *q, Matrix *k, Matrix *vt)
    {
        int size, rank, begin, chunk, n;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        chunk = (q->row + size - 1) / size;
        begin = chunk * rank;
        n = std::min(chunk, q->row - begin);
        if (n <= 0)
        {
            n = 0;
        }
        attention::Matrix *qq = getlines(q, n, begin);

        // Matrix *qk_s = matmul_T_scale(q, k, 1.0 / sqrt(k->col));
        Matrix *qk = matmul_T_cache(qq, k);
        Matrix *qk_s = scale(qk, 1.0 / sqrt(k->col));
        Matrix *qk_s_s = softmax(qk_s);
        Matrix *qkv = matmul_T_cache(qk_s_s, vt);
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


    attention::Matrix *vt = transpose(v);
    // int size, begin, chunk, n;
    // MPI_Comm_size(MPI_COMM_WORLD, &size);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // chunk = (q->row + size - 1) / size;
    // begin = chunk * rank;
    // n = chunk < (q->row - begin) ? chunk : (q->row - begin);
    // attention::Matrix *qq = getlines(q, n, begin);
    // attention::Matrix *vt = transpose(v);
    // Start attention.
    auto start = MPI_Wtime();
    auto qkv = attention::attention(q, k, vt);
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