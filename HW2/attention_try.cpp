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

    Matrix* getlines(Matrix* a, int n, int pos)
    {
        Matrix *c = new Matrix(n, a->col);
        memcpy(c->array2d, &(a->array2d[pos * a->col]), n * a->col * sizeof(double));
        return c;
    }

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

        for (int j = 0; j < mm - 3; j += 4)
        {
            for (int i = 0; i < a->row; ++i)
            {
                double cc[4] = {0};
                for (int k = 0; k < nn; ++k)
                {
                    double aik = a->data[i][k];
                    cc[0] += aik * b->data[j][k];
                    cc[1] += aik * b->data[j + 1][k];
                    cc[2] += aik * b->data[j + 2][k];
                    cc[3] += aik * b->data[j + 3][k];
                }
                memcpy(&(c->data[i][j]), cc, 4 * sizeof(double));
            }
        }
        // Not devided by 4
        for (int j = std::max(mm - 3, 0); j < mm; ++j)
        {
            for (int i = 0; i < a->row; ++i)
            {
                c->data[i][j] = 0;
                for (int k = 0; k < a->col; ++k)
                    c->data[i][j] += a->data[i][k] * b->data[j][k];
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

        for (int j = 0; j < mm; j ++)
        {
            for (int i = 0; i < a->row; ++i)
            {
                double cc = 0;
                for (int k = 0; k < nn; ++k)
                {
                    cc += a->data[i][k] * b->data[j][k];
                }
                c->data[i][j] = cc;
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

    Matrix* attention(Matrix *q, Matrix *k, Matrix *vt)
    {

        int size, rank, begin, chunk, n;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        chunk = (q->row + size - 1) / size;
        begin = chunk * rank;
        n = std::min(chunk, q->row - begin);
        if (n <= 0)
            n = 0;
        attention::Matrix *qq = getlines(q, n, begin);

        Matrix *qk = matmul_T_cache(qq, k);
        Matrix *qk_s = scale(qk, 1.0 / sqrt(k->col));
        Matrix *qk_s_s = softmax(qk_s);
        // attention::Matrix *vt = transpose(v);
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

    // int size, begin, chunk, n;
    // MPI_Comm_size(MPI_COMM_WORLD, &size);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // chunk = (q->row + size - 1) / size;
    // begin = chunk * rank;
    // n = std::min(chunk, q->row - begin);
    // if (n <= 0)
    //     n = 0;
    // attention::Matrix *qq = getlines(q, n, begin);
    attention::Matrix *vt = transpose(v);

    // Start attention.
    auto start = MPI_Wtime();
    auto qkv = attention::attention(q, k, vt);
    auto end = MPI_Wtime();

    // Reduce the answer
    double qkv_ans = reduce_the_sum(qkv);
    auto ti = end - start;
    MPI_Allreduce(MPI_IN_PLACE, &ti, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Your answer: %.10lf\n", qkv_ans);

        // Check the answer.
        bool correct = check(qkv_ans, ans);

        // Output the result.
        if (correct) {
            // printf("Correct! Time: %.10lf\n", end - start);
            printf("Correct! Time: %.10lf\n", ti);
        } else {
            printf("Wrong!\n");
        }
    }
    MPI_Finalize();

    return 0;
}