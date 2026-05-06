#include <assert.h>
#include <stdint.h>
/* #include <stdio.h> */

#ifndef SNF3x3CONST
#define SNF3x3CONST
#endif

static void initialize_PQ(int64_t P[3][3], int64_t Q[3][3]);
static int first(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3]);
static void first_one_loop(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3]);
static void first_column(int64_t A[3][3], int64_t P[3][3]);
static void zero_first_column(int64_t L[3][3], const int j,
                              SNF3x3CONST int64_t A[3][3]);
static int search_first_pivot(SNF3x3CONST int64_t A[3][3]);
static void first_finalize(int64_t L[3][3], SNF3x3CONST int64_t A[3][3]);
static int second(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3]);
static void second_one_loop(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3]);
static void second_column(int64_t A[3][3], int64_t P[3][3]);
static void zero_second_column(int64_t L[3][3], SNF3x3CONST int64_t A[3][3]);
static void second_finalize(int64_t L[3][3], SNF3x3CONST int64_t A[3][3]);
static void finalize(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3]);
static void finalize_sort(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3]);
static void finalize_disturb(int64_t A[3][3], int64_t Q[3][3], const int i,
                             const int j);
static void disturb_rows(int64_t L[3][3], const int i, const int j);
static void swap_diag_elems(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3],
                            const int i, const int j);
static void make_diagA_positive(int64_t A[3][3], int64_t P[3][3]);
static void flip_PQ(int64_t P[3][3], int64_t Q[3][3]);
static void swap_rows(int64_t L[3][3], const int i, const int j);
static void set_zero(int64_t L[3][3], const int i, const int j, const int64_t a,
                     const int64_t b, const int64_t r, const int64_t s,
                     const int64_t t);
static void extended_gcd(int64_t retvals[3], const int64_t a, const int64_t b);
static void extended_gcd_step(int64_t vals[6]);
static void flip_sign_row(int64_t L[3][3], const int i);
static void transpose(int64_t m[3][3]);
static void matmul(int64_t m[3][3], SNF3x3CONST int64_t a[3][3],
                   SNF3x3CONST int64_t b[3][3]);
static int64_t det(SNF3x3CONST int64_t m[3][3]);

/* static void test_set_A(int64_t A[3][3]);
 * static void test_show_A(SNF3x3CONST int64_t A[3][3]);
 * static void test_extended_gcd(void);
 * static void test_transpose(void);
 * static void test_swap_rows(void);
 * static void test_zero_first_column(void);
 * static void test_first_column(void);
 * static void test_first_one_loop(void);
 * static void test_first(void);
 * static void test_second_column(void);
 * static void test_second_one_loop(void);
 * static void test_second(void); */

int snf3x3(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3]) {
    int i;
    initialize_PQ(P, Q);

    for (i = 0; i < 100; i++) {
        if (first(A, P, Q)) {
            if (second(A, P, Q)) {
                finalize(A, P, Q);
                transpose(Q);
                goto succeeded;
            }
        }
    }
    return 0;

succeeded:
    return 1;
}

static void initialize_PQ(int64_t P[3][3], int64_t Q[3][3]) {
    int i, j;

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            if (i == j) {
                P[i][j] = 1;
                Q[i][j] = 1;
            } else {
                P[i][j] = 0;
                Q[i][j] = 0;
            }
        }
    }
}

static int first(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3]) {
    int64_t L[3][3];

    first_one_loop(A, P, Q);

    /* rows and columns are all zero except for the pivot */
    if ((A[1][0] == 0) && (A[2][0] == 0)) {
        return 1;
    }

    /* columns of the pivot are assumed zero because of first_one_loop. */
    /* rows of the pivot are non-zero, but divisible by the pivot. */
    /* first_finalize makes the rows be zero. */
    if ((A[1][0] % A[0][0] == 0) && (A[2][0] % A[0][0] == 0)) {
        first_finalize(L, A);
        matmul(A, L, A);
        matmul(P, L, P);
        return 1;
    }
    return 0;
}

static void first_one_loop(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3]) {
    first_column(A, P);
    transpose(A);
    first_column(A, Q);
    transpose(A);
}

static void first_column(int64_t A[3][3], int64_t P[3][3]) {
    int i;
    int64_t L[3][3];

    i = search_first_pivot(A);
    if (i > 0) {
        swap_rows(L, 0, i);
        matmul(A, L, A);
        matmul(P, L, P);
    }
    if (i < 0) {
        goto err;
    }

    if (A[1][0] != 0) {
        zero_first_column(L, 1, A);
        matmul(A, L, A);
        matmul(P, L, P);
    }
    if (A[2][0] != 0) {
        zero_first_column(L, 2, A);
        matmul(A, L, A);
        matmul(P, L, P);
    }

err:;
}

static void zero_first_column(int64_t L[3][3], const int j,
                              SNF3x3CONST int64_t A[3][3]) {
    int64_t vals[3];

    extended_gcd(vals, A[0][0], A[j][0]);
    set_zero(L, 0, j, A[0][0], A[j][0], vals[0], vals[1], vals[2]);
}

static int search_first_pivot(SNF3x3CONST int64_t A[3][3]) {
    int i;

    for (i = 0; i < 3; i++) {
        if (A[i][0] != 0) {
            return i;
        }
    }
    return -1;
}

static void first_finalize(int64_t L[3][3], SNF3x3CONST int64_t A[3][3]) {
    L[0][0] = 1;
    L[0][1] = 0;
    L[0][2] = 0;
    L[1][0] = -A[1][0] / A[0][0];
    L[1][1] = 1;
    L[1][2] = 0;
    L[2][0] = -A[2][0] / A[0][0];
    L[2][1] = 0;
    L[2][2] = 1;
}

static int second(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3]) {
    int64_t L[3][3];

    second_one_loop(A, P, Q);

    if (A[2][1] == 0) {
        return 1;
    }

    if (A[2][1] % A[1][1] == 0) {
        second_finalize(L, A);
        matmul(A, L, A);
        matmul(P, L, P);
        return 1;
    }

    return 0;
}

static void second_one_loop(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3]) {
    second_column(A, P);
    transpose(A);
    second_column(A, Q);
    transpose(A);
}

static void second_column(int64_t A[3][3], int64_t P[3][3]) {
    int64_t L[3][3];

    if ((A[1][1] == 0) && (A[2][1] != 0)) {
        swap_rows(L, 1, 2);
        matmul(A, L, A);
        matmul(P, L, P);
    }

    if (A[2][1] != 0) {
        zero_second_column(L, A);
        matmul(A, L, A);
        matmul(P, L, P);
    }
}

static void zero_second_column(int64_t L[3][3], SNF3x3CONST int64_t A[3][3]) {
    int64_t vals[3];

    extended_gcd(vals, A[1][1], A[2][1]);
    set_zero(L, 1, 2, A[1][1], A[2][1], vals[0], vals[1], vals[2]);
}

static void second_finalize(int64_t L[3][3], SNF3x3CONST int64_t A[3][3]) {
    L[0][0] = 1;
    L[0][1] = 0;
    L[0][2] = 0;
    L[1][0] = 0;
    L[1][1] = 1;
    L[1][2] = 0;
    L[2][0] = 0;
    L[2][1] = -A[2][1] / A[1][1];
    L[2][2] = 1;
}

static void finalize(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3]) {
    make_diagA_positive(A, P);

    finalize_sort(A, P, Q);
    finalize_disturb(A, Q, 0, 1);
    first(A, P, Q);
    finalize_sort(A, P, Q);
    finalize_disturb(A, Q, 1, 2);
    second(A, P, Q);
    flip_PQ(P, Q);
}

static void finalize_sort(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3]) {
    if (A[0][0] > A[1][1]) {
        swap_diag_elems(A, P, Q, 0, 1);
    }
    if (A[1][1] > A[2][2]) {
        swap_diag_elems(A, P, Q, 1, 2);
    }
    if (A[0][0] > A[1][1]) {
        swap_diag_elems(A, P, Q, 0, 1);
    }
}

static void finalize_disturb(int64_t A[3][3], int64_t Q[3][3], const int i,
                             const int j) {
    int64_t L[3][3];

    if (A[j][j] % A[i][i] != 0) {
        transpose(A);
        disturb_rows(L, i, j);
        matmul(A, L, A);
        matmul(Q, L, Q);
        transpose(A);
    }
}

static void disturb_rows(int64_t L[3][3], const int i, const int j) {
    L[0][0] = 1;
    L[0][1] = 0;
    L[0][2] = 0;
    L[1][0] = 0;
    L[1][1] = 1;
    L[1][2] = 0;
    L[2][0] = 0;
    L[2][1] = 0;
    L[2][2] = 1;
    L[i][i] = 1;
    L[i][j] = 1;
    L[j][i] = 0;
    L[j][j] = 1;
}

static void swap_diag_elems(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3],
                            const int i, const int j) {
    int64_t L[3][3];

    swap_rows(L, i, j);
    matmul(A, L, A);
    matmul(P, L, P);
    transpose(A);
    swap_rows(L, i, j);
    matmul(A, L, A);
    matmul(Q, L, Q);
    transpose(A);
}

static void make_diagA_positive(int64_t A[3][3], int64_t P[3][3]) {
    int i;
    int64_t L[3][3];

    for (i = 0; i < 3; i++) {
        if (A[i][i] < 0) {
            flip_sign_row(L, i);
            matmul(A, L, A);
            matmul(P, L, P);
        }
    }
}

static void flip_PQ(int64_t P[3][3], int64_t Q[3][3]) {
    int i, j;

    if (det(P) < 0) {
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                P[i][j] *= -1;
                Q[i][j] *= -1;
            }
        }
    }
}

static void swap_rows(int64_t L[3][3], const int r1, const int r2) {
    L[0][0] = 1;
    L[0][1] = 0;
    L[0][2] = 0;
    L[1][0] = 0;
    L[1][1] = 1;
    L[1][2] = 0;
    L[2][0] = 0;
    L[2][1] = 0;
    L[2][2] = 1;
    L[r1][r1] = 0;
    L[r2][r2] = 0;
    L[r1][r2] = 1;
    L[r2][r1] = 1;
}

static void set_zero(int64_t L[3][3], const int i, const int j, const int64_t a,
                     const int64_t b, const int64_t r, const int64_t s,
                     const int64_t t) {
    L[0][0] = 1;
    L[0][1] = 0;
    L[0][2] = 0;
    L[1][0] = 0;
    L[1][1] = 1;
    L[1][2] = 0;
    L[2][0] = 0;
    L[2][1] = 0;
    L[2][2] = 1;
    L[i][i] = s;
    L[i][j] = t;
    L[j][i] = -b / r;
    L[j][j] = a / r;
}

/**
 * Extended Euclidean algorithm
 * See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
 */
static void extended_gcd(int64_t retvals[3], const int64_t a, const int64_t b) {
    int i;
    int64_t vals[6];

    vals[0] = a; /* r0 */
    vals[1] = b; /* r1 */
    vals[2] = 1; /* s0 */
    vals[3] = 0; /* s1 */
    vals[4] = 0; /* t0 */
    vals[5] = 1; /* t1 */

    for (i = 0; i < 1000; i++) {
        extended_gcd_step(vals);
        if (vals[1] == 0) {
            break;
        }
    }

    retvals[0] = vals[0];
    retvals[1] = vals[2];
    retvals[2] = vals[4];

    assert(vals[0] == a * vals[2] + b * vals[4]);
}

static void extended_gcd_step(int64_t vals[6]) {
    int64_t q, r2, s2, t2;

    q = vals[0] / vals[1];
    r2 = vals[0] % vals[1];
    if (r2 < 0) {
        if (vals[1] > 0) {
            r2 += vals[1];
            q -= 1;
        }
        if (vals[1] < 0) {
            r2 -= vals[1];
            q += 1;
        }
    }
    s2 = vals[2] - q * vals[3];
    t2 = vals[4] - q * vals[5];
    vals[0] = vals[1];
    vals[1] = r2;
    vals[2] = vals[3];
    vals[3] = s2;
    vals[4] = vals[5];
    vals[5] = t2;
}

static void flip_sign_row(int64_t L[3][3], const int i) {
    L[0][0] = 1;
    L[0][1] = 0;
    L[0][2] = 0;
    L[1][0] = 0;
    L[1][1] = 1;
    L[1][2] = 0;
    L[2][0] = 0;
    L[2][1] = 0;
    L[2][2] = 1;
    L[i][i] = -1;
}

/**
 * Matrix operation utils
 */
static void transpose(int64_t m[3][3]) {
    int64_t tmp;
    int i, j;

    for (i = 0; i < 3; i++) {
        for (j = i; j < 3; j++) {
            tmp = m[i][j];
            m[i][j] = m[j][i];
            m[j][i] = tmp;
        }
    }
}

static void matmul(int64_t m[3][3], SNF3x3CONST int64_t a[3][3],
                   SNF3x3CONST int64_t b[3][3]) {
    int i, j;
    int64_t c[3][3];

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            m[i][j] = c[i][j];
        }
    }
}

static int64_t det(SNF3x3CONST int64_t m[3][3]) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) +
           m[0][1] * (m[1][2] * m[2][0] - m[1][0] * m[2][2]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

/* int main()
 * {
 *   test_extended_gcd();
 *   test_transpose();
 *   test_swap_rows();
 *   test_zero_first_column();
 *   test_first_column();
 *   test_first_one_loop();
 *   test_first();
 *   test_second_column();
 *   test_second_one_loop();
 *   test_second();
 * }
 *
 * static void test_set_A(int64_t A[3][3])
 * {
 *   int i, j;
 *
 *   for (i = 0; i < 3; i++) {
 *     for (j = 0; j < 3; j++) {
 *       A[i][j] = i * 3 + j;
 *     }
 *   }
 *   A[0][0] = 1;  /\* to avoid det(A) = 0 *\/
 * }
 *
 * static void test_show_A(SNF3x3CONST int64_t A[3][3])
 * {
 *   int i, j;
 *
 *   for (i = 0; i < 3; i++) {
 *     for (j = 0; j < 3; j++) {
 *       printf("%d ", A[i][j]);
 *     }
 *     printf("\n");
 *   }
 * }
 *
 * static void test_extended_gcd(void)
 * {
 *   int vals[3];
 *
 *   printf("Test extended_gcd\n");
 *   extended_gcd(vals, 4 , -3);
 *   printf("%d %d %d\n", vals[0], vals[1], vals[2]);
 * }
 *
 * static void test_transpose(void)
 * {
 *   int i, j;
 *   int64_t A[3][3];
 *
 *   printf("Test transpose\n");
 *
 *   test_set_A(A);
 *   test_show_A(A);
 *   transpose(A);
 *   test_show_A(A);
 * }
 *
 * static void test_swap_rows(void)
 * {
 *   int i, j;
 *   int64_t A[3][3], L[3][3];
 *
 *   printf("Test swap_rows 1 <-> 2\n");
 *
 *   test_set_A(A);
 *   test_show_A(A);
 *   swap_rows(L, 0, 1);
 *   matmul(A, L, A);
 *   test_show_A(A);
 * }
 *
 * static void test_zero_first_column(void)
 * {
 *   int i, j;
 *   int64_t A[3][3], L[3][3];
 *
 *   printf("Test zero_first_column\n");
 *
 *   test_set_A(A);
 *   test_show_A(A);
 *   zero_first_column(L, 1, A);
 *   matmul(A, L, A);
 *   test_show_A(A);
 *   zero_first_column(L, 2, A);
 *   matmul(A, L, A);
 *   test_show_A(A);
 * }
 *
 * static void test_first_column(void)
 * {
 *   int i, j;
 *   int64_t A[3][3], P[3][3];
 *
 *   printf("Test first_column\n");
 *
 *   test_set_A(A);
 *   test_show_A(A);
 *   first_column(A, P);
 *   test_show_A(A);
 *   transpose(A);
 *   first_column(A, P);
 *   transpose(A);
 *   test_show_A(A);
 * }
 *
 * static void test_first_one_loop(void)
 * {
 *   int i, j;
 *   int64_t A[3][3], P[3][3], Q[3][3];
 *
 *   printf("Test first_one_loop\n");
 *
 *   test_set_A(A);
 *   test_show_A(A);
 *   first_one_loop(A, P, Q);
 *   test_show_A(A);
 * }
 *
 * static void test_first(void)
 * {
 *   int i, j;
 *   int64_t A[3][3], P[3][3], Q[3][3];
 *
 *   printf("Test first\n");
 *
 *   test_set_A(A);
 *   test_show_A(A);
 *   first(A, P, Q);
 *   test_show_A(A);
 * }
 *
 * static void test_second_column(void)
 * {
 *   int i, j;
 *   int64_t A[3][3], P[3][3], Q[3][3];
 *
 *   printf("Test second_column\n");
 *
 *   test_set_A(A);
 *   test_show_A(A);
 *   first(A, P, Q);
 *   test_show_A(A);
 *   second_column(A, P);
 *   test_show_A(A);
 * }
 *
 * static void test_second_one_loop(void)
 * {
 *   int i, j;
 *   int64_t A[3][3], P[3][3], Q[3][3];
 *
 *   printf("Test second_one_loop\n");
 *
 *   test_set_A(A);
 *   test_show_A(A);
 *   first(A, P, Q);
 *   test_show_A(A);
 *   second_one_loop(A, P, Q);
 *   test_show_A(A);
 * }
 *
 * static void test_second(void)
 * {
 *   int i, j;
 *   int64_t A[3][3], P[3][3], Q[3][3];
 *
 *   printf("Test second\n");
 *
 *   test_set_A(A);
 *   test_show_A(A);
 *   first(A, P, Q);
 *   test_show_A(A);
 *   second(A, P, Q);
 *   test_show_A(A);
 * } */
