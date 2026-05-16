#ifndef __snf3x3_H__
#define __snf3x3_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define SNF3X3_MAJOR_VERSION 0
#define SNF3X3_MINOR_VERSION 1
#define SNF3X3_MICRO_VERSION 0

int snf3x3(int64_t A[3][3], int64_t P[3][3], int64_t Q[3][3]);

#ifdef __cplusplus
}
#endif
#endif
