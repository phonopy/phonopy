/* Copyright (C) 2016 Atsushi Togo */
/* All rights reserved. */

/* This file is part of spglib. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */

/* * Redistributions of source code must retain the above copyright */
/*   notice, this list of conditions and the following disclaimer. */

/* * Redistributions in binary form must reproduce the above copyright */
/*   notice, this list of conditions and the following disclaimer in */
/*   the documentation and/or other materials provided with the */
/*   distribution. */

/* * Neither the name of the phonopy project nor the names of its */
/*   contributors may be used to endorse or promote products derived */
/*   from this software without specific prior written permission. */

/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
/* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT */
/* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS */
/* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE */
/* COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, */
/* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, */
/* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; */
/* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER */
/* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT */
/* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN */
/* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE */
/* POSSIBILITY OF SUCH DAMAGE. */

#include <string.h>
#include <stdio.h>
#include "arithmetic.h"

#include "debug.h"

static int arithmetic_crystal_classes[231] = {
  0,
  1,  2,  3,  3,  4,  5,  5,  6,  6,  7,
  7,  8,  7,  7,  8,  9,  9,  9,  9, 10,
  10, 11, 12, 12, 13, 13, 13, 13, 13, 13,
  13, 13, 13, 13, 14, 14, 14, 15, 15, 15,
  15, 16, 16, 17, 17, 17, 18, 18, 18, 18,
  18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
  18, 18, 19, 19, 19, 19, 19, 19, 20, 20,
  21, 21, 21, 21, 22, 22, 22, 22, 23, 23,
  24, 25, 26, 26, 26, 26, 27, 27, 28, 28,
  28, 28, 28, 28, 28, 28, 29, 29, 30, 30,
  30, 30, 30, 30, 30, 30, 31, 31, 31, 31,
  32, 32, 32, 32, 33, 33, 33, 33, 34, 34,
  35, 35, 36, 36, 36, 36, 36, 36, 36, 36,
  36, 36, 36, 36, 36, 36, 36, 36, 37, 37,
  37, 37, 38, 38, 38, 39, 40, 41, 42, 43,
  42, 43, 42, 43, 44, 45, 46, 45, 46, 47,
  47, 48, 48, 49, 49, 50, 50, 51, 51, 51,
  51, 51, 51, 52, 53, 53, 54, 54, 54, 54,
  54, 54, 55, 55, 55, 55, 56, 56, 57, 57,
  58, 58, 58, 58, 59, 60, 61, 59, 61, 62,
  62, 63, 63, 64, 62, 64, 65, 65, 66, 66,
  67, 65, 65, 67, 68, 69, 70, 68, 69, 70,
  71, 71, 71, 71, 72, 72, 72, 72, 73, 73};
