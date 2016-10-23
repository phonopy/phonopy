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

static const char arithmetic_crystal_class_symbols[74][7] = {
  "      ", /*  0 */
  "1P    ", /*  1 */
  "-1P   ", /*  2 */
  "2P    ", /*  3 */
  "2C    ", /*  4 */
  "mP    ", /*  5 */
  "mC    ", /*  6 */
  "2/mP  ", /*  7 */
  "2/mC  ", /*  8 */
  "222P  ", /*  9 */
  "222C  ", /* 10 */
  "222F  ", /* 11 */
  "222I  ", /* 12 */
  "mm2P  ", /* 13 */
  "mm2C  ", /* 14 */
  "2mmC  ", /* 15 */
  "mm2F  ", /* 16 */
  "mm2I  ", /* 17 */
  "mmmP  ", /* 18 */
  "mmmC  ", /* 19 */
  "mmmF  ", /* 20 */
  "mmmI  ", /* 21 */
  "4P    ", /* 22 */
  "4I    ", /* 23 */
  "-4P   ", /* 24 */
  "-4I   ", /* 25 */
  "4/mP  ", /* 26 */
  "4/mI  ", /* 27 */
  "422P  ", /* 28 */
  "422I  ", /* 29 */
  "4mmP  ", /* 30 */
  "4mmI  ", /* 31 */
  "-42mP ", /* 32 */
  "-4m2P ", /* 33 */
  "-4m2I ", /* 34 */
  "-42mI ", /* 35 */
  "4/mmmP", /* 36 */
  "4/mmmI", /* 37 */
  "3P    ", /* 38 */
  "3R    ", /* 39 */
  "-3P   ", /* 40 */
  "-3R   ", /* 41 */
  "312P  ", /* 42 */
  "321P  ", /* 43 */
  "32R   ", /* 44 */
  "3m1P  ", /* 45 */
  "31mP  ", /* 46 */
  "3mR   ", /* 47 */
  "-31mP ", /* 48 */
  "-3m1P ", /* 49 */
  "-3mR  ", /* 50 */
  "6P    ", /* 51 */
  "-6P   ", /* 52 */
  "6/mP  ", /* 53 */
  "622P  ", /* 54 */
  "6mmP  ", /* 55 */
  "-62mP ", /* 56 */
  "-6m2P ", /* 57 */
  "6/mmm ", /* 58 */
  "23P   ", /* 59 */
  "23F   ", /* 60 */
  "23I   ", /* 61 */
  "m-3P  ", /* 62 */
  "m-3F  ", /* 63 */
  "m-3I  ", /* 64 */
  "432P  ", /* 65 */
  "432F  ", /* 66 */
  "432I  ", /* 67 */
  "-43mP ", /* 68 */
  "-43mF ", /* 69 */
  "-43mI ", /* 70 */
  "m-3mP ", /* 71 */
  "m-3mF ", /* 72 */
  "m-3mI "  /* 73 */
};

int arth_get_symbol(char symbol[7], const int spgroup_number)
{
  int i, arth_number;

  if (spgroup_number < 1 || spgroup_number > 230) {
    return 0;
  }

  arth_number = arithmetic_crystal_classes[spgroup_number];
  strcpy(symbol, arithmetic_crystal_class_symbols[arth_number]);
  for (i = 0; i < 6; i++) {
    if (symbol[i] == ' ') {symbol[i] = '\0';}
  }

  return arth_number;
}
