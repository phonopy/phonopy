/* Copyright (C) 2015 Atsushi Togo */
/* All rights reserved. */

/* This file is part of phonopy. */

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

#include <math.h>
#include <stdlib.h>

static void get_derivative_nac(double *ddnac,
			       double *dnac,
			       const int num_patom,
			       const double *lattice,
			       const double *mass,
			       const double *q,
			       const double *born,
			       const double *dielectric,
			       const double *q_direction,
			       const double factor);
static double get_A(const int atom_i,
		    const int cart_i,
		    const double q[3],
		    const double *born);
static double get_C(const double q[3],
		    const double *dielectric);
static double get_dA(const int atom_i,
		     const int cart_i,
		     const int cart_j,
		     const double *born);
static double get_dC(const int cart_i,
		     const int cart_j,
		     const int cart_k,
		     const double q[3],
		     const double *dielectric);

void get_derivative_dynmat_at_q(double *derivative_dynmat,
				const int num_patom, 
				const int num_satom,
				const double *fc,
				const double *q,
				const double *lattice, /* column vector */
				const double *r,
				const int *multi,
				const double *mass,
				const int *s2p_map, 
				const int *p2s_map,
				const double nac_factor,
				const double *born,
				const double *dielectric,
				const double *q_direction)
{
  int i, j, k, l, m, n, adrs, adrsT, is_nac;
  double coef[3], real_coef[3], imag_coef[3];
  double c, s, phase, mass_sqrt, fc_elem, factor, real_phase, imag_phase;
  double ddm_real[3][3][3], ddm_imag[3][3][3];
  double *ddnac, *dnac;

  if (born) {
    is_nac = 1;
    if (q_direction) {
      if (fabs(q_direction[0]) < 1e-5 &&
	  fabs(q_direction[1]) < 1e-5 &&
	  fabs(q_direction[2]) < 1e-5) {
	is_nac = 0;
      }
    } else {
      if (fabs(q[0]) < 1e-5 &&
	  fabs(q[1]) < 1e-5 &&
	  fabs(q[2]) < 1e-5) {
	is_nac = 0;
      }
    }
  } else {
    is_nac = 0;
  }

  if (is_nac) {
    ddnac = (double*) malloc(sizeof(double) * num_patom * num_patom * 27);
    dnac = (double*) malloc(sizeof(double) * num_patom * num_patom * 9);
    factor = nac_factor * num_patom / num_satom;
    get_derivative_nac(ddnac,
		       dnac,
		       num_patom,
		       lattice,
		       mass,
		       q,
		       born,
		       dielectric,
		       q_direction,
		       factor);
  }

  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < num_patom; j++) {
      mass_sqrt = sqrt(mass[i] * mass[j]);

      for (k = 0; k < 3; k++) {
	for (l = 0; l < 3; l++) {
	  for (m = 0; m < 3; m++) {
	    ddm_real[m][k][l] = 0;
	    ddm_imag[m][k][l] = 0;
	  }
	}
      }

      for (k = 0; k < num_satom; k++) { /* Lattice points of right index of fc */
	if (s2p_map[k] != p2s_map[j]) {
	  continue;
	}

	real_phase = 0;
	imag_phase = 0;
	for (l = 0; l < 3; l++) {
	  real_coef[l] = 0;
	  imag_coef[l] = 0;
	}
	for (l = 0; l < multi[k * num_patom + i]; l++) {
	  phase = 0;
	  for (m = 0; m < 3; m++) {
	    phase += q[m] * r[k * num_patom * 81 + i * 81 + l * 3 + m];
	  }
	  s = sin(phase * 2 * M_PI);
	  c = cos(phase * 2 * M_PI);

	  real_phase += c;
	  imag_phase += s;
	  
	  for (m = 0; m < 3; m++) {
	    coef[m] = 0;
	    for (n = 0; n < 3; n++) {
	      coef[m] += 2 * M_PI *
		lattice[m * 3 + n] * r[k * num_patom * 81 + i * 81 + l * 3 + n];
	    }
	  }
	  
	  for (m =0; m < 3; m++) {
	    real_coef[m] -= coef[m] * s;
	    imag_coef[m] += coef[m] * c;
	  }
	}

	real_phase /= multi[k * num_patom + i];
	imag_phase /= multi[k * num_patom + i];
	
	for (l = 0; l < 3; l++) {
	  real_coef[l] /= multi[k * num_patom + i];
	  imag_coef[l] /= multi[k * num_patom + i];
	}

	for (l = 0; l < 3; l++) {
	  for (m = 0; m < 3; m++) {
	    fc_elem = fc[p2s_map[i] * num_satom * 9 + k * 9 + l * 3 + m]
	      / mass_sqrt;
	    if (is_nac) {
	      fc_elem += dnac[i * 9 * num_patom + j * 9 + l * 3 + m];
	    }
	    for (n = 0; n < 3; n++) {
	      ddm_real[n][l][m] += fc_elem * real_coef[n];
	      ddm_imag[n][l][m] += fc_elem * imag_coef[n];
	      if (is_nac) {
		ddm_real[n][l][m] +=
		  ddnac[n * num_patom * num_patom * 9 +
			i * 9 * num_patom + j * 9 + l * 3 + m] * real_phase;
		ddm_imag[n][l][m] +=		
		  ddnac[n * num_patom * num_patom * 9 +
			i * 9 * num_patom + j * 9 + l * 3 + m] * imag_phase;
	      }
	    }
	  }
	}
      }
      
      for (k = 0; k < 3; k++) {
	for (l = 0; l < 3; l++) {
	  for (m = 0; m < 3; m++) {
	    adrs = (k * num_patom * num_patom * 18 +
		    (i * 3 + l) * num_patom * 6 + j * 6 + m * 2);
	    derivative_dynmat[adrs] += ddm_real[k][l][m];
	    derivative_dynmat[adrs + 1] += ddm_imag[k][l][m];
	  }
	}
      }
    }
  }

  /* Symmetrize to be a Hermitian matrix */
  for (i = 0; i < 3; i++) {
    for (j = i; j < num_patom * 3; j++) {
      for (k = 0; k < num_patom * 3; k++) {
	adrs = i * num_patom * num_patom * 18 + j * num_patom * 6 + k * 2;
	adrsT = i * num_patom * num_patom * 18 + k * num_patom * 6 + j * 2;
	derivative_dynmat[adrs] += derivative_dynmat[adrsT];
	derivative_dynmat[adrs] /= 2;
	derivative_dynmat[adrs + 1] -= derivative_dynmat[adrsT+ 1];
	derivative_dynmat[adrs + 1] /= 2;
	derivative_dynmat[adrsT] = derivative_dynmat[adrs];
	derivative_dynmat[adrsT + 1] = -derivative_dynmat[adrs + 1];
      }
    }
  }


  if (is_nac) {
    free(ddnac);
    free(dnac);
  }
}

/* D_nac = a * AB/C */
/* dD_nac = a * D_nac * (A'/A + B'/B - C'/C) */
static void get_derivative_nac(double *ddnac,
			       double *dnac,
			       const int num_patom,
			       const double *lattice,
			       const double *mass,
			       const double *q,
			       const double *born,
			       const double *dielectric,
			       const double *q_direction,
			       const double factor)
{
  int i, j, k, l, m;
  double a, b, c, da, db, dc, volume, mass_sqrt;
  double q_cart[3], rec_lat[9];

  volume =
    lattice[0] * (lattice[4] * lattice[8] - lattice[5] * lattice[7]) +
    lattice[1] * (lattice[5] * lattice[6] - lattice[3] * lattice[8]) +
    lattice[2] * (lattice[3] * lattice[7] - lattice[4] * lattice[6]);

  rec_lat[0] = (lattice[4] * lattice[8] - lattice[5] * lattice[7]) / volume;
  rec_lat[1] = (lattice[5] * lattice[6] - lattice[3] * lattice[8]) / volume;
  rec_lat[2] = (lattice[3] * lattice[7] - lattice[4] * lattice[6]) / volume;
  rec_lat[3] = (lattice[7] * lattice[2] - lattice[8] * lattice[1]) / volume;
  rec_lat[4] = (lattice[8] * lattice[0] - lattice[6] * lattice[2]) / volume;
  rec_lat[5] = (lattice[6] * lattice[1] - lattice[7] * lattice[0]) / volume;
  rec_lat[6] = (lattice[1] * lattice[5] - lattice[2] * lattice[4]) / volume;
  rec_lat[7] = (lattice[2] * lattice[3] - lattice[0] * lattice[5]) / volume;
  rec_lat[8] = (lattice[0] * lattice[4] - lattice[1] * lattice[3]) / volume;
    
  for (i = 0; i < 3; i++) {
    q_cart[i] = 0;
    for (j = 0; j < 3; j++) {
      if (q_direction) {
	q_cart[i] += rec_lat[i * 3 + j] * q_direction[j];
      } else {
	q_cart[i] += rec_lat[i * 3 + j] * q[j];
      }
    }
  }

  c = get_C(q_cart, dielectric);

  for (i = 0; i < num_patom; i++) { /* atom_i */
    for (j = 0; j < num_patom; j++) { /* atom_j */
      mass_sqrt = sqrt(mass[i] * mass[j]);
      for (k = 0; k < 3; k++) { /* derivative direction */
	for (l = 0; l < 3; l++) { /* alpha */
	  a = get_A(i, l, q_cart, born);
	  da = get_dA(i, l, k, born);
	  for (m = 0; m < 3; m++) { /* beta */
	    b = get_A(j, m, q_cart, born);
	    db = get_dA(j, m, k, born);
	    dc = get_dC(l, m, k, q_cart, dielectric);
	    ddnac[k * num_patom * num_patom * 9 +
		  i * 9 * num_patom + j * 9 + l * 3 + m] =
	      (da * b + db * a - a * b * dc / c) / (c * mass_sqrt) * factor;
	    if (k == 0) {
	      dnac[i * 9 * num_patom + j * 9 + l * 3 + m] =
		a * b / (c * mass_sqrt) * factor; 
	    }
	  }
	}
      }
    }
  }
}

static double get_A(const int atom_i,
		    const int cart_i,
		    const double q[3],
		    const double *born)
{
  int i;
  double sum;

  sum = 0;
  for (i = 0; i < 3; i++) {
    sum += q[i] * born[atom_i * 9 + i * 3 + cart_i];
  }
  
  return sum;
}

static double get_C(const double q[3],
		    const double *dielectric)
{
  int i, j;
  double sum;

  sum = 0;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      sum += q[i] * dielectric[i * 3 + j] * q[j];
    }
  }

  return sum;
}

static double get_dA(const int atom_i,
		     const int cart_i,
		     const int cart_j,
		     const double *born)
{
  return born[atom_i * 9 + cart_j * 3 + cart_i];
}

static double get_dC(const int cart_i,
		     const int cart_j,
		     const int cart_k,
		     const double q[3],
		     const double *dielectric)
{
  if (cart_k == 0) {
    return (2 * q[0] * dielectric[0] +
	    q[1] * (dielectric[1] + dielectric[3]) +
	    q[2] * (dielectric[2] + dielectric[6]));
  }
  if (cart_k == 1) {
    return (2 * q[1] * dielectric[4] +
	    q[2] * (dielectric[5] + dielectric[7]) +
	    q[0] * (dielectric[1] + dielectric[3]));
  }
  if (cart_k == 2) {
    return (2 * q[2] * dielectric[8] +
	    q[0] * (dielectric[2] + dielectric[6]) +
	    q[1] * (dielectric[5] + dielectric[7]));
  }
  return 0;
}
