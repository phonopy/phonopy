#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdint.h>

#include "recgrid.h"

namespace nb = nanobind;

int64_t py_get_grid_index_from_address(nb::ndarray<> py_address,
                                       nb::ndarray<> py_D_diag) {
    int64_t *address;
    int64_t *D_diag;
    int64_t gp;

    address = (int64_t *)py_address.data();
    D_diag = (int64_t *)py_D_diag.data();

    gp = recgrid_get_grid_index_from_address(address, D_diag);

    return gp;
}

int64_t py_get_ir_grid_map(nb::ndarray<> py_grid_mapping_table,
                           nb::ndarray<> py_D_diag, nb::ndarray<> py_is_shift,
                           nb::ndarray<> py_rotations) {
    int64_t *D_diag;
    int64_t *is_shift;
    int64_t(*rot)[3][3];
    int64_t num_rot;

    int64_t *grid_mapping_table;
    int64_t num_ir;

    D_diag = (int64_t *)py_D_diag.data();
    is_shift = (int64_t *)py_is_shift.data();
    rot = (int64_t(*)[3][3])py_rotations.data();
    num_rot = (int64_t)py_rotations.shape(0);
    grid_mapping_table = (int64_t *)py_grid_mapping_table.data();

    num_ir = recgrid_get_ir_grid_map(grid_mapping_table, rot, num_rot, D_diag,
                                     is_shift);
    return num_ir;
}

void py_get_gr_grid_addresses(nb::ndarray<> py_gr_grid_addresses,
                              nb::ndarray<> py_D_diag) {
    int64_t(*gr_grid_addresses)[3];
    int64_t *D_diag;

    gr_grid_addresses = (int64_t(*)[3])py_gr_grid_addresses.data();
    D_diag = (int64_t *)py_D_diag.data();

    recgrid_get_all_grid_addresses(gr_grid_addresses, D_diag);
}

int64_t py_get_reciprocal_rotations(nb::ndarray<> py_rec_rotations,
                                    nb::ndarray<> py_rotations,
                                    int64_t is_time_reversal) {
    int64_t(*rec_rotations)[3][3];
    int64_t(*rotations)[3][3];
    int64_t num_rot, num_rec_rot;

    rec_rotations = (int64_t(*)[3][3])py_rec_rotations.data();
    rotations = (int64_t(*)[3][3])py_rotations.data();
    num_rot = (int64_t)py_rotations.shape(0);

    num_rec_rot = recgrid_get_reciprocal_point_group(
        rec_rotations, rotations, num_rot, is_time_reversal, 1);

    return num_rec_rot;
}

bool py_transform_rotations(nb::ndarray<> py_transformed_rotations,
                            nb::ndarray<> py_rotations, nb::ndarray<> py_D_diag,
                            nb::ndarray<> py_Q) {
    int64_t(*transformed_rotations)[3][3];
    int64_t(*rotations)[3][3];
    int64_t *D_diag;
    int64_t(*Q)[3];
    int64_t num_rot, succeeded;

    transformed_rotations = (int64_t(*)[3][3])py_transformed_rotations.data();
    rotations = (int64_t(*)[3][3])py_rotations.data();
    D_diag = (int64_t *)py_D_diag.data();
    Q = (int64_t(*)[3])py_Q.data();
    num_rot = (int64_t)py_transformed_rotations.shape(0);

    succeeded = recgrid_transform_rotations(transformed_rotations, rotations,
                                            num_rot, D_diag, Q);
    if (succeeded) {
        return true;
    } else {
        return false;
    }
}

bool py_get_snf3x3(nb::ndarray<> py_D_diag, nb::ndarray<> py_P,
                   nb::ndarray<> py_Q, nb::ndarray<> py_A) {
    int64_t *D_diag;
    int64_t(*P)[3];
    int64_t(*Q)[3];
    int64_t(*A)[3];
    int64_t succeeded;

    D_diag = (int64_t *)py_D_diag.data();
    P = (int64_t(*)[3])py_P.data();
    Q = (int64_t(*)[3])py_Q.data();
    A = (int64_t(*)[3])py_A.data();

    succeeded = recgrid_get_snf3x3(D_diag, P, Q, A);
    if (succeeded) {
        return true;
    } else {
        return false;
    }
}

int64_t py_get_bz_grid_addresses(
    nb::ndarray<> py_bz_grid_addresses, nb::ndarray<> py_bz_map,
    nb::ndarray<> py_bzg2grg, nb::ndarray<> py_D_diag, nb::ndarray<> py_Q,
    nb::ndarray<> py_PS, nb::ndarray<> py_reciprocal_lattice, int64_t type) {
    int64_t(*bz_grid_addresses)[3];
    int64_t *bz_map;
    int64_t *bzg2grg;
    int64_t *D_diag;
    int64_t(*Q)[3];
    int64_t *PS;
    double(*reciprocal_lattice)[3];
    int64_t num_total_gp;

    bz_grid_addresses = (int64_t(*)[3])py_bz_grid_addresses.data();
    bz_map = (int64_t *)py_bz_map.data();
    bzg2grg = (int64_t *)py_bzg2grg.data();
    D_diag = (int64_t *)py_D_diag.data();
    Q = (int64_t(*)[3])py_Q.data();
    PS = (int64_t *)py_PS.data();
    reciprocal_lattice = (double(*)[3])py_reciprocal_lattice.data();

    num_total_gp =
        recgrid_get_bz_grid_addresses(bz_grid_addresses, bz_map, bzg2grg,
                                      D_diag, Q, PS, reciprocal_lattice, type);

    return num_total_gp;
}

int64_t py_rotate_bz_grid_addresses(int64_t bz_grid_index,
                                    nb::ndarray<> py_rotation,
                                    nb::ndarray<> py_bz_grid_addresses,
                                    nb::ndarray<> py_bz_map,
                                    nb::ndarray<> py_D_diag,
                                    nb::ndarray<> py_PS, int64_t type) {
    int64_t(*bz_grid_addresses)[3];
    int64_t(*rotation)[3];
    int64_t *bz_map;
    int64_t *D_diag;
    int64_t *PS;
    int64_t ret_bz_gp;

    bz_grid_addresses = (int64_t(*)[3])py_bz_grid_addresses.data();
    rotation = (int64_t(*)[3])py_rotation.data();
    bz_map = (int64_t *)py_bz_map.data();
    D_diag = (int64_t *)py_D_diag.data();
    PS = (int64_t *)py_PS.data();

    ret_bz_gp = recgrid_rotate_bz_grid_index(
        bz_grid_index, rotation, bz_grid_addresses, bz_map, D_diag, PS, type);

    return ret_bz_gp;
}

NB_MODULE(_recgrid, m) {
    m.def("grid_index_from_address", &py_get_grid_index_from_address);
    m.def("ir_grid_map", &py_get_ir_grid_map);
    m.def("gr_grid_addresses", &py_get_gr_grid_addresses);
    m.def("reciprocal_rotations", &py_get_reciprocal_rotations);
    m.def("transform_rotations", &py_transform_rotations);
    m.def("snf3x3", &py_get_snf3x3);
    m.def("bz_grid_addresses", &py_get_bz_grid_addresses);
    m.def("rotate_bz_grid_index", &py_rotate_bz_grid_addresses);
}
