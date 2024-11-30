import phonopy
from phonopy.structure.cells import isclose

ph_iter = phonopy.load("phonopy_sscha_fc_JPCM2022.yaml.xz", log_level=1)
ph_ha = phonopy.load("phonopy_fc222_JPCM2022.yaml.xz", log_level=1)
ph_ha_444 = phonopy.load("phonopy_fc444_JPCM2022.yaml.xz", log_level=1)

if not isclose(ph_iter.unitcell, ph_ha.unitcell):
    raise RuntimeError("Unitcells of ph_iter and ph_ha are inconsistent.")
if not isclose(ph_iter.unitcell, ph_ha_444.unitcell):
    raise RuntimeError("Unitcells of ph_iter and ph_ha_444 are inconsistent.")

dim = [4, 4, 4]
ipfc_iter_444 = ph_iter.ph2ph(dim).force_constants
ipfc_ha_444 = ph_ha.ph2ph(dim).force_constants
delta_ipfc_444 = ipfc_iter_444 - ipfc_ha_444
ph_ha_444_force_constants = ph_ha_444.force_constants + delta_ipfc_444

ph_ipfc_444 = ph_ha_444.copy()
ph_ipfc_444.force_constants = ph_ha_444_force_constants
ph_ipfc_444.nac_params = ph_iter.nac_params

ph_ipfc_444.save(
    filename="phonopy_ipfc444_JPCM2022.yaml",
    settings={"force_sets": False, "displacements": False, "force_constants": True},
    compression=True,
)
