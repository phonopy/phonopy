"""Example of QHA calculation by Al."""

import numpy as np
import yaml
from yaml import CLoader as Loader

from phonopy import PhonopyQHA

volumes = []
energies = []
for line in open("e-v.dat"):
    v, e = line.split()
    volumes.append(float(v))
    energies.append(float(e))

entropy = []
cv = []
fe = []
for index in range(-5, 6):
    filename = "thermal_properties.yaml-%d" % index
    print("Reading %s" % filename)
    thermal_properties = yaml.load(open(filename), Loader=Loader)["thermal_properties"]
    temperatures = [v["temperature"] for v in thermal_properties]
    cv.append([v["heat_capacity"] for v in thermal_properties])
    entropy.append([v["entropy"] for v in thermal_properties])
    fe.append([v["free_energy"] for v in thermal_properties])

qha = PhonopyQHA(
    volumes,
    energies,
    temperatures=temperatures,
    free_energy=np.transpose(fe),
    cv=np.transpose(cv),
    entropy=np.transpose(entropy),
    t_max=400,
    verbose=True,
)

# qha.plot_helmholtz_volume().show()
# qha.plot_volume_temperature().show()
qha.plot_thermal_expansion().show()
# plot = qha.plot_volume_expansion()
# if plot:
#     plot.show()
# qha.plot_gibbs_temperature().show()
# qha.plot_bulk_modulus_temperature().show()
# qha.plot_heat_capacity_P_numerical().show()
# qha.plot_heat_capacity_P_polyfit().show()
# qha.plot_gruneisen_temperature().show()
