import numpy as np
import math


#  calculates high-energy process lengths important for RREA physics
class ProcessLength():
    relative_air_concentration = 1
    electric_field = 200  # kV per m
    speed_of_light = 3e8  # m per s
    distance_between_cells = 100  # m
    electric_field_between_cells = 0  # kV per m

    # a function which interpolates the standard atmosphere model to find the air density
    # on the considered thunderstorm altitude
    @staticmethod
    def air_density_from_altitude(altitude):
        altitude_array = np.arange(0, 22, 2)  # km
        air_density = np.array([1.2250, 1.0065, 0.8194, 0.6601, 0.5258, 0.4135, 0.3119, 0.2279, 0.1665, 0.1216, 0.0889])
        for i in range(len(altitude_array)):
            if altitude == altitude_array[i]:
                return air_density[i]
            if altitude < altitude_array[i]:
                return air_density[i - 1] * np.exp(
                    (altitude - altitude_array[i - 1]) * np.log(air_density[i] / air_density[i - 1]) / (
                                altitude_array[i] - altitude_array[i - 1]))

    def critical_field(self):
        return 276 * self.relative_air_concentration  # kV per m

    def gamma_production_length(self):
        return 600 / 1.225 * 0.4135 / self.relative_air_concentration  # m

    def positron_production_length(self):
        return 5000 / 1.225 * 0.4135 / self.relative_air_concentration  # m

    def gamma_decay_length(self):
        return 1200 / 1.225 * 0.4135 / self.relative_air_concentration  # m

    def positron_annihilation_length(self):
        return 500 / 1.225 * 0.4135 / self.relative_air_concentration  # m

    def rrea_reproduction_by_positrons_length(self):
        return 106.6 / 1.225 * 0.4135 / self.relative_air_concentration  # m

    def rrea_reproduction_by_positrons_length_alternative(self):
        return 106.6 * (200 - 276 / 1.225 * 0.4135) / (self.electric_field - 276 * self.relative_air_concentration)  # m

    def rrea_reproduction_by_gamma_length(self):
        return 1050 / 1.225 * 0.4135 / self.relative_air_concentration  # m

    def dwyer_avalanche_growth_length(self):
        if self.electric_field / 285 / self.relative_air_concentration > 1:
            return 7300 / (
                        self.electric_field - 276 * self.relative_air_concentration)  # returns rrea growth length in m
        # elif electric_field > 285 * relative_air_concentration:
        #   return 5100 / (electric_field - 285 * relative_air_concentration)  # returns rrea growth length in m
        else:
            raise Exception('Semicritical electric field')

    def isotropic_reversal(self):
        x = self.electric_field / self.relative_air_concentration / 1.2
        a = 0.5
        b = 3.03775569
        c = 0.00743241
        if (c * x - b) < 0:
            return a + (1 - a) * math.erf(c * x - b)
        else:
            return a + (1 - a) * (c * x - b) / (1 + (c * x - b))

    def dwyer_electron_reversal(self):
        reversal = 0.538 * (1 - np.exp((-1 * self.electric_field + self.relative_air_concentration * 235) / (
                    self.relative_air_concentration * 310))) + np.heaviside(
            self.electric_field - self.relative_air_concentration * 1000, 0) * 0.268 * (
                               self.electric_field - self.relative_air_concentration * 1000) / (
                               self.relative_air_concentration * 2000)
        if reversal > 1:
            return 1
        elif reversal < 0:
            return 0
        else:
            return reversal

    def dwyer_positron_reversal(self):
        reversal = 0.84 * (1 - np.exp((-1 * self.electric_field + self.relative_air_concentration * 150) / (
                    self.relative_air_concentration * 400))) * (1 - np.exp(
            (-1 * self.electric_field + self.relative_air_concentration * 276) / (
                        self.relative_air_concentration * 55)))
        if reversal > 1:
            return 1
        elif reversal < 0:
            return 0
        else:
            return reversal

    def electron_propagation_between_cells_fracture(self, energy_min=0.1, energy_max=40):
        #  energy in MeV, electric field in kV per m
        electric_field = self.electric_field_between_cells / 1000  # MeV / m
        critical_electric_field = self.critical_field() / 1000  # MeV / m
        return (np.exp(
            -(energy_min + self.distance_between_cells * (critical_electric_field - electric_field)) / 7.3) - np.exp(
            -energy_max / 7.3)) / (np.exp(-energy_min / 7.3) - np.exp(-energy_max / 7.3))


#  local multiplication factor in the multicell reactor without runaway electron transport
def modified_local_gain_coefficient(process_length, cell_length):
    return process_length.isotropic_reversal() * process_length.dwyer_avalanche_growth_length() / process_length.gamma_production_length() / cell_length * (process_length.dwyer_avalanche_growth_length() * np.exp(cell_length / process_length.dwyer_avalanche_growth_length()) - process_length.dwyer_avalanche_growth_length() - cell_length)


#  local multiplication factor in the multicell reactor with runaway electron transport
def local_gain_coefficient_with_electrons(process_length, cell_length):
    return modified_local_gain_coefficient(process_length, cell_length) + process_length.dwyer_avalanche_growth_length() / process_length.gamma_production_length() * (np.exp(cell_length / process_length.dwyer_avalanche_growth_length()) - 1) * 0.5 * process_length.electron_propagation_between_cells_fracture() * process_length.isotropic_reversal() * process_length.dwyer_avalanche_growth_length() / cell_length * (np.exp(cell_length / process_length.dwyer_avalanche_growth_length()) - 1) / (1 - 0.5 * process_length.electron_propagation_between_cells_fracture() * process_length.isotropic_reversal() * np.exp(cell_length / process_length.dwyer_avalanche_growth_length()))


#  global multiplication factor in the multicell reactor
def global_multiplication_factor(process_length, local_multiplication_factor, thundercloud_spatial_scale, cell_length):
    return process_length.rrea_reproduction_by_gamma_length() * process_length.speed_of_light / 3 * (3 * (local_multiplication_factor - 1) / process_length.rrea_reproduction_by_gamma_length() / process_length.rrea_reproduction_by_gamma_length() - np.square(math.pi / thundercloud_spatial_scale) - np.square(2.404 / thundercloud_spatial_scale / 2))
