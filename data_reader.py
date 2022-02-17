import csv
import os
import numpy as np
import high_energy_atmospheric_physics_process_length


class DataReader:
    path = os.getcwd() + '\\Ph_list_ Ph_option4\\'  # path to Geant4 data
    process_length = high_energy_atmospheric_physics_process_length.ProcessLength()  # class, high-energy process
    # lengths calculator

    def get_data(self, particle_type='elect', seed_energy=5, electric_field=140,
                 cell_length=500, density=1.225, cut=40):
        """
        get_data() reads .csv file with Geant4 simulation results.
        :param particle_type: 'elect' for electrons, 'gamma' for gamma-ray photons, 'pos' for positrons.
        :param seed_energy: the energy of the seed particle in [MeV].
        :param electric_field: cell electric field in [kV per m].
        :param cell_length: cell length in [m].
        :param density: air density within the cell in [kg per m**3].
        :param cut: simulation energy cut in [keV].
        :return: energy, position, direction
        energy: 1-d numpy array with particles kinetic energy in [MeV]. Number of element corresponds to number of a
        particle.
        position: 2-d numpy array with particles' positions relative to the centre of the cell. position[i, j]:
        i - number of a particle, j - coordinate (0 for x, 1 for y, 2 for z).
        direction: 2-d numpy array with particles' velocity directions. position[i, j]:
        i - number of a particle, j - coordinate (0 for x, 1 for y, 2 for z). Positive position[i, 2] means that
        particle goes in the direction opposite to the electric field direction (electrons accelerate in this way).
        """
        self.check_statistics(electric_field=electric_field, cell_length=cell_length, density=density)
        with open(self.path + 'doc_' + particle_type + '_' + str(seed_energy) + '-' + str(electric_field) +
                              'volume' + str(int(cell_length / 2)) + '_denc' + str(density) + '_cut' + str(cut) + '.csv',
                  newline='') as File:
            reader = csv.reader(File, delimiter=' ')
            data = []
            for row in reader:
                data.append(row)
        energy = np.zeros(len(data))
        position = np.zeros(len(data) * 3).reshape(len(data), 3)
        direction = np.zeros(len(data) * 3).reshape(len(data), 3)
        for i in range(len(data)):
            energy[i] = float(data[i][3])
            position[i, :] = np.fromstring(data[i][5].replace('(', '').replace(')', ''), dtype=np.float, sep=',')
            direction[i, :] = np.fromstring(data[i][7].replace('(', '').replace(')', ''), dtype=np.float, sep=',')
        return energy, position, direction

    def check_statistics(self, electric_field=140, cell_length=500, density=1.225):
        """
        simple statistics check: if the cell length is less than 3 times bigger
        than RREA e-folding length then the data is considered to be with low statistics and, therefore, unreliable
        :param electric_field: cell electric field in [kV per m]
        :param cell_length: cell length in [m]
        :param density: air density within the cell in [kg per m**3]
        :return: raises exception if the cell is too short
        """
        self.process_length.electric_field = electric_field
        self.process_length.relative_air_concentration = density / 1.225
        if float(cell_length) / self.process_length.dwyer_avalanche_growth_length() < 3:
            raise Exception('Too short cell, RREA e-folding length for given parameters = ' +
                            str(self.process_length.dwyer_avalanche_growth_length()) + ' m, cell length = ' +
                            str(cell_length) + ' m.\nCell length should be at least ' +
                                               '3 times bigger than RREA growth length.')
        pass
