import csv
import os
import numpy as np
import high_energy_atmospheric_physics_process_length


class DataReader:
    path = os.getcwd() + '\\Ph_list_ Ph_option4\\'
    process_length = high_energy_atmospheric_physics_process_length.ProcessLength()

    def get_data(self, particle_type='elect', seed_energy=5, electric_field=140, volume=500, density=1.225, cut=40):
        self.check_statistics(electric_field=electric_field, volume=volume, density=density)
        with open(self.path + 'doc_' + particle_type + '_' + str(seed_energy) + '-' + str(electric_field) +
                              'volume' + str(int(volume / 2)) + '_denc' + str(density) + '_cut' + str(cut) + '.csv',
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

    def check_statistics(self, electric_field=140, volume=500, density=1.225):
        self.process_length.electric_field = electric_field
        self.process_length.relative_air_concentration = density / 1.225
        if float(volume) / self.process_length.dwyer_avalanche_growth_length() < 3:
            raise Exception('Too short cell, RREA e-folding length for given parameters = ' +
                            str(self.process_length.dwyer_avalanche_growth_length()) + ' m, cell length = ' +
                            str(volume) + ' m. Cell length should be at least 3 times bigger than RREA growth length.')
