import numpy as np
import matplotlib.pyplot as plt


class Analyse:

    @staticmethod
    def plot_energy_spectrum(energy_data):
        plt.hist(energy_data)
        plt.show()
        pass
