import data_reader
import data_analysis


def main():
    read_data = data_reader.DataReader()
    analyse = data_analysis.Analyse()
    energy, position, direction = read_data.get_data(electric_field=140, cell_length=500, density=0.2)
    analyse.plot_energy_spectrum(energy)


if __name__ == '__main__':
    main()
