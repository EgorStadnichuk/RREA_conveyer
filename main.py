import data_reader


def main():
    read_data = data_reader.DataReader()
    data = read_data.get_data(electric_field=140, volume=500, density=0.6)
    print(data)


if __name__ == '__main__':
    main()
