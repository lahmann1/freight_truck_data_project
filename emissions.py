import pandas as pd
import matplotlib.pyplot as plt
import addcopyfighandler
import numpy as np

from labels import labels


def get_data(path):
    # Read the data frame
    df = pd.read_csv(path)

    # Trucks only
    df = df[df['dms_mode'] == 1]                                        # Domestic truck
    df = df[(df['fr_inmode'] == 1) | (df['fr_inmode'].isnull())]        # Imported by truck
    df = df[(df['fr_outmode'] == 1) | (df['fr_outmode'].isnull())]      # Exported by truck

    # Originate in CA
    mask = df['dms_orig'] == 61
    for id in [61, 62, 63, 64, 65, 69]:
        mask = mask | (df['dms_orig'] == id)

    # OR arrive in CA
    for id in [61, 62, 63, 64, 65, 69]:
        mask = mask | (df['dms_dest'] == id)

    df = df[mask]
    return df


def main():
    data = get_data('data/FAF5_CA_Truck.csv')

    # PLOT THE EMISSIONS BY CAT
    hist = plt.hist(data['sctg2'], weights=161.8 * data['tmiles_2022'], bins=np.arange(0.5, 43.5), edgecolor='k')
    plt.close()

    n = hist[0]
    indexes = np.argsort(n)[::-1]
    n = n[indexes]
    sorted_labels = np.array(labels)[indexes]

    plt.bar(np.arange(1, 43), n, edgecolor='k')
    plt.title('Emissions by Category')
    plt.xticks(np.arange(1, 43), sorted_labels, rotation=90)
    plt.ylabel('Metric Tons of CO2')
    plt.tight_layout()
    plt.show()

    # PLOT THE VALUE BY CAT
    hist = plt.hist(data['sctg2'], weights=1e-3 * data['value_2022'], bins=np.arange(0.5, 43.5), edgecolor='k')
    plt.close()

    n = hist[0]
    indexes = np.argsort(n)[::-1]
    n = n[indexes]
    sorted_labels = np.array(labels)[indexes]

    plt.bar(np.arange(1, 43), n, edgecolor='k')
    plt.title('Value by Category')
    plt.xticks(np.arange(1, 43), sorted_labels, rotation=90)
    plt.ylabel('Billion USD')
    plt.tight_layout()
    plt.show()

    # PLOT THE MILES BY CAT
    miles = 1e3 * data['tmiles_2022'] / data['tons_2022']
    hist = plt.hist(data[miles.notnull()]['sctg2'], weights=miles[miles.notnull()], bins=np.arange(0.5, 43.5), edgecolor='k')
    plt.close()

    n = hist[0]
    indexes = np.argsort(n)[::-1]
    n = n[indexes]
    sorted_labels = np.array(labels)[indexes]

    plt.bar(np.arange(1, 43), n, edgecolor='k')
    plt.title('Miles by Category')
    plt.xticks(np.arange(1, 43), sorted_labels, rotation=90)
    plt.ylabel('Miles')
    plt.tight_layout()
    plt.show()

    # PLOT THE NUMBER OF TRUCKS BY CAT
    truck_labels = {
        32: 'Long Haul',
        21: 'Short Haul',
        6.9: 'Intercity'
    }
    for capacity in truck_labels:
        hist = plt.hist(data['sctg2'], weights=1e3 * data['tons_2022'] / capacity / 3 / 1.1, bins=np.arange(0.5, 43.5), edgecolor='k')
        plt.close()

        n = hist[0]
        indexes = np.argsort(n)[::-1]
        n = n[indexes]
        sorted_labels = np.array(labels)[indexes]

        plt.bar(np.arange(1, 43), n, edgecolor='k')

        plt.title('Number of Trips for %s trucks' % truck_labels[capacity])
        plt.xticks(np.arange(1, 43), sorted_labels, rotation=90)
        plt.ylabel('Minimum number of trips')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
