import pandas as pd
import matplotlib.pyplot as plt
import addcopyfighandler
import numpy as np
import statsmodels.api as sm

from labels import labels


def get_data(path):
    # Read the data frame
    df = pd.read_csv(path)

    # Domestic only
    df = df[df['trade_type'] == 1]

    # Trucks only
    df = df[df['dms_mode'] == 1]  # Domestic truck
    df = df[(df['fr_inmode'] == 1) | (df['fr_inmode'].isnull())]  # Imported by truck
    df = df[(df['fr_outmode'] == 1) | (df['fr_outmode'].isnull())]  # Exported by truck

    # Originate in CA
    mask = df['dms_orig'] == 61
    for id in [61, 62, 63, 64, 65, 69]:
        mask = mask | (df['dms_orig'] == id)

    # OR arrive in CA
    for id in [61, 62, 63, 64, 65, 69]:
        mask = mask | (df['dms_dest'] == id)

    df = df[mask]
    return df


# Classes 2B
def add_fields(data, truck_info=None):
    if truck_info is None:
        truck_info = {
            'Class 4-7': {
                'classes': [4, 5, 6, 7],
                'min_weight': 14001,
                'max_weight': 33000,
                'min_distance': 0.0,
                'max_distance': 186.0,
                'emissions': 1413.0,
                'lifetime': 15.0,
                'annual_miles': 55000,
                'cost_to_transition': 250000
            },
            'Class 8': {
                'classes': [8],
                'min_weight': 33001,
                'max_weight': 80000,
                'min_distance': 186.0,
                'max_distance': 1e36,
                'emissions': 2350.0,
                'lifetime': 15.0,
                'annual_miles': 85000,
                'cost_to_transition': 440000
            }
        }

    # Calculate the miles of each flow
    data['flow_miles_2022'] = 1e3 * data['tmiles_2022'] / data['tons_2022']

    # Disregard data where miles is null (very low weight probably)
    data = data[data['flow_miles_2022'].notnull()]

    # Assign a truck type based on the distance of the flow
    # Also calculate emissions based on the assumed truck type
    for type in truck_info:
        mask = (data['flow_miles_2022']
                >= truck_info[type]['min_distance']) & (data['flow_miles_2022'] < truck_info[type]['max_distance'])
        data.loc[mask, 'truck_type'] = type
        data.loc[mask, 'min_num_trips'] = 1000.0 * data.loc[mask, 'tons_2022'] / (truck_info[type]['max_weight'] / 2000.0)
        data.loc[mask, 'min_total_miles'] = data.loc[mask, 'min_num_trips'] * data.loc[mask, 'flow_miles_2022']
        data.loc[mask, 'min_trucks'] = np.ceil(data.loc[mask, 'min_total_miles'] * truck_info[type]['lifetime'] / truck_info[type]['annual_miles'])
        data.loc[mask, 'emissions_2022'] = 1e-6 * truck_info[type]['emissions'] * data.loc[mask, 'min_total_miles']
        data.loc[mask, 'total_transition_cost'] = data.loc[mask, 'min_trucks'] * truck_info[type]['cost_to_transition']
        data.loc[mask, 'transition_rate'] = data.loc[mask, 'min_trucks'] / 20.0

    print(data.columns)
    return data


def make_plot(data, weights, title="", y_label="", highlight_indexes=None, color='steelblue', highlight_color='midnightblue', width=0.8, offset=0.0, label=""):
    hist = np.histogram(data['sctg2'], weights=weights, bins=np.arange(0.5, 43.5))
    n = hist[0]

    indexes = np.argsort(n)[::-1]
    n = n[indexes]
    sorted_labels = np.array(labels)[indexes]

    plt.bar(np.arange(1, 43) - offset, n, edgecolor='k', color=color, width=width, label=label)
    plt.title(title)
    plt.xticks(np.arange(1, 43), sorted_labels, rotation=90)
    plt.ylabel(y_label)
    if highlight_indexes is not None:
        temp = np.in1d(indexes, highlight_indexes).nonzero()[0]
        plt.bar(temp + 1, n[temp], edgecolor='k', color=highlight_color)
        print('Total %s: %.2e' % (y_label, n.sum()))
        print('Total highlighted %s: %.2e' % (y_label, n[highlight_indexes].sum()))
        print('%.1f percent' % (100.0 * n[highlight_indexes].sum() / n.sum()))
    plt.tight_layout()

    return indexes


def linear_regression(data):
    dependent_var = 'min_trucks'
    independent_vars = ['tmiles_2022', 'tons_2022', 'current_value_2022']
    Y = data[dependent_var].to_numpy().astype(float)
    X = data[independent_vars].to_numpy().astype(float)
    # X = sm.add_constant(X)
    mod = sm.OLS(Y, X)
    mod.exog_names[:] = independent_vars
    # mod.exog_names[:] = ['constant'] + independent_vars
    fii = mod.fit()
    print(fii.summary())

    print(np.arange(len(X)).shape, X.shape)
    indexes = np.argsort(Y)
    plt.scatter(np.arange(len(Y)), fii.fittedvalues[indexes], label='Model')
    plt.scatter(np.arange(len(Y)), Y[indexes], label='Data')
    plt.yscale('log')
    plt.legend()
    plt.ylabel('Number of Trucks')
    plt.xticks([], [])
    plt.xlabel('Flow sorted by Number of Trucks')
    plt.ylim([1, 10**5])
    plt.show()


def main():
    # Get the data
    data = get_data('data/FAF5_CA_Truck.csv')

    # Add some fields to the data
    data = add_fields(data)

    # Do a linear regression
    linear_regression(data)

    # Get the top 5 emitters todo: weight based on vehicle type
    indexes = make_plot(data, data['emissions_2022'])
    top_5_emitters = indexes[0:5]

    # Plot emissions
    make_plot(data, data['emissions_2022'], "Emissions by Category", "Tonnes of CO2", top_5_emitters)
    plt.show()

    # Plot values
    make_plot(data, 1e-3 * data['current_value_2022'], "Value by Category", "Billion USD", top_5_emitters)
    plt.show()

    # Plot miles
    make_plot(data, data['min_total_miles'], "Miles by Category", "Miles", top_5_emitters)
    plt.show()

    # Plot trucks
    make_plot(data, data['min_trucks'], "Trucks by Category", "Trucks", top_5_emitters)
    plt.show()

    # Plot Transition Costs
    make_plot(data, 1e-9 * data['total_transition_cost'], "Transition Costs by Category", "Billion USD", top_5_emitters)
    plt.show()

    # Plot Transition Truck Rate
    make_plot(data, data['transition_rate'], "Transition Rates by Category", "Trucks per Year", top_5_emitters)
    plt.show()

    # Divide by truck type
    temp = data[data['truck_type'] == 'Class 4-7']
    make_plot(temp, temp['min_trucks'], "Trucks by Category", "Trucks", offset=0.2, width=0.4, color='r', label='Class 4-7')
    temp = data[data['truck_type'] == 'Class 8']
    make_plot(temp, temp['min_trucks'], "Trucks by Category", "Trucks", offset=-0.2, width=0.4, color='b', label='Class 8')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
