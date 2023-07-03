import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Function to generate season date ranges for a given year


def get_season_dates(year: int) -> tuple:

    autumn_winter_start = mdates.datestr2num(f"{year}-03-22")
    autumn_winter_end = mdates.datestr2num(f"{year}-09-20")
    spring_summer_start = mdates.datestr2num(f"{year}-09-22")
    spring_summer_end = mdates.datestr2num(f"{year+1}-03-20")

    return (
        spring_summer_start,
        spring_summer_end,
        autumn_winter_start,
        autumn_winter_end,
    )


# Function to generate a time series plot of a polygon with colored background according to season


def plot_ts(
    *pol_list: pd.Series,
    start_date: pd.Timestamp = pd.Timestamp("2000-01-03"),
    end_date: pd.Timestamp = pd.Timestamp("2022-12-25"),
    size: tuple = (25, 5),
    show_dots: bool = False,
):

    fig, ax = plt.subplots()

    for i, pol in enumerate(pol_list):

        # Filter polygon dates
        pol = pol[(pol.index >= start_date) & (pol.index <= end_date)]

        # Add dots to each value if required
        if show_dots:
            ax.scatter(pol.index, pol)

        # Plot
        ax.plot(pol)

    # Set every year as an x-axis tick
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # Set figure size
    plt.gcf().set_size_inches(*size)

    # Add season color bands
    start_year = pol.index.min().year
    end_year = pol.index.max().year

    for year in range(start_year, end_year + 1):
        (
            spring_summer_start,
            spring_summer_end,
            autumn_winter_start,
            autumn_winter_end,
        ) = get_season_dates(year)
        ax.axvspan(spring_summer_start, spring_summer_end, facecolor="red", alpha=0.3)
        ax.axvspan(autumn_winter_start, autumn_winter_end, facecolor="blue", alpha=0.3)

    # Add a custom legend
    legend_elements = [
        Patch(facecolor="red", alpha=0.3, label="Spring & Summer"),
        Patch(facecolor="blue", alpha=0.3, label="Autum & Winter"),
    ]
    ax.legend(handles=legend_elements)

    # Set the labels
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean NDVI")

    # Set xticks
    plt.xticks(rotation=45)
    plt.show()
