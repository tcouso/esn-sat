import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def get_season_dates(year: int) -> tuple:
    """
    Function to generate season date ranges for a given year.

    Parameters:
        year (int): The year for which to generate the season date ranges.

    Returns:
        tuple: A tuple containing the start and end dates for the spring & summer and autumn & winter seasons.
    """
    # Convert season start and end dates to numerical format
    autumn_winter_start = mdates.datestr2num(f"{year}-03-22")  # March 22nd
    autumn_winter_end = mdates.datestr2num(f"{year}-09-20")  # September 20th
    spring_summer_start = mdates.datestr2num(f"{year}-09-22")  # September 22nd
    spring_summer_end = mdates.datestr2num(
        f"{year+1}-03-20"
    )  # March 20th of the following year

    return (
        spring_summer_start,
        spring_summer_end,
        autumn_winter_start,
        autumn_winter_end,
    )


def plot_ts(
    *pol_list: pd.Series,
    labels: list = None,
    start_date: pd.Timestamp = pd.Timestamp("2000-01-03"),
    end_date: pd.Timestamp = pd.Timestamp("2022-12-25"),
    size: tuple = (25, 5),
    show_dots: bool = False,
):
    """
    Function to generate a time series plot of multiple polygons with colored background according to season.

    Parameters:
        *pol_list (pd.Series): Variable number of pandas Series objects representing the time series data.
        labels (list, optional): List of labels for each time series (default: None).
        start_date (pd.Timestamp, optional): The start date for the plot (default: January 3, 2000).
        end_date (pd.Timestamp, optional): The end date for the plot (default: December 25, 2022).
        size (tuple, optional): The size of the figure (default: (25, 5)).
        show_dots (bool, optional): Whether to show dots for each data point (default: False).

    Returns:
        None
    """
    # Create figure and axis
    fig, ax = plt.subplots()

    for i, pol in enumerate(pol_list):
        # Filter polygon dates
        pol = pol[(pol.index >= start_date) & (pol.index <= end_date)]

        # Add dots to each value if required
        if show_dots:
            ax.scatter(pol.index, pol)

        # Plot with label
        if labels:
            ax.plot(pol, label=labels[i])
        else:
            ax.plot(pol)

    # Set every year as an x-axis tick
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # Set figure size
    plt.gcf().set_size_inches(*size)

    # Add season color bands
    start_year = pol.index.min().year
    end_year = pol.index.max().year

    for year in range(start_year, end_year + 1):
        # Retrieve season dates for the year
        (
            spring_summer_start,
            spring_summer_end,
            autumn_winter_start,
            autumn_winter_end,
        ) = get_season_dates(year)

        # Add color bands for each season
        ax.axvspan(spring_summer_start, spring_summer_end, facecolor="red", alpha=0.3)
        ax.axvspan(autumn_winter_start, autumn_winter_end, facecolor="blue", alpha=0.3)

    # Add a custom legend
    ax.legend()

    # Set the labels
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean NDVI")

    # Set xticks
    plt.xticks(rotation=45)
    plt.show()
