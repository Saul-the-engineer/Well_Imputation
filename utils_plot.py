import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional
import gc

import numpy as np
import pandas as pd
from scipy import interpolate
import datetime as dt


class Plotter:
    """
    A class for creating various plots and visualizations.

    Args:
        figures_directory (str): The root directory where figures will be saved.

    Attributes:
        figures_directory (str): The root directory where figures will be saved.
    """

    def __init__(self, figures_directory: str) -> None:
        """
        Initialize the Plotter object.

        Args:
            figures_directory (str): The root directory where figures will be saved.
        """
        self.figures_root = figures_directory
        sns.set_style("whitegrid")  # Set Seaborn style
        sns.set_palette("husl")  # Set Seaborn color palette

    def plot_training_metrics(
        self,
        data: Dict[str, list],
        name: str,
        show: bool = False,
        save: bool = True,
    ) -> None:
        """
        Plot training metrics.

        :param data: A dictionary containing training metrics data.
        :type data: Dict[str, list]
        :param name: The name for the plot and saved file.
        :type name: str
        :param show: Whether to display the plot. Defaults to False.
        :type show: bool, optional
        """
        fig, ax = plt.subplots()

        for key, values in data.items():
            ax.plot(values, label=key)

        ax.grid(True)
        ax.set_ylim(-0.5, 5)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        ax.legend(data.keys(), title="Legend")
        ax.set_title(f"Training History: {name}")

        if save:
            plt.savefig(f"{self.figures_root}/{name}_training_history.png")

        if not show:
            plt.close(fig)

        gc.collect()


def plot_groundwater_trends(
    self,
    interpolated_data: pd.Series,
    extrapolated_data_df: pd.DataFrame,
    extrapolation_metadata_df: pd.DataFrame,
    raw_data: pd.Series,
    plot_name: str,
    display_plot: bool = False,
    save_plot: bool = True,
) -> None:
    """
    Plot groundwater level trends.

    :param interpolated_data: Interpolated data for the trends.
    :type interpolated_data: pd.Series
    :param extrapolated_data_df: Extrapolated data for the left and right percentages.
    :type extrapolated_data_df: pd.DataFrame
    :param extrapolation_metadata_df: Metadata for left and right percentages.
    :type extrapolation_metadata_df: pd.DataFrame
    :param raw_data: Raw groundwater level data.
    :type raw_data: pd.Series
    :param plot_name: The name for the plot and saved file.
    :type plot_name: str
    :param display_plot: Whether to display the plot, defaults to False.
    :type display_plot: bool, optional
    :param save_plot: Whether to save the plot as an image, defaults to True.
    :type save_plot: bool, optional
    """

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    num_plots = len(extrapolated_data_df.columns)

    colormap = plt.cm.nipy_spectral
    colors = [colormap(i) for i in np.linspace(0.3, 0.9, num_plots)]
    ax1.set_prop_cycle("color", colors)

    for i in range(num_plots):
        ax1.plot(extrapolated_data_df.index, extrapolated_data_df.iloc[:, i])
    ax1.plot(interpolated_data.index, interpolated_data, color="dimgrey")
    ax1.scatter(raw_data.index, raw_data, color="none", edgecolors="black", s=11)
    ax1.legend(
        extrapolated_data_df.columns.tolist() + ["Prior", "Observations"],
        title="Total Data Percentage",
    )
    ax1.set_title("Prior")
    ax1.set_ylabel("Groundwater Level")
    plt.tight_layout()

    if save_plot:
        plt.savefig(f"{self.figures_root}/{plot_name}_trend.png")

    if display_plot:
        plt.show(fig)
    else:
        fig.clf()
        plt.close(fig)
    gc.collect()

    def plot_prior_features(
        self,
        y: pd.Series,
        rw: pd.DataFrame,
        name: str,
        save: bool = False,
        display: bool = False,
    ) -> None:
        """
        Plot long-term trends with observations.

        Args:
            y (pd.Series): Groundwater level observations.
            rw (pd.DataFrame): Long-term trends data.
            name (str): The name for the plot and saved file.
            save (bool, optional): Whether to save the plot as an image. Defaults to False.
            extension (str, optional): The file extension for saving the plot. Defaults to ".png".
            show (bool, optional): Whether to display the plot. Defaults to False.
        """
        fig = plt.figure(figsize=(12, 8))
        plt.plot(rw)
        plt.scatter(y.index, y, s=3, c="black")
        plt.ylabel("Groundwater Level")
        plt.legend(rw.columns.tolist() + ["Observations"])
        plt.title(f"Long-Term Trends: {name}")

        if save:
            plt.savefig(f"{self.figures_root}/{name}_prior_features.png")

        if display:
            plt.show()
        else:
            fig.clf()
            plt.close(fig)

        gc.collect()

    def plot_qq(
        self,
        prediction: pd.Series,
        observation: pd.Series,
        name: str,
        limit_low: Union[int, float] = 0,
        limit_high: Union[int, float] = 1,
        show: bool = False,
        save: bool = True,
    ) -> None:
        """
        Plot Prediction vs. Observation correlation.

        Args:
            prediction (pd.Series): Predicted values.
            observation (pd.Series): Observed values.
            name (str): The name for the plot and saved file.
            limit_low (Union[int, float], optional): Lower limit for axes. Defaults to 0.
            limit_high (Union[int, float], optional): Upper limit for axes. Defaults to 1.
            extension (str, optional): The file extension for saving the plot. Defaults to ".png".
            show (bool, optional): Whether to display the plot. Defaults to False.
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.scatter(prediction, observation)
        plt.ylabel("Observation")
        plt.xlabel("Prediction")
        plt.legend(["Prediction", "Observation"])
        plt.title(f"Prediction Correlation: {name}")

        cor_line_x = np.linspace(limit_low, limit_high, 9)
        cor_line_y = cor_line_x
        plt.xlim(limit_low, limit_high)
        plt.ylim(limit_low, limit_high)
        plt.plot(cor_line_x, cor_line_y, color="r")
        ax1.set_aspect("equal", adjustable="box")

        if save:
            plt.savefig(f"{self.figures_root}/{name}_qq.png")

        if show:
            plt.show()
        else:
            fig.clf()
            plt.close(fig)

        gc.collect()

    def plot_observation_vs_prediction(
        self,
        prediction: pd.Series,
        observation: pd.Series,
        name: str,
        metrics: Optional[Dict[str, pd.Series]] = None,
        error_on: bool = False,
        show: bool = False,
        save: bool = True,
    ) -> None:
        """
        Plot Observation vs. Prediction with optional error metrics.

        Args:
            prediction (pd.Series): Y values for prediction.
            observation (pd.Series): Y values for observations.
            name (str): The name for the plot and saved file.
            metrics (dict, optional): Dictionary containing error metrics.
            error_on (bool, optional): Whether to include error metrics in the plot. Defaults to False.
            show (bool, optional): Whether to display the plot. Defaults to False.
            save (bool, optional): Whether to save the plot. Defaults to True.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(prediction.index, prediction, "darkblue", label="Prediction")
        ax.plot(
            observation.index, observation, label="Observations", color="darkorange"
        )
        ax.set_ylabel("Groundwater Level")
        ax.legend()
        ax.set_title(f"Observation Vs Prediction: {name}")

        if error_on and metrics is not None:
            train_metrics = metrics.get("Train", pd.Series()).to_string(
                index=True, float_format="{0:.3}".format
            )
            validation_metrics = metrics.get("Validation", pd.Series()).to_string(
                index=True, float_format="{0:.3}".format
            )
            ax.text(
                x=0.0,
                y=-0.15,
                s=train_metrics,
                fontsize=12,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.text(
                x=0.25,
                y=-0.15,
                s=validation_metrics,
                fontsize=12,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        if save:
            if error_on:
                extent = ax.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted()
                )
                plt.savefig(
                    f"{self.figures_root}/{name}_observation_vs_prediction.png",
                    bbox_inches=extent.expanded(1.1, 1.6),
                )
            else:
                plt.savefig(f"{self.figures_root}/{name}_observation_vs_prediction.png")

        if show:
            plt.show()
        else:
            plt.close(fig)
        gc.collect()

    def plot_residual_errors(
        self,
        prediction: pd.Series,
        observation: pd.Series,
        name: str,
        show: bool = False,
        save: bool = True,
    ) -> None:
        """
        Plot residual errors and observation data.

        Args:
            prediction (pd.Series): Series containing predicted values.
            observation (pd.Series): Series containing observed values.
            name (str): The name for the plot and saved file.
            show (bool, optional): Whether to display the plot. Defaults to False.
            save (bool, optional): Whether to save the plot. Defaults to True.
        """
        date_rng = pd.DataFrame(
            np.arange(0, len(observation), 1), index=observation.index
        )
        data = pd.concat(
            [prediction, observation], axis=1, keys=["Prediction_Y", "Observation_Y"]
        ).dropna()
        data["Spline Index"] = np.arange(len(data))
        residual = data["Prediction_Y"] - data["Observation_Y"]
        spline = interpolate.UnivariateSpline(data["Spline Index"], residual, s=1000000)

        fig, axs = plt.subplots(2, 1, figsize=(12, 12))
        axs[0].plot(
            data.index,
            residual,
            marker="o",
            linestyle="None",
            markersize=5,
            color="black",
        )
        axs[0].plot(
            observation.index,
            np.zeros(shape=(len(observation), 1)),
            color="royalblue",
            linewidth=2.0,
        )
        axs[0].plot(
            data.index, spline(data["Spline Index"]), color="red", linewidth=3.0
        )
        axs[0].legend(["Residual", "Zero", "Spline Trend"])
        axs[0].set_ylabel("Prediction Residual Error")
        axs[0].set_title(f"Residual Error: {name}")

        axs[1].plot(
            observation.index,
            observation,
            marker="o",
            linestyle="None",
            markersize=5,
            color="black",
        )
        axs[1].plot(
            observation.index,
            observation.mean() * np.ones(shape=(len(observation), 1)),
            color="royalblue",
            linewidth=2.0,
        )
        axs[1].legend(["Observations", "Mean"])
        axs[1].set_ylabel("Groundwater Level")
        axs[1].set_title(f"Groundwater Observations: {name}")

        if save:
            plt.savefig(f"{self.figures_root}/{name}_residual_errors")

        if show:
            plt.show()
        else:
            plt.close(fig)
        gc.collect()

    def plot_imputation_vs_observation(
        self,
        prediction: pd.Series,
        observation: pd.Series,
        name: str,
        show: bool = False,
        save: bool = True,
    ) -> None:
        """
        Plot observation vs. imputation.

        Args:
            prediction (pd.Series): Series containing imputed values.
            observation (pd.Series): Series containing observations.
            name (str): The name for the plot and saved file.
            show (bool, optional): Whether to display the plot. Defaults to False.
            save (bool, optional): Whether to save the plot. Defaults to True.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(prediction.index, prediction, "darkblue", label="Imputed Values")
        ax.plot(
            observation.index,
            observation,
            label="Smoothed Observations",
            color="darkorange",
        )
        ax.set_ylabel("Groundwater Level")
        ax.set_xlabel("Date")
        ax.legend()
        ax.set_title(f"Observation Vs Imputation: {name}")

        if save:
            plt.savefig(f"{self.figures_root}/{name}_imputation_vs_observation.png")

        if show:
            plt.show()
        else:
            plt.close(fig)
        gc.collect()

    def plot_predictions_vs_raw_observation(
        self,
        prediction: pd.Series,
        raw: pd.Series,
        name: str,
        aquifer: str,
        metrics: pd.DataFrame = None,
        error_on: bool = False,
        test: bool = False,
        show: bool = False,
        save: bool = True,
    ) -> None:
        """
        Plot predictions vs. raw observations.

        Args:
            prediction (pd.Series): Series containing predicted values.
            raw (pd.Series): Series containing raw observations.
            name (str): The name for the plot and saved file.
            aquifer (str): Name of the aquifer.
            metrics (pd.DataFrame, optional): DataFrame containing metrics data. Defaults to None.
            error_on (bool, optional): Whether to display error metrics. Defaults to False.
            test (bool, optional): Whether to include test metrics. Defaults to False.
            show (bool, optional): Whether to display the plot. Defaults to False.
            save (bool, optional): Whether to save the plot. Defaults to True.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(
            prediction.index, prediction, "darkblue", label="Prediction", linewidth=1.0
        )
        ax.scatter(
            raw.index, raw, color="darkorange", marker="*", s=10, label="Observations"
        )
        ax.set_title(f"{aquifer}: Well: {name} Predicted Values")
        ax.set_ylabel("Groundwater Level")
        ax.legend(fontsize="medium")

        if error_on:
            train_metrics = metrics[
                ["Train ME", "Train RMSE", "Train MAE", "Train r2"]
            ].to_string(index=True, float_format="{0:.3}".format)
            validation_metrics = metrics[
                ["Validation ME", "Validation RMSE", "Validation MAE", "Validation r2"]
            ].to_string(index=True, float_format="{0:.3}".format)
            ax.text(
                x=0.0,
                y=-0.15,
                s=train_metrics,
                fontsize=12,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.text(
                x=0.25,
                y=-0.15,
                s=validation_metrics,
                fontsize=12,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            if test:
                test_metrics = metrics[
                    ["Test ME", "Test RMSE", "Test MAE", "Test r2"]
                ].to_string(index=True, float_format="{0:.3}".format)
                ax.text(
                    x=0.5,
                    y=-0.15,
                    s=test_metrics,
                    fontsize=12,
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

            if save:
                fig.savefig(
                    f"{self.figures_root}/{name}_prediction_vs_raw",
                    bbox_inches=extent.expanded(1.3, 1.6),
                )
        else:
            if save:
                fig.savefig(f"{self.figures_root}/{name}_prediction_vs_raw")

        if show:
            plt.show()
        else:
            plt.close(fig)
        gc.collect()

    def plot_filled_gaps_vs_raw_observation(
        self,
        prediction: pd.Series,
        raw: pd.Series,
        name: str,
        aquifer: str,
        df_spread: pd.DataFrame = None,
        conf_interval: pd.Series = None,
        ci: int = 1,
        metrics: pd.DataFrame = None,
        error_on: bool = False,
        test: bool = False,
        show: bool = False,
        save: bool = False,
    ) -> None:
        """
        Plot filled gaps vs. raw observations.

        :param prediction: The imputed values for plotting.
        :type prediction: pd.Series
        :param raw: The raw observations for plotting.
        :type raw: pd.Series
        :param name: The name of the well.
        :type name: str
        :param aquifer: The aquifer name.
        :type aquifer: str
        :param df_spread: DataFrame with spread data, optional.
        :type df_spread: pd.DataFrame, optional
        :param conf_interval: Confidence interval data, optional.
        :type conf_interval: pd.Series, optional
        :param ci: Confidence interval factor, defaults to 1.
        :type ci: int, optional
        :param metrics: DataFrame with metrics data, optional.
        :type metrics: pd.DataFrame, optional
        :param error_on: Whether to include error metrics, defaults to False.
        :type error_on: bool, optional
        :param test: Whether to include test metrics, defaults to False.
        :type test: bool, optional
        :param show: Whether to display the plot, defaults to False.
        :type show: bool, optional
        :param save: Whether to save the plot, defaults to False.
        :type save: bool, optional
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(
            prediction.index, prediction, "darkblue", label="Prediction", linewidth=1.0
        )
        ax.plot(raw.index, raw, color="darkorange", label="Observations", linewidth=1.0)
        ax.set_title(f"{aquifer}: Well: {name} Imputed Values")
        ax.set_ylabel("Groundwater Level")

        if conf_interval and not df_spread.empty:
            upper = df_spread["mean"] + df_spread["std"] * ci
            lower = df_spread["mean"] - df_spread["std"] * ci
            data_min = min(prediction.min(), raw.min())
            data_max = max(prediction.max(), raw.max())
            ax.fill_between(
                df_spread.index,
                upper,
                lower,
                color="blue",
                alpha=0.2,
                label=f"Std Dev: {ci}",
            )
            ax.set_ylim(data_min - 5, data_max + 5)

        ax.legend(fontsize="medium")

        if error_on:
            train_metrics = metrics[
                ["Train ME", "Train RMSE", "Train MAE", "Train r2"]
            ].to_string(index=True, float_format="{0:.3}".format)
            validation_metrics = metrics[
                ["Validation ME", "Validation RMSE", "Validation MAE", "Validation r2"]
            ].to_string(index=True, float_format="{0:.3}".format)
            ax.text(
                x=0.0,
                y=-0.15,
                s=train_metrics,
                fontsize=12,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.text(
                x=0.25,
                y=-0.15,
                s=validation_metrics,
                fontsize=12,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            if test:
                test_metrics = metrics[
                    ["Test ME", "Test RMSE", "Test MAE", "Test r2"]
                ].to_string(index=True, float_format="{0:.3}".format)
                ax.text(
                    x=0.5,
                    y=-0.15,
                    s=test_metrics,
                    fontsize=12,
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

            if save:
                fig.savefig(
                    f"{self.figures_root}/{name}_05_Filled_vs_Raw",
                    bbox_inches=extent.expanded(1.3, 1.6),
                )
        elif save:
            fig.savefig(f"{self.figures_root}/{name}_05_Filled_vs_Raw")

        if show:
            plt.show()
        else:
            plt.close(fig)
        gc.collect()

    def plot_impution_vs_raw_observation(
        self,
        prediction: pd.Series,
        raw: pd.Series,
        name: str,
        aquifer: str,
        metrics: pd.DataFrame = None,
        show: bool = False,
        save: bool = False,
    ) -> None:
        """
        Plot imputation vs. raw observations.

        :param prediction: The model's imputed values for plotting.
        :type prediction: pd.Series
        :param raw: The raw observations for plotting.
        :type raw: pd.Series
        :param name: The name of the well.
        :type name: str
        :param aquifer: The aquifer name.
        :type aquifer: str
        :param metrics: DataFrame with optional metrics data, defaults to None.
        :type metrics: pd.DataFrame, optional
        :param show: Whether to display the plot, defaults to False.
        :type show: bool, optional
        :param save: Whether to save the plot, defaults to False.
        :type save: bool, optional
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(prediction.index, prediction, "darkblue", label="Model", linewidth=1.0)
        ax.scatter(
            raw.index, raw, color="darkorange", marker="*", s=10, label="Observations"
        )
        ax.set_title(f"{aquifer}: Well: {name} Raw vs Model")
        ax.legend(fontsize="medium")

        if save:
            plt.savefig(f"{self.figures_root}/{name}_06_Imputation_vs_Raw")

        if show:
            plt.show()
        else:
            plt.close(fig)
        gc.collect()

    def plot_scatter_prediction_vs_observation(
        self,
        prediction: pd.Series,
        y_train: pd.Series,
        y_val: pd.Series,
        name: str,
        metrics: pd.DataFrame = None,
        error_on: bool = False,
        show: bool = False,
        save: bool = False,
    ) -> None:
        """
        Plot a scatter plot of prediction vs. observation.

        :param prediction: The model's predicted values for plotting.
        :type prediction: pd.Series
        :param y_train: Training data for plotting.
        :type y_train: pd.Series
        :param y_val: Validation data for plotting.
        :type y_val: pd.Series
        :param name: The name of the well.
        :type name: str
        :param metrics: DataFrame with optional metrics data, defaults to None.
        :type metrics: pd.DataFrame, optional
        :param error_on: Whether to include error metrics, defaults to False.
        :type error_on: bool, optional
        :param show: Whether to display the plot, defaults to False.
        :type show: bool, optional
        :param save: Whether to save the plot, defaults to False.
        :type save: bool, optional
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(prediction.index, prediction, "darkblue", linewidth=1.0)
        ax.scatter(
            y_train.index,
            y_train,
            color="darkorange",
            marker="*",
            s=10,
            label="Training Data",
        )
        ax.scatter(
            y_val.index, y_val, color="lightgreen", s=10, label="Validation Data"
        )
        ax.legend()
        ax.set_ylabel("Groundwater Level")
        ax.set_title(f"Observation Vs Prediction: {name}")

        if error_on:
            train_metrics = metrics[
                ["Train ME", "Train RMSE", "Train MAE", "Train r2"]
            ].to_string(index=True, float_format="{0:.3}".format)
            validation_metrics = metrics[
                ["Validation ME", "Validation RMSE", "Validation MAE", "Validation r2"]
            ].to_string(index=True, float_format="{0:.3}".format)
            ax.text(
                x=0.0,
                y=-0.15,
                s=train_metrics,
                fontsize=12,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.text(
                x=0.25,
                y=-0.15,
                s=validation_metrics,
                fontsize=12,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            if save:
                fig.savefig(
                    f"{self.figures_root}/{name}_07_Observation",
                    bbox_inches=extent.expanded(1.2, 1.6),
                )
        elif save:
            fig.savefig(f"{self.figures_root}/{name}_07_Observation")

        if show:
            plt.show()
        else:
            plt.close(fig)
        gc.collect()

    def plot_prediction_vs_test(
        self,
        prediction: pd.Series,
        well_set_original: pd.Series,
        y_test: pd.Series,
        name: str,
        metrics: pd.DataFrame = None,
        error_on: bool = False,
        test: bool = False,
        show: bool = False,
    ) -> None:
        """
        Plot a comparison between predictions and test data.

        :param prediction: The model's predicted values for plotting.
        :type prediction: pd.Series
        :param well_set_original: Training data for plotting.
        :type well_set_original: pd.Series
        :param y_test: Test data for plotting.
        :type y_test: pd.Series
        :param name: The name of the well.
        :type name: str
        :param metrics: DataFrame with optional metrics data, defaults to None.
        :type metrics: pd.DataFrame, optional
        :param error_on: Whether to include error metrics, defaults to False.
        :type error_on: bool, optional
        :param test: Whether to include test metrics, defaults to False.
        :type test: bool, optional
        :param show: Whether to display the plot, defaults to False.
        :type show: bool, optional
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(prediction.index, prediction, "darkblue", linewidth=1.0)
        ax.scatter(
            well_set_original.index,
            well_set_original,
            color="darkorange",
            marker="*",
            s=10,
            label="Training Data",
        )
        ax.scatter(y_test.index, y_test, color="lightgreen", s=10, label="Test Data")
        ax.set_ylabel("Groundwater Level")
        ax.legend()
        ax.set_title(f"Observation Vs Prediction: {name}")
        ax.axvline(dt.datetime(int(self.cut_left), 1, 1), linewidth=0.25)
        ax.axvline(dt.datetime(int(self.cut_right), 1, 1), linewidth=0.25)

        if error_on:
            train_metrics = metrics[
                ["Train ME", "Train RMSE", "Train MAE", "Train r2"]
            ].to_string(index=True, float_format="{0:.3}".format)
            validation_metrics = metrics[
                ["Validation ME", "Validation RMSE", "Validation MAE", "Validation r2"]
            ].to_string(index=True, float_format="{0:.3}".format)
            ax.text(
                x=0.0,
                y=-0.15,
                s=train_metrics,
                fontsize=12,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.text(
                x=0.25,
                y=-0.15,
                s=validation_metrics,
                fontsize=12,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

            if test:
                test_metrics = metrics[
                    ["Test ME", "Test RMSE", "Test MAE", "Test r2"]
                ].to_string(index=True, float_format="{0:.3}".format)
                ax.text(
                    x=0.5,
                    y=-0.15,
                    s=test_metrics,
                    fontsize=12,
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            if test:
                fig.savefig(
                    f"{self.figures_root}/{name}_08_Test",
                    bbox_inches=extent.expanded(1.2, 1.6),
                )
        else:
            fig.savefig(f"{self.figures_root}/{name}_08_Test")

        if show:
            plt.show()
        else:
            plt.close(fig)
        gc.collect()

    def prediction_kfold(
        self,
        prediction: pd.Series,
        well_set_original: pd.Series,
        y_test: pd.Series,
        name: str,
        metrics: pd.DataFrame = None,
        error_on: bool = False,
        test: bool = False,
        show: bool = False,
        plot: bool = False,
    ) -> None:
        """
        Plot predictions with optional training and test data for k-fold analysis.

        :param prediction: Model's predicted values.
        :type prediction: pd.Series
        :param well_set_original: Training data for plotting.
        :type well_set_original: pd.Series
        :param y_test: Test data for plotting.
        :type y_test: pd.Series
        :param name: The name of the well.
        :type name: str
        :param metrics: DataFrame with optional metrics data, defaults to None.
        :type metrics: pd.DataFrame, optional
        :param error_on: Whether to include error metrics, defaults to False.
        :type error_on: bool, optional
        :param test: Whether to include test metrics, defaults to False.
        :type test: bool, optional
        :param show: Whether to display the plot, defaults to False.
        :type show: bool, optional
        :param plot: Whether to create and save the plot, defaults to False.
        :type plot: bool, optional
        """
        if plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(prediction.index, prediction, "darkblue", linewidth=1.0)
            ax.scatter(
                well_set_original.index,
                well_set_original,
                color="darkorange",
                marker="*",
                s=10,
                label="Training Data",
            )
            ax.scatter(
                y_test.index, y_test, color="lightgreen", s=10, label="Test Data"
            )
            ax.set_ylabel("Groundwater Level")
            ax.legend()
            ax.set_title(f"Observation Vs Prediction: {name}")
            ax.axvline(y_test.index[0], linewidth=0.25)
            ax.axvline(y_test.index[-1], linewidth=0.25)

            if error_on:
                train_metrics = metrics[
                    ["Train ME", "Train RMSE", "Train MAE", "Train r2"]
                ].to_string(index=True, float_format="{0:.3}".format)
                validation_metrics = metrics[
                    [
                        "Validation ME",
                        "Validation RMSE",
                        "Validation MAE",
                        "Validation r2",
                    ]
                ].to_string(index=True, float_format="{0:.3}".format)
                ax.text(
                    x=0.0,
                    y=-0.15,
                    s=train_metrics,
                    fontsize=12,
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.text(
                    x=0.25,
                    y=-0.15,
                    s=validation_metrics,
                    fontsize=12,
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

                if test:
                    test_metrics = metrics[
                        ["Test ME", "Test RMSE", "Test MAE", "Test r2"]
                    ].to_string(index=True, float_format="{0:.3}".format)
                    ax.text(
                        x=0.5,
                        y=-0.15,
                        s=test_metrics,
                        fontsize=12,
                        horizontalalignment="left",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )

                extent = ax.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted()
                )
                if test:
                    fig.savefig(
                        f"{self.figures_root}/{name}_08_kfold",
                        bbox_inches=extent.expanded(1.2, 1.6),
                    )
                else:
                    fig.savefig(f"{self.figures_root}/{name}_08_kfold")

                if show:
                    plt.show()
                else:
                    plt.close(fig)
                gc.collect()

    def prediction_vs_test_kfold(
        self,
        prediction: pd.Series,
        well_set_original: pd.Series,
        name: str,
        metrics: pd.DataFrame = None,
        error_on: bool = False,
        test: bool = False,
        show: bool = False,
    ) -> None:
        """
        Plot model predictions vs. training data for k-fold analysis.

        :param prediction: Model's predicted values.
        :type prediction: pd.Series
        :param well_set_original: Training data for plotting.
        :type well_set_original: pd.Series
        :param name: The name of the well.
        :type name: str
        :param metrics: DataFrame with optional metrics data, defaults to None.
        :type metrics: pd.DataFrame, optional
        :param error_on: Whether to include error metrics, defaults to False.
        :type error_on: bool, optional
        :param test: Whether to include test metrics, defaults to False.
        :type test: bool, optional
        :param show: Whether to display the plot, defaults to False.
        :type show: bool, optional
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(prediction.index, prediction, "darkblue", linewidth=1.0)
        ax.scatter(
            well_set_original.index,
            well_set_original,
            color="darkorange",
            marker="*",
            s=10,
            label="Training Data",
        )
        ax.set_ylabel("Groundwater Level")
        ax.legend()
        ax.set_title(f"Observation Vs Prediction: {name}")

        if error_on:
            train_metrics = metrics[
                ["Train ME", "Train RMSE", "Train MAE", "Train r2"]
            ].to_string(index=True, float_format="{0:.3}".format)
            validation_metrics = metrics[
                [
                    "Validation ME",
                    "Validation RMSE",
                    "Validation MAE",
                    "Validation r2",
                ]
            ].to_string(index=True, float_format="{0:.3}".format)
            ax.text(
                x=0.0,
                y=-0.15,
                s=train_metrics,
                fontsize=12,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.text(
                x=0.25,
                y=-0.15,
                s=validation_metrics,
                fontsize=12,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

            if test:
                test_metrics = metrics[
                    ["Test ME", "Test RMSE", "Test MAE", "Test r2"]
                ].to_string(index=True, float_format="{0:.3}".format)
                ax.text(
                    x=0.5,
                    y=-0.15,
                    s=test_metrics,
                    fontsize=12,
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            if test:
                fig.savefig(
                    f"{self.figures_root}/{name}_08_Test_kfold",
                    bbox_inches=extent.expanded(1.2, 1.6),
                )
            else:
                fig.savefig(f"{self.figures_root}/{name}_08_Test_kfold")

            if show:
                plt.show()
            else:
                plt.close(fig)
            gc.collect()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        show: bool = False,
    ) -> None:
        """
        Plot feature importance.

        :param importance_df: DataFrame containing feature importances.
        :type importance_df: pd.DataFrame
        :param show: Whether to display the plot, defaults to False.
        :type show: bool, optional
        """
        # All Data
        importance_df.boxplot(figsize=(20, 10))
        plt.xlabel("Feature")
        plt.ylabel("Relative Importance")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{self.figures_root}/feature_importance_complete")

        if show:
            plt.show()
        else:
            plt.close()
        gc.collect()

        # Calculate Mean and sort
        importance_mean_df = importance_df.mean()
        importance_mean_df = pd.DataFrame(
            importance_mean_df.sort_values(axis=0, ascending=False)
        ).T
        importance_mean = (
            importance_df.transpose()
            .reindex(list(importance_mean_df.columns))
            .transpose()
        )
        importance_mean.iloc[:, :10].boxplot(figsize=(5, 5))
        plt.xticks(rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Relative Importance")
        plt.title("Most Prevalent Features")
        plt.tight_layout()
        plt.savefig(f"{self.figures_root}/feature_importance_upper")

        if show:
            plt.show()
        else:
            plt.close()
        gc.collect()
