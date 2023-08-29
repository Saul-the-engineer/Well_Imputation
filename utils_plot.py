import matplotlib.pyplot as plt
import gc


class plot_functions():
    def __init__(self):
        pass

    def rw_plot(self, y, rw, name, save = False, extension = '.png', show=False):
            fig = plt.figure(figsize=(12, 8))
            plt.plot(rw)
            plt.scatter(y.index, y, s=3, c = 'black')
            plt.ylabel('Groundwater Level')
            plt.legend(rw.columns.tolist() + ['Observations'])
            plt.title('Long-Term Trends: ' + str(name))
            if save: plt.savefig(self.figures_root + '/' + str(name) + '_00_RW' + extension)
            if show: plt.show()
            else:
                fig.clf()
                plt.close(fig)
            gc.collect()
        
    def Q_Q_plot(self, Prediction, Observation, name, limit_low = 0, limit_high = 1, extension = '.png', show=False):
        #Plotting Prediction Correlation
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.scatter(Prediction, Observation)
        plt.ylabel('Observation')
        plt.xlabel('Prediction')
        plt.legend(['Prediction', 'Observation'])
        plt.title('Prediction Correlation: ' + name)
        cor_line_x = np.linspace(limit_low, limit_high, 9)
        cor_line_y = cor_line_x
        plt.xlim(limit_low, limit_high)
        plt.ylim(limit_low, limit_high)
        plt.plot(cor_line_x, cor_line_y, color='r')
        ax1.set_aspect('equal', adjustable='box')
        plt.savefig(self.figures_root + '/' + name + '_01_Q_Q' + extension)
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()

    def observeation_vs_prediction_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name, metrics=None, error_on = False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction_X, Prediction_Y, "darkblue")
        # Potential bug # pandas version 1.4 pandas.errors.InvalidIndexError: (slice(None, None, None), None)
        ax.plot(Observation_X, Observation_Y, label= 'Observations', color='darkorange')
        ax.set_ylabel('Groundwater Level')
        ax.legend(['Prediction', 'Observation'])
        ax.set_title('Observation Vs Prediction: ' + name)
        if error_on:
        ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)          
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(self.figures_root  + '/' + name + '_02_Observation', bbox_inches=extent.expanded(1.1, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_02_Observation')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
        
    def residual_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name, show=False):
        date_rng = pd.DataFrame(np.arange(0, len(Observation_X), 1), index = Observation_X)
        data = self.Data_Join(Prediction_Y, Observation_Y).dropna()
        data = self.Data_Join(data, date_rng).dropna()
        data.columns = ['Prediction_Y', 'Observation_Y', 'Spline Index']
        residual = data['Prediction_Y'] - data['Observation_Y']
        spline = interpolate.UnivariateSpline(data['Spline Index'], residual, s=1000000)
        
        fig, axs = plt.subplots(2, 1, figsize = (12,12))
        axs[0].plot(data.index, residual, marker = 'o', linestyle='None', markersize=5, color = "black")
        axs[0].plot(Observation_X, np.zeros(shape = (len(Observation_X), 1)), color = 'royalblue', linewidth= 2.0)
        axs[0].plot(data.index, spline(data['Spline Index']), color = "red", linewidth= 3.0)
        axs[0].legend(['Residual', 'Zero', 'Spline Trend'])
        
        axs[0].set_ylabel('Prediction Residual Error')
        axs[0].set_title('Residual Error: ' + str(name))
        
        axs[1].plot(Observation_X, Observation_Y, marker = 'o', linestyle='None', markersize=5, color = "black")
        axs[1].plot(Observation_X, Observation_Y[name].mean()*np.ones(shape = (len(Observation_X), 1)), color = 'royalblue', linewidth= 2.0)
        axs[1].legend(['Observations', 'Mean'])
        axs[1].set_ylabel('Groundwater Level')
        axs[1].set_title('Groundwater Observations: ' + str(name))
        
        plt.savefig(self.figures_root  + '/' + str(name) + '_03_Residual_Plot')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)  
        gc.collect()

    def observeation_vs_imputation_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name, show=False):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(Prediction_X, Prediction_Y, "darkblue")
        plt.plot(Observation_X, Observation_Y, label= 'Observations', color='darkorange')
        plt.ylabel('Groundwater Level')
        plt.xlabel('Date')
        plt.legend(['Imputed Values', 'Smoothed Observations'])
        plt.title('Observation Vs Imputation: ' + name)
        plt.savefig(self.figures_root  + '/' + name + '_04_Imputation')
        if show: plt.show()
        else: 
            fig.clf()
            plt.close(fig)
        gc.collect()

    def raw_observation_vs_prediction(self, Prediction, Raw, name, Aquifer, metrics=None, error_on = False, test=False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, 'darkblue', label='Prediction', linewidth=1.0)
        ax.scatter(Raw.index, Raw, color='darkorange', marker = '*', s=10, label= 'Observations')
        ax.set_title(Aquifer + ': ' + 'Well: ' + name + ' Predicted Values')
        ax.set_ylabel('Groundwater Level')
        ax.legend(fontsize = 'medium')
        if error_on:
        ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)          
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if test:
            ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        fig.savefig(self.figures_root  + '/' + name + '_05_Prediction_vs_Raw', bbox_inches=extent.expanded(1.3, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_05_Prediction_vs_Raw')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
        
    def raw_observation_vs_filled(self, Prediction, Raw, name, Aquifer, df_spread=None, conf_interval = None, 
                                ci = 1, metrics=None, error_on = False, test=False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, 'darkblue', label='Prediction', linewidth=1.0)
        ax.plot(Raw.index, Raw, color='darkorange', label= 'Observations', linewidth=1.0)
        ax.set_title(Aquifer + ': ' + 'Well: ' + name + ' Imputed Values')
        ax.set_ylabel('Groundwater Level')
        if conf_interval and not df_spread.empty:
            upper = df_spread['mean'] + df_spread['std'] * ci
            lower = df_spread['mean'] - df_spread['std'] * ci
            data_min = min(Prediction.min(), Raw.min())
            data_max = max(Prediction.max(), Raw.max())
            ax.fill_between(df_spread.index, upper, lower, color='blue', alpha=0.2, label=f'Std Dev: {ci}')
            ax.set_ylim(data_min - 5, data_max + 5)
        ax.legend(fontsize = 'medium')

        if error_on:
        ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)          
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if test:
            ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        fig.savefig(self.figures_root  + '/' + name + '_05_Filled_vs_Raw', bbox_inches=extent.expanded(1.3, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_05_Filled_vs_Raw')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
    
    def raw_observation_vs_imputation(self, Prediction, Raw, name, Aquifer, metrics=None, show=False):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(Prediction.index, Prediction, 'darkblue', label='Model', linewidth=1.0)
        plt.scatter(Raw.index, Raw, color='darkorange', marker = '*', s=10, label= 'Observations')
        plt.title(Aquifer + ': ' + 'Well: ' + name + ' Raw vs Model')
        plt.legend(fontsize = 'medium')
        plt.savefig(self.figures_root  + '/' + name + '_06_Imputation_vs_Raw')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()

    def observeation_vs_prediction_scatter_plot(self, Prediction, Y_train, Y_val, name, metrics=None, error_on = False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
        ax.scatter(Y_train.index, Y_train, color='darkorange', marker='*', s=10)
        ax.scatter(Y_val.index, Y_val, color='lightgreen', s=10)  
        ax.legend(['Prediction', 'Training Data', 'Validation Data'])
        ax.set_ylabel('Groundwater Level')
        ax.set_title('Observation Vs Prediction: ' + name)
        if error_on:
        ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)          
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(self.figures_root  + '/' + name + '_07_Observation', bbox_inches=extent.expanded(1.2, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_07_Observation')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
    
    def prediction_vs_test(self, Prediction, Well_set_original, y_test, name, metrics=None, error_on = False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
        ax.scatter(Well_set_original.index, Well_set_original, color='darkorange', marker='*', s=10)
        ax.scatter(y_test.index, y_test, color='lightgreen', s=10)
        ax.set_ylabel('Groundwater Level')
        ax.legend(['Prediction', 'Training Data', 'Test Data'])
        ax.set_title('Observation Vs Prediction: ' + name)
        ax.axvline(dt.datetime(int(self.cut_left), 1, 1), linewidth=0.25)
        ax.axvline(dt.datetime(int(self.cut_right), 1, 1), linewidth=0.25)
        if error_on:
        ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)  
        ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(self.figures_root  + '/' + name + '_08_Test', bbox_inches=extent.expanded(1.2, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_08_Test')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
        
    def prediction_kfold(self, Prediction, Well_set_original, y_test, name, metrics=None, error_on = False, show=False, plot=False):
        if plot == True:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
            ax.scatter(Well_set_original.index, Well_set_original, color='darkorange', marker='*', s=10)
            ax.scatter(y_test.index, y_test, color='lightgreen', s=10)
            ax.set_ylabel('Groundwater Level')
            ax.legend(['Prediction', 'Training Data', 'Test Data'])
            ax.set_title('Observation Vs Prediction: ' + name)
            ax.axvline(y_test.index[0], linewidth=0.25)
            ax.axvline(y_test.index[-1], linewidth=0.25)
            if error_on:
            ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                    fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                    fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)  
            ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                    fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(self.figures_root  + '/' + name + '_08_kfold', bbox_inches=extent.expanded(1.2, 1.6))
            else: fig.savefig(self.figures_root  + '/' + name + '_08_kfold')
            if show: plt.show()
            else:
                fig.clf()
                plt.close(fig)
            gc.collect()
        
    def prediction_vs_test_kfold(self, Prediction, Well_set_original, name, metrics=None, error_on = False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
        ax.scatter(Well_set_original.index, Well_set_original, color='darkorange', marker='*', s=10)
        ax.set_ylabel('Groundwater Level')
        ax.legend(['Prediction', 'Training Data', 'Test Data'])
        ax.set_title('Observation Vs Prediction: ' + name)
        if error_on:
        ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)  
        ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(self.figures_root  + '/' + name + '_08_Test_kfold', bbox_inches=extent.expanded(1.2, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_08_Test_kfold')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)    
        gc.collect()
        
    def Feature_Importance_box_plot(self, importance_df, show=False):
        #All Data       
        importance_df.boxplot(figsize=(20,10))
        plt.xlabel('Feature')
        plt.ylabel('Relative Importance')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(self.figures_root  + '/' + 'Feature_Importance_Complete')
        if show: plt.show()
        else: plt.close()
    
        #Calc Mean and sort
        importance_mean_df = importance_df.mean()
        importance_mean_df = pd.DataFrame(importance_mean_df.sort_values(axis=0, ascending=False)).T
        importance_mean = importance_df.transpose().reindex(list(importance_mean_df.columns)).transpose()
        importance_mean.iloc[:,:10].boxplot(figsize=(5,5))
        plt.xticks(rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Relative Importance')
        plt.title('Most Prevalent Features:')
        plt.tight_layout()
        plt.savefig(self.figures_root  + '/' + 'Feature_Importance_Uppper')
        if show: plt.show()
        else: plt.close()
        
        #Lower
        importance_mean.iloc[:,importance_mean.shape[1]-10:].boxplot(figsize=(5,5))
        plt.xticks(rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Relative Importance')
        plt.title('Least Prevalent Features:')
        plt.tight_layout()
        plt.savefig(self.figures_root  + '/' + 'Feature_Importance_Lower')
        if show: plt.show()
        else: plt.close()
        gc.collect()

    def feature_plot(self, Feature_Data, raw, name, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Feature_Data)
        for i, n in enumerate(Feature_Data.columns[1:]):
            ax.scatter(raw.index, raw[n], color='none', edgecolors='black', s=10)
        legend = Feature_Data.columns.tolist() + ['Observed Measurements']
        ax.legend(legend, loc="lower left", bbox_to_anchor=(0.02, -0.25),
        ncol=2, fancybox=True, shadow=True)
        ax.set_ylabel('Groundwater Level')
        ax.set_title('Well Feature Correlation: ' + str(name))
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(self.figures_root  + '/' + str(name) + '_09_Features', bbox_inches=extent.expanded(1.3, 1.8))
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()

    def Aquifer_Plot(self, imputed_df, show=False):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(imputed_df)
        plt.title('Measured and Interpolated data for all wells')
        plt.savefig(self.figures_root  + '/' + 'Aquifer_Plot')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
            