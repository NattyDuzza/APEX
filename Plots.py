#plotting functionality
import matplotlib.pyplot as plt

class Plots:
    """Class for handling plotting functionality."""

    def __init__(self, sacc_workspace=None):
        """Initialize the Plots class."""

        self.sacc_workspace = sacc_workspace

    def create_grid_plot(self, subplot_titles, subplot_tracer_combos, measured_data, measured_data_err, modelled_data, residuals=False, cut_positions=None):
        """Create a grid plot with specified subplot titles and tracer combinations.
        
        Parameters:
        measured_data: [[ells], [c_ells]]
        modelled_data: [[ells], [c_ells], [mask]]
        """
        # Implementation of grid plot creation
        fig = plt.figure(figsize=(10,15))

        outer_grid = fig.add_gridspec(len(subplot_tracer_combos), len(subplot_titles), hspace=0.4, wspace=0.4)


        if residuals:
            
            '''
            mask = modelled_data[2]

            residual_values = []
            residual_ells = []
            for i in range(len(measured_data[1])):
                residual_val = (measured_data[1][i][mask[i]] - modelled_data[1][i]) / measured_data_err[i][mask[i]]
                residual_values.append(residual_val)
                residual_ells.append(measured_data[0][i][mask[i]])
            '''

            for i in range(len(subplot_tracer_combos)):

                row = i % 4
                col = i // 4
                
                inner_grid = outer_grid[row, col].subgridspec(2, 1, height_ratios=[3, 1])
                ax_main = fig.add_subplot(inner_grid[0])
                ax_res = fig.add_subplot(inner_grid[1], sharex=ax_main)
                if cut_positions is not None:
                    ax_main.axvspan(0, cut_positions[0], color='gray', alpha=0.5)
                    ax_main.axvspan(cut_positions[1], max(measured_data[0][i]), color='gray', alpha=0.5)


                ax_main.errorbar(measured_data[0][i], measured_data[1][i], yerr=measured_data_err[i], fmt='k.')

                ax_main.plot(modelled_data[0][i], modelled_data[1][i])

                ax_main.set_yscale('log')
                ax_main.set_xscale('log')

                '''
                ax_res.axhline(0, color='k', linestyle='--')
                ax_res.errorbar(residual_ells, residual_values, yerr=1, fmt='k.')

                ax_res.axvspan(0, cut_positions[0], color='gray', alpha=0.5)
                ax_res.axvspan(cut_positions[1], max(measured_data[i][0]), color='gray', alpha=0.5)
                '''
            fig.tight_layout()
            plt.show()


