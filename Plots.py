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
        measured_data: [ells, c_ells, mask]
        """
        # Implementation of grid plot creation
        fig = plt.figure(figsize=(10,15))

        outer_grid = fig.add_gridspec(len(subplot_tracer_combos), len(subplot_titles), hspace=0.4, wspace=0.4)


        if residuals:

            residual_values = (measured_data[1][mask] - modelled_data[1]) / measured_data_err[mask]

            for i in range(len(subplot_tracer_combos)):
                for j in range(len(subplot_titles)):
                    inner_grid = outer_grid[i, j].subgridspec(2, 1, height_ratios=[3, 1])
                    ax_main = fig.add_subplot(inner_grid[0])
                    ax_residuals = fig.add_subplot(inner_grid[1], sharex=ax_main)
                    if cut_positions is not None:
                        ax_main.axvspan(0, cut_positions[0], color='gray', alpha=0.5)
                        ax_main.axvspan(cut_positions[1], max(measured_data[0]), color='gray', alpha=0.5)


                    ax_main.errorbar(measured_data[0], measured_data[1], yerr=measured_data_err, fmt='k.')

                    ax_main.plot(modelled_data[0], modelled_data[1])

                    ax_main.set_yscale('log')
                    ax_main.set_xscale('log')

                    ax_res.axhline(0, color='k', linestyle='--')
                    ax_res.errorbar(measured_data[0], residual_values, yerr=1, fmt='k.')

                    ax_res.axvspan(0, cut_positions[0], color='gray', alpha=0.5)
                    ax_res.axvspan(cut_positions[1], max(measured_data[0]), color='gray', alpha=0.5)

            fig.tight_layout()
            plt.show()


