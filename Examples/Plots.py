#plotting functionality
import matplotlib.pyplot as plt

class Plots:
    """Class for handling plotting functionality."""

    def __init__(self, sacc_workspace=None):
        """Initialize the Plots class."""

        self.sacc_workspace = sacc_workspace

    def create_grid_plot(self, subplot_titles, subplot_tracer_combos, measured_data, measured_data_err, modelled_data, residuals=False, cut_positions=(None, None), full_ells=False, variable_cuts=False):
        """Create a grid plot with specified subplot titles and tracer combinations.
        
        Parameters:
        measured_data: [[ells], [c_ells]]
        modelled_data: [[ells], [c_ells], [mask]]
        """
        # Implementation of grid plot creation
        fig = plt.figure(figsize=(15,40))

        outer_grid = fig.add_gridspec(len(subplot_tracer_combos), len(subplot_titles), hspace=0.3, wspace=0.1)

        masks = modelled_data[2]
        

        for i in range(len(subplot_tracer_combos)):

            row = i % 4
            col = i // 4
            
            inner_grid = outer_grid[row, col].subgridspec(2, 1, height_ratios=[4, 1], hspace=0)
            

            ax_main = fig.add_subplot(inner_grid[0])

            if row == 0 and subplot_titles[col] is not None:
                ax_main.set_title(subplot_titles[col], fontsize=16, pad=20)

            if variable_cuts:
                cut_positions[1] = modelled_data[0][i].max()

                cut_positions[0] = modelled_data[0][i].min()

            print(max(measured_data[0][i]), cut_positions[1])

            ax_main.axvspan(0, cut_positions[0], color='gray', alpha=0.5)
            ax_main.axvspan(cut_positions[1], max(measured_data[0][i]), color='gray', alpha=0.5)
            
            #set ax limits
            ax_main.set_xlim(min(0.8*measured_data[0][i]), max(measured_data[0][i])*1.05)


            

            negative_data_mask = measured_data[1][i] < 0
            positive_data_mask = measured_data[1][i] >=0 
            
            if len(measured_data[0][i][positive_data_mask]) > 0:
                ax_main.errorbar(measured_data[0][i][positive_data_mask], measured_data[1][i][positive_data_mask], yerr=measured_data_err[i][positive_data_mask], fmt='k.', zorder=5, label='Measured')
            
            if len(measured_data[0][i][negative_data_mask]) > 0:
                print(measured_data[1][i][negative_data_mask]*-1)
                ax_main.plot(measured_data[0][i][negative_data_mask], measured_data[1][i][negative_data_mask]*-1, 'o', zorder=55, label='Negative Measured', markersize=2, markerfacecolor='none', markeredgecolor='k')


            # Plot modelled data in modelled range
            if full_ells:
                ax_main.plot(modelled_data[0][i], modelled_data[1][i][masks[i]], zorder=10, label='Modelled', color='C1')

                # Plot modelled data outside of modelled range with dots
                ax_main.plot(measured_data[0][i][~masks[i]], modelled_data[1][i][~masks[i]], 'k.', zorder=0, markersize=2,  color='C1')
            
            else:
                ax_main.plot(modelled_data[0][i], modelled_data[1][i], zorder=10, label='Modelled', color='C1')


            ax_main.set_yscale('log')
            ax_main.set_xscale('log')

            #only include bin label on right most column
            if col == len(subplot_titles) - 1:
                ax_main.text(0.95, 0.95, f'Bin {row + 1}',
                        transform=ax_main.transAxes, 
                        fontsize=14,
                        ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', lw=1.0))

            #add titles to each subplot, e.g. each tracer combination 'HSC0 x CIB' or 'HSC0 x HSC0'
            #ax_main.set_title(f'{subplot_tracer_combos[i][0]} x {subplot_tracer_combos[i][1]}')

            #Only label the y-axis of the first column row
            if i < len(subplot_tracer_combos) // 2:
                ax_main.set_ylabel(r'$C_\ell$', fontsize=15)
            if i == (len(subplot_tracer_combos) / 2) -1 or i == (len(subplot_tracer_combos) - 1):
                if residuals == False:
                    ax_main.set_xlabel(r'$\ell$', fontsize=15)
           

            #ax_main.legend()

            if residuals:

                ax_res = fig.add_subplot(inner_grid[1], sharex=ax_main)
                
                if i < len(subplot_tracer_combos) // 2:
                    ax_res.set_ylabel(r'$\Delta \ell$', fontsize=15)
                if i == (len(subplot_tracer_combos) / 2) -1 or i == (len(subplot_tracer_combos) - 1):
                    print(i%4)
                    ax_res.set_xlabel(r'$\ell$', fontsize=15)

                # Calculate residuals
                errs_res = measured_data_err[i][masks[i]]
                errs_res[errs_res == 0] = 1e-12  # Avoid division by zero

                residual_values = (measured_data[1][i][masks[i]] - modelled_data[1][i][masks[i]]) / measured_data_err[i][masks[i]]
                residual_ells = measured_data[0][i][masks[i]]

                ax_res.axhline(0, color='k', linestyle='--')
                ax_res.errorbar(residual_ells, residual_values, yerr=1, fmt='k.')

                ax_res.axvspan(0, cut_positions[0], color='gray', alpha=0.5)
                ax_res.axvspan(cut_positions[1], max(measured_data[0][i]), color='gray', alpha=0.5)

                #ax_res.set_ylim(-3, 3)




            '''
            ax_res.axhline(0, color='k', linestyle='--')
            ax_res.errorbar(residual_ells, residual_values, yerr=1, fmt='k.')

            ax_res.axvspan(0, cut_positions[0], color='gray', alpha=0.5)
            ax_res.axvspan(cut_positions[1], max(measured_data[i][0]), color='gray', alpha=0.5)
            '''
            #fig.tight_layout()
        plt.show()


