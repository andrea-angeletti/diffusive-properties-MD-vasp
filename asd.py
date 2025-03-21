import numpy as np
import sys
import matplotlib.pyplot as plt

def analyze_data(start_step, end_step, file_path):
    # Load the entire dataset using numpy.load()
    data = np.load(file_path)

    # Calculate the sum of components for the entire dataset
    total_summed_data = np.sum(data, axis=1)
    total_step_numbers = np.arange(1, len(data) + 1)

    # Select data within the specified range for fitting
    selected_data = data[start_step - 1:end_step]
    selected_summed_data = np.sum(selected_data, axis=1)
    selected_step_numbers = np.arange(start_step, end_step + 1)

    # Perform linear fit only on the selected range
    coefficients = np.polyfit(selected_step_numbers, selected_summed_data, 1)
    
    # Calculate diffusion coefficient from the slope
    diffusion_coefficient = coefficients[0] / 600000  # Calculate D as slope/600000

    # Format the diffusion coefficient for the legend with LaTeX
    # Convert diffusion_coefficient to scientific notation with LaTeX formatting
    exponent = np.floor(np.log10(diffusion_coefficient))
    base = diffusion_coefficient / 10**exponent
    diff_coeff_formatted = f"${base:.1f}\\times 10^{{{int(exponent)}}}$ m²/s"

    # Set up the plot
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

    # Plot the total MSD for the entire dataset, converting step numbers to nanoseconds
    plt.plot(total_step_numbers / 1000000, total_summed_data, label='MSD Total', color='black', linestyle='-', linewidth=5)

    # Plot the linear fit only on the selected range, converting step numbers to nanoseconds
    plt.plot(selected_step_numbers / 1000000, np.polyval(coefficients, selected_step_numbers), 
             label=f'Linear Fit (D={diff_coeff_formatted})', color='crimson', linestyle='--', linewidth=3)

    # Colors and labels for the components
    colors = ['cornflowerblue', 'goldenrod', 'forestgreen']  # Colors for x, y, z
    labels = ['x', 'y', 'z']  # Component labels

    # Plotting each component of the MSD for the entire dataset, converting step numbers to nanoseconds
    for i in range(min(3, data.shape[1])):  # Ensure there are at least three components
        component_msd = data[:, i]
        plt.plot(total_step_numbers / 1000000, component_msd, label=f'MSD Component {labels[i]}', color=colors[i], linewidth=4)

    # Set font size and labels with Unicode
    plt.xlabel('Time step (ns)', fontsize=20)  # Adjusting the unit in label to reflect nanoseconds
    plt.ylabel('MSD (Å²)', fontsize=20)  # Using Unicode for angstrom squared
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)

    plt.tight_layout()

    # Save the figure to the current directory
    plt.savefig("msd_analysis_plot.png", dpi=300)  # Save as PNG with high resolution

    plt.show()

    return coefficients, diffusion_coefficient

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script_name.py start_step end_step file_path.npy")
        sys.exit(1)

    start_step = int(sys.argv[1])
    end_step = int(sys.argv[2])
    file_path = sys.argv[3]

    coefficients, diffusion_coefficient = analyze_data(start_step, end_step, file_path)
    print("Coefficients of the linear fit:", coefficients)
    print(f"Calculated diffusion coefficient (D): {diffusion_coefficient} m²/s")

