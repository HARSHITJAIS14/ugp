import os
import glob
import matplotlib.pyplot as plt

def parse_raspa_output(filepath):
    """
    Parses a RASPA .data output file to extract the external pressure and absolute loading.
    
    Returns:
        tuple: (pressure_bar, absolute_loading) or (None, None) if not found.
    """
    pressure_bar = None
    absolute_loading = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Extract External Pressure
            # e.g.: External Pressure: 100000 [-]
            if line.startswith("External Pressure:"):
                try:
                    parts = line.split()
                    pressure_pa = float(parts[2])
                    pressure_bar = pressure_pa / 100000.0
                except (IndexError, ValueError):
                    pass
            
            # Extract Absolute Loading
            # e.g.: Average loading absolute [mol/kg framework]      12.01202   +/- 0.04353
            elif "Average loading absolute [mol/kg framework]" in line:
                try:
                    parts = line.split()
                    # We want the numeric value. Typically it is index 5 or so. Let's find it.
                    if 'framework]' in parts:
                        idx = parts.index('framework]')
                        absolute_loading = float(parts[idx + 1])
                except (IndexError, ValueError):
                    pass
            
    return pressure_bar, absolute_loading

def generate_isotherm_plot(job_dir, plot_dir):
    """
    Scans a directory for .data files, parses them, and generates an isotherm plot.
    
    Args:
        job_dir (str): Path to the directory containing .data files for a specific job.
        plot_dir (str): Path to save the output plot PNG.
    """
    data_files = glob.glob(os.path.join(job_dir, "*.data"))
    if not data_files:
        print(f"⚠ No .data files found in {job_dir}")
        return None

    results = []
    for filepath in data_files:
        p, l = parse_raspa_output(filepath)
        if p is not None and l is not None:
            results.append((p, l))

    if not results:
        print("⚠ Could not extract valid pressure/loading data from the files.")
        return None

    # Sort by pressure
    results.sort(key=lambda x: x[0])
    
    pressures = [r[0] for r in results]
    loadings = [r[1] for r in results]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(pressures, loadings, marker='o', linestyle='-', color='b', mfc='w')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Pressure (bar)', fontsize=12)
    plt.ylabel('Absolute Loading (mol/kg)', fontsize=12)
    plt.title('CO2 Adsorption Isotherm', fontsize=14)
    
    plt.tight_layout()

    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    
    # Extract robust job name from job_dir (e.g., job_12345 from Output/job_12345)
    job_name = os.path.basename(os.path.normpath(job_dir))
    if not job_name or job_name == "Output":
        job_name = "isotherm"
        
    output_filepath = os.path.join(plot_dir, f"isotherm_{job_name}.png")
    
    plt.savefig(output_filepath, dpi=300)
    plt.close()
    
    print(f"✔ Isotherm plot generated successfully: {output_filepath}")
    return output_filepath

if __name__ == "__main__":
    # Test stub
    import sys
    if len(sys.argv) > 2:
        generate_isotherm_plot(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python plot_isotherm.py <job_dir> <plot_dir>")
