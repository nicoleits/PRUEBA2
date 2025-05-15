import PySAM.Pvwattsv7 as pv
import PySAM.Lcoefcr as Lcoefcr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_incident_energy(solar_resource_file):
    """
    Calculate annual incident energy from TMY data
    
    Args:
        solar_resource_file (str): Path to the solar resource file
        
    Returns:
        float: Annual incident energy in kWh/m²
    """
    try:
        # Read TMY data with flexible parsing
        df = pd.read_csv(solar_resource_file, 
                        skiprows=2,  # Skip two metadata rows
                        sep=',',     # Comma separator
                        on_bad_lines='skip',  # Skip problematic lines
                        encoding='utf-8')
        
        # Find the GHI column (it might be named differently)
        ghi_columns = [col for col in df.columns if 'GHI' in col.upper()]
        if not ghi_columns:
            raise ValueError(f"No GHI column found in {solar_resource_file}")
        
        ghi_column = ghi_columns[0]
        
        # Calculate incident energy (assuming GHI is in W/m²)
        # Convert hourly data to kWh/m² and sum for annual total
        annual_incident_energy = df[ghi_column].sum() / 1000  # Convert W/m² to kWh/m²
        
        return annual_incident_energy
        
    except Exception as e:
        print(f"Error reading file {solar_resource_file}: {str(e)}")
        return None

def calculate_lcoe(annual_energy, system_capacity_kw, fixed_charge_rate=0.08, project_lifetime=20,
                  capex_pv=1000, fixed_om_cost=50, variable_om_cost=0.01, inverter_lifetime=10,
                  system_losses=14.0):
    """
    Calculate Levelized Cost of Energy (LCOE)
    
    Args:
        annual_energy (float): Annual energy production in kWh
        system_capacity_kw (float): System capacity in kW
        fixed_charge_rate (float): Fixed charge rate (default 8%)
        project_lifetime (int): Project lifetime in years (default 20)
        capex_pv (float): Capital cost per kW (default $1000/kW)
        fixed_om_cost (float): Fixed O&M cost per kW per year (default $50/kW/year)
        variable_om_cost (float): Variable O&M cost per kWh (default $0.01/kWh)
        inverter_lifetime (int): Inverter lifetime in years (default 10)
        system_losses (float): System losses in percentage (default 14%)
        
    Returns:
        float: LCOE in $/kWh
    """
    # Economic parameters
    capital_cost = system_capacity_kw * capex_pv
    
    # Calculate inverter replacement costs
    num_replacements = int(project_lifetime / inverter_lifetime) - 1
    if num_replacements > 0:
        inverter_cost = system_capacity_kw * 200  # $200/kW for inverter replacement
        for i in range(num_replacements):
            replacement_year = (i + 1) * inverter_lifetime
            capital_cost += inverter_cost / (1 + fixed_charge_rate)**replacement_year
    
    fixed_operating_cost = system_capacity_kw * fixed_om_cost
    variable_operating_cost = variable_om_cost
    
    # Apply system losses
    annual_energy = annual_energy * (1 - system_losses/100)
    
    # Calculate present value factors
    discount_rate = fixed_charge_rate
    pv_factor = (1 - (1 + discount_rate)**-project_lifetime) / discount_rate
    
    # Calculate total costs
    total_capital_cost = capital_cost
    total_fixed_om_cost = fixed_operating_cost * pv_factor
    total_variable_om_cost = variable_operating_cost * annual_energy * pv_factor
    
    # Calculate total energy production
    total_energy = annual_energy * project_lifetime
    
    # Calculate LCOE
    lcoe = (total_capital_cost + total_fixed_om_cost + total_variable_om_cost) / total_energy
    
    return lcoe

def run_sensitivity_analysis(annual_energy, system_capacity_kw, base_lcoe, location_name):
    """
    Run sensitivity analysis for different parameters
    
    Args:
        annual_energy (float): Base annual energy production
        system_capacity_kw (float): System capacity
        base_lcoe (float): Base case LCOE
        location_name (str): Name of the location
        
    Returns:
        tuple: (parameters, variations, impacts)
    """
    # Define parameter variations
    parameters = ['FCR', 'CapEx PV', 'Spot Price', 'Inverter Lifetime', 'System Losses']
    
    # Base case parameters
    base_params = {
        'FCR': 0.08,
        'CapEx PV': 1000,
        'Spot Price': 0.01,
        'Inverter Lifetime': 10,
        'System Losses': 14.0
    }
    
    # Define variations for each parameter
    variations = {
        'FCR': {'base': base_params['FCR'], 'low': 0.06, 'high': 0.10},
        'CapEx PV': {'base': base_params['CapEx PV'], 'low': 800, 'high': 1200},
        'Spot Price': {'base': base_params['Spot Price'], 'low': 0.005, 'high': 0.015},
        'Inverter Lifetime': {'base': base_params['Inverter Lifetime'], 'low': 8, 'high': 12},
        'System Losses': {'base': base_params['System Losses'], 'low': 12.0, 'high': 16.0}
    }
    
    impacts = []
    
    print(f"\nSensitivity Analysis for {location_name}:")
    print(f"Base Annual Energy: {annual_energy/1e6:.2f} GWh")
    print(f"Base LCOE: {base_lcoe:.4f} $/kWh")
    
    for param in parameters:
        # Calculate low variation
        if param == 'FCR':
            lcoe_low = calculate_lcoe(annual_energy, system_capacity_kw, 
                                    fixed_charge_rate=variations[param]['low'])
        elif param == 'CapEx PV':
            lcoe_low = calculate_lcoe(annual_energy, system_capacity_kw, 
                                    capex_pv=variations[param]['low'])
        elif param == 'Spot Price':
            lcoe_low = calculate_lcoe(annual_energy, system_capacity_kw, 
                                    variable_om_cost=variations[param]['low'])
        elif param == 'Inverter Lifetime':
            lcoe_low = calculate_lcoe(annual_energy, system_capacity_kw, 
                                    inverter_lifetime=variations[param]['low'])
        else:  # System Losses
            lcoe_low = calculate_lcoe(annual_energy, system_capacity_kw, 
                                    system_losses=variations[param]['low'])
            
        # Calculate high variation
        if param == 'FCR':
            lcoe_high = calculate_lcoe(annual_energy, system_capacity_kw, 
                                     fixed_charge_rate=variations[param]['high'])
        elif param == 'CapEx PV':
            lcoe_high = calculate_lcoe(annual_energy, system_capacity_kw, 
                                     capex_pv=variations[param]['high'])
        elif param == 'Spot Price':
            lcoe_high = calculate_lcoe(annual_energy, system_capacity_kw, 
                                     variable_om_cost=variations[param]['high'])
        elif param == 'Inverter Lifetime':
            lcoe_high = calculate_lcoe(annual_energy, system_capacity_kw, 
                                     inverter_lifetime=variations[param]['high'])
        else:  # System Losses
            lcoe_high = calculate_lcoe(annual_energy, system_capacity_kw, 
                                     system_losses=variations[param]['high'])
        
        # Calculate impact
        impact_low = (lcoe_low - base_lcoe) / base_lcoe * 100
        impact_high = (lcoe_high - base_lcoe) / base_lcoe * 100
        impacts.append((impact_low, impact_high))
        
        print(f"\n{param}:")
        print(f"  Low variation: {variations[param]['low']} -> LCOE: {lcoe_low:.4f} $/kWh (Impact: {impact_low:.1f}%)")
        print(f"  High variation: {variations[param]['high']} -> LCOE: {lcoe_high:.4f} $/kWh (Impact: {impact_high:.1f}%)")
    
    return parameters, variations, impacts

def plot_tornado(parameters, impacts, location_name):
    """
    Create tornado plot for sensitivity analysis
    
    Args:
        parameters (list): List of parameter names
        impacts (list): List of (low, high) impact tuples
        location_name (str): Name of the location
    """
    # Sort parameters by impact range
    sorted_indices = np.argsort([abs(high - low) for low, high in impacts])
    sorted_params = [parameters[i] for i in sorted_indices]
    sorted_impacts = [impacts[i] for i in sorted_indices]
    
    # Create tornado plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    y_pos = np.arange(len(sorted_params))
    width = 0.35
    
    # Plot low variations
    ax.barh(y_pos - width/2, [low for low, _ in sorted_impacts], 
            width, color='red', label='Low Variation')
    
    # Plot high variations
    ax.barh(y_pos + width/2, [high for _, high in sorted_impacts], 
            width, color='green', label='High Variation')
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_params)
    ax.set_xlabel('Impact on LCOE (%)')
    ax.set_title(f'Sensitivity Analysis - {location_name}')
    ax.legend()
    ax.grid(True, axis='x')
    
    # Add value labels
    for i, (low, high) in enumerate(sorted_impacts):
        ax.text(low, i - width/2, f'{low:.1f}%', ha='right', va='center')
        ax.text(high, i + width/2, f'{high:.1f}%', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(f'tornado_analysis_{location_name.lower()}.png')
    plt.close()

def simulate_pv_plant(solar_resource_file, system_capacity_kw, location_name):
    """
    Simulate a PV plant using PVWatts model
    
    Args:
        solar_resource_file (str): Path to the solar resource file
        system_capacity_kw (float): System capacity in kW
        location_name (str): Name of the location for labeling
        
    Returns:
        tuple: (annual_energy, lcoe, incident_energy)
    """
    # Calculate incident energy
    incident_energy = calculate_incident_energy(solar_resource_file)
    if incident_energy is None:
        print(f"Warning: Could not calculate incident energy for {location_name}")
        incident_energy = 0
    
    # Create PVWatts model
    pv_model = pv.new()
    pv_model.SolarResource.solar_resource_file = solar_resource_file

    # Configure system parameters
    pv_model.SystemDesign.system_capacity = system_capacity_kw
    pv_model.SystemDesign.dc_ac_ratio = 1.2
    pv_model.SystemDesign.array_type = 1  # Fixed tilt
    pv_model.SystemDesign.azimuth = 180   # South facing
    pv_model.SystemDesign.tilt = 20       # 20 degree tilt
    pv_model.SystemDesign.gcr = 0.4       # Ground coverage ratio
    pv_model.SystemDesign.inv_eff = 96    # Inverter efficiency
    pv_model.SystemDesign.losses = 14.0   # System losses

    # Run PVWatts simulation
    pv_model.execute()
    annual_energy = pv_model.Outputs.annual_energy

    # Calculate base case LCOE
    base_lcoe = calculate_lcoe(annual_energy, system_capacity_kw, fixed_charge_rate=0.08, project_lifetime=20)
    
    # Run sensitivity analysis
    parameters, variations, impacts = run_sensitivity_analysis(annual_energy, system_capacity_kw, base_lcoe, location_name)
    
    # Create tornado plot
    plot_tornado(parameters, impacts, location_name)

    return annual_energy, base_lcoe, incident_energy

def plot_combined_tornado(all_results):
    """
    Create a combined tornado plot for all locations
    
    Args:
        all_results (dict): Dictionary containing results for each location
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define parameters and their positions
    parameters = ['FCR', 'CapEx PV', 'Spot Price', 'Inverter Lifetime', 'System Losses']
    y_pos = np.arange(len(parameters))
    width = 0.25  # Width of bars
    
    # Colors for each location
    colors = {
        'Calama': 'red',
        'Salvador': 'green',
        'Vallenar': 'blue'
    }
    
    # Plot bars for each location
    for i, location in enumerate(all_results.keys()):
        impacts = all_results[location]['impacts']
        # Sort impacts by absolute range
        sorted_indices = np.argsort([abs(high - low) for low, high in impacts])
        sorted_impacts = [impacts[i] for i in sorted_indices]
        
        # Calculate position offset
        offset = (i - 1) * width
        
        # Plot low variations
        low_bars = ax.barh(y_pos + offset, [low for low, _ in sorted_impacts], 
                width, color=colors[location], alpha=0.6, 
                label=f'{location} (Low)')
        
        # Plot high variations
        high_bars = ax.barh(y_pos + offset, [high for _, high in sorted_impacts], 
                width, color=colors[location], alpha=0.3, 
                label=f'{location} (High)')
        
        # Add value labels for low variations
        for j, bar in enumerate(low_bars):
            width_bar = bar.get_width()
            ax.text(width_bar, bar.get_y() + bar.get_height()/2,
                   f'{width_bar:.1f}%',
                   ha='right', va='center', fontsize=8)
        
        # Add value labels for high variations
        for j, bar in enumerate(high_bars):
            width_bar = bar.get_width()
            ax.text(width_bar, bar.get_y() + bar.get_height()/2,
                   f'{width_bar:.1f}%',
                   ha='left', va='center', fontsize=8)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(parameters)
    ax.set_xlabel('Impact on LCOE (%)')
    ax.set_title('Sensitivity Analysis - All Locations')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    # Add grid
    ax.grid(True, axis='x')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig('tornado_analysis_combined.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # System capacity (50MW = 50,000 kW)
    system_capacity_kw = 50000

    # Define locations and their solar resource files
    locations = [
        {
            "name": "Calama",
            "solar_resource": "/home/nicole/UA/prueba2/calama_TMY_limpio_originales.csv"
        },
        {
            "name": "Salvador", 
            "solar_resource": "/home/nicole/UA/prueba2/salvador_TMY_limpio_originales.csv"
        },
        {
            "name": "Vallenar",
            "solar_resource": "/home/nicole/UA/prueba2/vallenar_TMY_limpio_originales.csv"
        }
    ]

    # Store results
    results = []
    all_results = {}

    # Simulate for each location
    for loc in locations:
        print(f"\nProcessing {loc['name']}...")
        annual_energy, lcoe, incident_energy = simulate_pv_plant(
            loc["solar_resource"],
            system_capacity_kw,
            loc["name"]
        )
        
        # Store results for combined analysis
        parameters, variations, impacts = run_sensitivity_analysis(
            annual_energy, system_capacity_kw, lcoe, loc["name"]
        )
        all_results[loc["name"]] = {
            "annual_energy": annual_energy,
            "lcoe": lcoe,
            "impacts": impacts
        }
        
        results.append({
            "Location": loc["name"],
            "Annual Energy (GWh)": annual_energy / 1e6,  # Convert to GWh
            "LCOE ($/kWh)": lcoe,
            "Incident Energy (kWh/m²)": incident_energy
        })

    # Create results DataFrame
    df_results = pd.DataFrame(results)
    print("\nSimulation Results:")
    print(df_results.to_string(index=False))

    # Save results to CSV
    df_results.to_csv("pv_simulation_results.csv", index=False)
    print("\nResults saved to pv_simulation_results.csv")

    # Create bar plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Annual Energy plot
    ax1.bar(df_results["Location"], df_results["Annual Energy (GWh)"])
    ax1.set_xlabel("Location")
    ax1.set_ylabel("Annual Energy (GWh)")
    ax1.set_title("Annual Energy Production by Location")
    ax1.grid(True)

    # LCOE plot
    ax2.bar(df_results["Location"], df_results["LCOE ($/kWh)"])
    ax2.set_xlabel("Location")
    ax2.set_ylabel("LCOE ($/kWh)")
    ax2.set_title("Levelized Cost of Energy (20-year, 8% FCR)")
    ax2.grid(True)

    # Incident Energy plot
    ax3.bar(df_results["Location"], df_results["Incident Energy (kWh/m²)"])
    ax3.set_xlabel("Location")
    ax3.set_ylabel("Incident Energy (kWh/m²)")
    ax3.set_title("Annual Incident Energy by Location")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("pv_simulation_results.png")
    plt.close()
    
    # Create combined tornado plot
    plot_combined_tornado(all_results)
    print("\nCombined tornado plot saved as 'tornado_analysis_combined.png'")

if __name__ == "__main__":
    main() 