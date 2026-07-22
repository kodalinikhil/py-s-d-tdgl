import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tdgl
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Mathematical Formulation & Parameters
    epsilon = 0.25 # m_v = 2.0 -> epsilon = 1 / (2 * m_v)
    
    layer = tdgl.Layer(
        coherence_length=1.0,
        london_lambda=5.0, # kappa = 5.0
        thickness=0.1,
        gamma_d=1.0,
        gamma_s=1.0,
        alpha_d=1.0,
        alpha_s=0.5,
        beta_d=1.0,
        beta_s=1.0,
        gamma_1=1.5,
        gamma_2=0.5,
        epsilon=epsilon
    )

    # 2. Grid & Geometry
    width = 12.8
    box = tdgl.geometry.box(width, width)
    film = tdgl.Polygon("film", points=box).resample(400)
    device = tdgl.Device("goncalves_square", layer=layer, film=film)
    device.make_mesh(max_edge_length=0.2)

    # 3. Execution Protocol: Sweep Up and Down
    ha_values_up = [0.0, 0.5, 1.0, 1.5, 2.0]
    ha_values_down = [1.5, 1.0, 0.5, 0.0]
    sweep = ha_values_up + ha_values_down
    
    seed_solution = None
    
    for i, Ha in enumerate(sweep):
        print(f"--- Sweeping to Ha = {Ha} ({i+1}/{len(sweep)}) ---")
        options = tdgl.SolverOptions(
            solve_time=200, 
            dt_init=0.005,
            dt_max=0.05,
            adaptive=True,
            simulate_d_wave=True,
            output_file=f"goncalves_Ha_{Ha}_{i}.h5"
        )
        
        solution = tdgl.solve(
            device,
            options=options,
            applied_vector_potential=tdgl.sources.ConstantField(Ha),
            seed_solution=seed_solution
        )
        seed_solution = solution
        
    # 4. Output: Extract and plot at Ha = 0.0
    print("Simulation complete. Generating plots for trapped flux state...")
    
    tdgl_data = seed_solution.tdgl_data
    mesh = device.mesh
    x, y = mesh.sites[:, 0], mesh.sites[:, 1]
    
    sq_psi_d = np.abs(tdgl_data.psi_d)**2
    sq_psi_s = np.abs(tdgl_data.psi_s)**2
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    sc1 = axs[0, 0].tricontourf(x, y, mesh.elements, sq_psi_d, levels=100, cmap='viridis')
    axs[0, 0].set_title(r'$|\psi_d|^2$ (Vortex Cores)')
    plt.colorbar(sc1, ax=axs[0, 0])
    
    sc2 = axs[0, 1].tricontourf(x, y, mesh.elements, sq_psi_s, levels=100, cmap='plasma')
    axs[0, 1].set_title(r'$|\psi_s|^2$ (Induced Peaks)')
    plt.colorbar(sc2, ax=axs[0, 1])
    
    sc3 = axs[1, 0].tricontourf(x, y, mesh.elements, sq_psi_d - sq_psi_s, levels=100, cmap='coolwarm')
    axs[1, 0].set_title(r'Difference $|\psi_d|^2 - |\psi_s|^2$')
    plt.colorbar(sc3, ax=axs[1, 0])
    
    # Supercurrent quiver
    try:
        J_site = device.mesh.edge_to_site_average(tdgl_data.supercurrent)
        axs[1, 1].quiver(x, y, J_site[:, 0], J_site[:, 1], color='white', alpha=0.5)
    except AttributeError:
        pass
    axs[1, 1].tricontourf(x, y, mesh.elements, sq_psi_d, levels=100, cmap='viridis')
    axs[1, 1].set_title(r'Supercurrent Quiver over $|\psi_d|^2$')
    
    for ax in axs.flat:
        ax.set_aspect('equal')
        
    plt.tight_layout()
    plt.savefig("my_scripts/goncalves_trapped_flux.png", dpi=300)
    print("Plots saved to 'my_scripts/goncalves_trapped_flux.png'.")

if __name__ == "__main__":
    main()
