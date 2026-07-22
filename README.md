# py(s+d)TDGL

## Motivation
`py(s+d)TDGL` solves a 2D generalized time-dependent Ginzburg-Landau (TDGL) equation, enabling simulations of vortex and phase dynamics in thin film superconducting devices. It extends the standard 2D TDGL equation by incorporating additional degrees of freedom associated with multiple superconducting order parameters, such as the d-wave and s-wave order parameters.

## Install `py(s+d)TDGL`

At the moment, to install py(s+d)TDGL, you can clone the repository and install it in editable mode.

## About `py(s+d)TDGL`

N.B. There is still much testing to be done; this project is in its early stages.

py(s+d)TDGL is based on the following paper:
Gonçalves, W. C., Sardella, E., Becerra, V. F., Milošević, M. V., & Peeters, F. M. (2014). Numerical solution of the time dependent Ginzburg-Landau equations for mixed (d + s)-wave superconductors. Journal of Mathematical Physics, 55, 041501. https://doi.org/10.1063/1.4870874

Unlike it, we do not set the electric potential to zero and we solve the Poisson equation, as that was how base pyTDGL was implemented.
Now, there are two complex order parameters, and we can model unconventional superconductors, such as high-Tc cuprates, iron pnictides, and heavy-fermion superconductors. 
Like pyTDGL, this is accurate at temperatures near Tc and for superconductors in the dirty limit. A new issue that arises for the phenomenological description of s+d order parameters is that if relaxation times of the d-wave and s-wave components are vastly different, it may not capture some of the microscopic physics. See: 
Zhu, J.-X., Kim, W., Ting, C. S., & Hu, C.-R. (1999). Time-dependent Ginzburg-Landau equations for mixed d- and s-wave superconductors. Physical Review B, 59(17), 11527–11534. https://doi.org/10.1103/PhysRevB.59.11527

As such, this program is most effective if all we care about is the static end state (equilibrium) and not the dynamics.
To revert to the single-band s-wave case, set simulate_d_wave=False in the solver parameters.

### Authors

- Nikhil Kodali

### Citing `py(s+d)TDGL`

Cite the github repository.

### Acknowledgments

This is forked off of Logan Bishop-Van Horn's pyTDGL repository: https://github.com/loganbvh/py-tdgl. His work, and the work of all who have contributed directly or indirectly to pyTDGL is acknowledged here.

