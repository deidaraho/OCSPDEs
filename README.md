# OCSPDEs
Optimal Control Solver to Navier-Stokes Coupled with Heat Transfer Equations

This implementation of an optimal control solver to Navier-Stokes Equation coupled with Heat Transfer Equations described in the paper, 

"Zoned HVAC Control via PDE-Constrained Optimization", ACC 2016.

This model simplifes the HAVC system as a laminar fluid with boudanry velocity control and internal heat control.

Non-linear partial differential equations' solver is in 'model_new2.py',
related linearized partial differential equations' solver is in 'model_new2_lin.py',
'model_new2_fenics' is used to discrize the partial differential equations,
and the geometry is in './geo' folder.