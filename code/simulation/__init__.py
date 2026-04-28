"""Vehicle simulation package.

Key modules:
- VehicleModel: JAX two-track vehicle dynamics model (Magic Tire Formula, RK4)
- simulation: data generation pipeline and make_simulator() factory
- noise: calibrated observation and process noise injection
- utils: coordinate transform utilities (ENU, WGS-84, rotation matrices)
"""
