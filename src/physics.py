import aerosandbox as asb
import numpy as np

class AeroSolver:
    def __init__(self, altitude=0, velocity=25, chord=1.0):
        """
        :param altitude: meters above sea level (0 = sea level)
        :param velocity: meters per second (m/s)
        :param chord: length of the wing (meters)
        """
        # 1. Get Atmospheric properties at this altitude
        atmo = asb.Atmosphere(altitude=altitude)
        
        # 2. Calculate Reynolds (Speed * Size / Viscosity)
        self.reynolds = (velocity * chord) / atmo.kinematic_viscosity()
        
        # 3. Calculate Mach (Speed / Speed of Sound)
        self.mach = velocity / atmo.speed_of_sound()

    def solve(self, x_coords, y_coords, alpha=5.0):
        coords = np.stack((x_coords, y_coords), axis=1)
        af = asb.Airfoil(name="Custom", coordinates=coords)
        
        # NeuralFoil handles the prediction
        results = af.get_aero_from_neuralfoil(
            alpha=alpha, 
            Re=self.reynolds,
            mach=self.mach
        )
        
        return {
            "cl": np.array(results['CL']).flatten(),
            "cd": np.array(results['CD']).flatten(),
            "success": True
        }