import numpy as np
import matplotlib.pyplot as plt

class AirfoilGeometry:
    """
    A utility class to generate and visualize airfoil geometries, 
    specifically NACA 4-digit series and Bezier-based curves.
    """

    def __init__(self, n_points: int = 100):
        """
        Initializes the geometry generator with a specific number of points.
        Uses cosine spacing to cluster points at the leading and trailing edges.
        """
        self.n_points = n_points
        # Cosine spacing for better solver convergence (higher density at LE and TE)
        beta = np.linspace(0, np.pi, self.n_points)
        self.x = 0.5 * (1 - np.cos(beta))

    def generate_naca4(self, m_int: int, p_int: int, t_int: int):
        """
        Wrapper to generate NACA 4-digit coordinates using integer inputs.
        
        Args:
            m_int (int): Maximum camber (1st digit, e.g., 2 for 2%)
            p_int (int): Position of max camber (2nd digit, e.g., 4 for 40%)
            t_int (int): Maximum thickness (3rd & 4th digits, e.g., 12 for 12%)
        
        Returns:
            Coordinates from the internal solver.
        """
        # Convert integer digits to physical decimal values
        m = m_int / 100.0
        p = p_int / 10.0
        t = t_int / 100.0
        
        return self._generate_naca4_coordinates(m, p, t)
    
    def _generate_naca4_coordinates(self, m: float, p: float, t: float):
        """
        Core mathematical implementation of the NACA 4-digit airfoil equations.
        
        :param m: Max camber decimal (e.g. 0.02)
        :param p: Position of max camber decimal (e.g. 0.4)
        :param t: Max thickness decimal (e.g. 0.12)
        :return: Tuple of (x_upper, y_upper, x_lower, y_lower)
        """
        # 1. Thickness distribution (yt) calculation
        # Standard constants for the NACA thickness profile
        a0, a1, a2, a3, a4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1015
        yt = 5 * t * (a0*np.sqrt(self.x) + a1*self.x + a2*self.x**2 + a3*self.x**3 + a4*self.x**4)

        # 2. Mean Camber Line (yc) and local Slope (dyc/dx)
        yc = np.zeros_like(self.x)
        dyc_dx = np.zeros_like(self.x)
        
        # Avoid division by zero if p=0 (symmetric airfoil)
        p = max(p, 0.001)

        mask_front = self.x <= p
        mask_back = self.x > p

        # Calculation for the part forward of maximum camber
        yc[mask_front] = (m / p**2) * (2*p*self.x[mask_front] - self.x[mask_front]**2)
        dyc_dx[mask_front] = (2*m / p**2) * (p - self.x[mask_front])

        # Calculation for the part aft of maximum camber
        yc[mask_back] = (m / (1-p)**2) * ((1-2*p) + 2*p*self.x[mask_back] - self.x[mask_back]**2)
        dyc_dx[mask_back] = (2*m / (1-p)**2) * (p - self.x[mask_back])

        # 3. Combine Thickness and Camber perpendicular to the mean line
        theta = np.arctan(dyc_dx)
        
        x_up = self.x - yt * np.sin(theta)
        y_up = yc + yt * np.cos(theta)
        
        x_lo = self.x + yt * np.sin(theta)
        y_lo = yc - yt * np.cos(theta)

        return x_up, y_up, x_lo, y_lo

    def get_naca_string(self, m: int, p: int, t: int) -> str:
        """Returns the formatted NACA string (e.g., 2, 4, 12 -> '2412')."""
        return f"{m}{p}{t:02d}"

    def generate_bezier(self, top_control_points, bottom_control_points):
        """
        Generates airfoil coordinates using Bezier curves for custom shapes.
        
        Args:
            top_control_points: List of (x, y) tuples for the upper surface.
            bottom_control_points: List of (x, y) tuples for the lower surface.
        """
        def get_bezier_point(t_val, points):
            n = len(points) - 1
            res = np.zeros(2)
            for i, p in enumerate(points):
                # Bernstein polynomial basis function
                coeff = self._combination(n, i) * (1 - t_val)**(n - i) * t_val**i
                res += coeff * np.array(p)
            return res

        # Generate points along the parameter space [0, 1]
        t_space = np.linspace(0, 1, self.n_points)
        upper = np.array([get_bezier_point(t, top_control_points) for t in t_space])
        lower = np.array([get_bezier_point(t, bottom_control_points) for t in t_space])
        
        return upper[:,0], upper[:,1], lower[:,0], lower[:,1]

    def _combination(self, n, k):
        """Math helper for binomial coefficients used in Bezier curves."""
        from math import comb
        return comb(n, k)
    
    def get_coords_for_solver(self, xu, yu, xl, yl):
        """
        Formats coordinates for aerodynamic solvers (TE -> LE -> TE).
        Ensures the leading edge point is not duplicated.
        """
        # 1. Reverse the upper surface (Start at TE, go to LE)
        # 2. Skip the first point of the lower surface to avoid duplicating the LE (0,0)
        x_combined = np.concatenate([xu[::-1], xl[1:]])
        y_combined = np.concatenate([yu[::-1], yl[1:]])
        
        return x_combined, y_combined
    
    def plot_naca4(self, m: int, p: int, t: int):
        """
        Generates and plots the airfoil with clear visual distinctions 
        between the upper and lower surfaces and point distribution.
        """
        xu, yu, xl, yl = self.generate_naca4(m, p, t)
        naca_code = self.get_naca_string(m, p, t)

        plt.figure(figsize=(12, 4))
        plt.plot(xu, yu, 'b-', label='Upper Surface (Extrados)')
        plt.plot(xl, yl, 'r-', label='Lower Surface (Intrados)')

        # Scatter plot to visualize the density of Cosine Spacing
        plt.scatter(xu, yu, s=10, color='blue', alpha=0.5)
        plt.scatter(xl, yl, s=10, color='red', alpha=0.5)

        plt.title(f"Visual Check: NACA {naca_code}")
        plt.xlabel("x/c")
        plt.ylabel("y/c")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal') # Critical to maintain geometric proportions
        plt.show()

    def plot_bezier(self, top_pts, bottom_pts):
        """
        Generates and plots the airfoil with clear visual distinctions 
        between the upper and lower surfaces and point distribution.
        """
        xu, yu, xl, yl = self.generate_bezier(top_pts, bottom_pts)

        plt.figure(figsize=(12, 4))
        plt.plot(xu, yu, 'b-', label='Upper Surface (Extrados)')
        plt.plot(xl, yl, 'r-', label='Lower Surface (Intrados)')

        # Scatter plot to visualize the density of Cosine Spacing
        plt.scatter(xu, yu, s=10, color='blue', alpha=0.5)
        plt.scatter(xl, yl, s=10, color='red', alpha=0.5)

        plt.title(f"Visual Check: Bezier Curve")
        plt.xlabel("x/c")
        plt.ylabel("y/c")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal') # Critical to maintain geometric proportions
        plt.show()