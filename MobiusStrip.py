import numpy as np
import matplotlib.pyplot as plt


class MobiusStrip:
    def __init__(self, R, w, n):
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.X, self.Y, self.Z = self._generate_mesh()

    def _generate_mesh(self):
        U, V = self.U, self.V
        X = (self.R + (V/2) * np.cos(U / 2)) * np.cos(U)
        Y = (self.R + (V/2) * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)
        return X, Y, Z

    def compute_surface_area(self):
        dU = (2 * np.pi) / (self.n - 1)
        dV = self.w / (self.n - 1)

        Xu = np.gradient(self.X, axis=1) / dU
        Xv = np.gradient(self.X, axis=0) / dV
        Yu = np.gradient(self.Y, axis=1) / dU
        Yv = np.gradient(self.Y, axis=0) / dV
        Zu = np.gradient(self.Z, axis=1) / dU
        Zv = np.gradient(self.Z, axis=0) / dV

        cross_x = Yu * Zv - Zu * Yv
        cross_y = Zu * Xv - Xu * Zv
        cross_z = Xu * Yv - Yu * Xv
        dA = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

        # Double integration using trapezoidal rule
        area_v = np.trapezoid(dA, self.v, axis=0)
        area = np.trapezoid(area_v, self.u)
        return area

    def compute_edge_length(self):
        u_vals = self.u
        v_edge = self.w / 2

        x = (self.R + v_edge * np.cos(u_vals / 2)) * np.cos(u_vals)
        y = (self.R + v_edge * np.cos(u_vals / 2)) * np.sin(u_vals)
        z = v_edge * np.sin(u_vals / 2)

        length = sum(
            np.linalg.norm([x[i+1] - x[i], y[i+1] - y[i], z[i+1] - z[i]])
            for i in range(len(u_vals) - 1)
        )
        return length

    def plot(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, rstride=4, cstride=4,
                        color='lightsteelblue', edgecolor='gray', alpha=0.9)
        ax.set_title("Möbius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    try:
        R = float(input("Enter radius R (e.g., 1.0): "))
        w = float(input("Enter strip width w (e.g., 0.4): "))
        n = int(input("Enter resolution n (e.g., 300): "))

        mobius = MobiusStrip(R, w, n)
        area = mobius.compute_surface_area()
        edge_length = mobius.compute_edge_length()

        print(f"\nSurface Area ≈ {area:.4f}")
        print(f"Edge Length ≈ {edge_length:.4f}\n")
        mobius.plot()

    except Exception as e:
        print(f"Error: {e}")
