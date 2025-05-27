import numpy as np

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

        area_v = np.trapz(dA, self.v, axis=0)
        area = np.trapz(area_v, self.u)
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

    def export_obj(self, filename="mobius_strip.obj"):
        with open(filename, 'w') as f:
            # Write vertices
            for i in range(self.n):
                for j in range(self.n):
                    f.write(f"v {self.X[i][j]} {self.Y[i][j]} {self.Z[i][j]}\n")

            # Write faces (simple quad mesh)
            for i in range(self.n - 1):
                for j in range(self.n - 1):
                    idx = lambda a, b: a * self.n + b + 1
                    v1 = idx(i, j)
                    v2 = idx(i, j + 1)
                    v3 = idx(i + 1, j + 1)
                    v4 = idx(i + 1, j)
                    f.write(f"f {v1} {v2} {v3} {v4}\n")


if __name__ == "__main__":
    try:
        R = float(input())
        w = float(input())
        n = int(input())

        mobius = MobiusStrip(R, w, n)
        area = mobius.compute_surface_area()
        edge_length = mobius.compute_edge_length()

        print(f"\nSurface Area ≈ {area:.4f}")
        print(f"Edge Length ≈ {edge_length:.4f}")
        mobius.export_obj("mobius_strip.obj")
        print("Möbius strip exported as 'mobius_strip.obj'")

    except Exception as e:
        print(f"Error: {e}")
