import numpy as np

from funrobo_kinematics.core.trajectory_generator import (
    MultiAxisTrajectoryGenerator,
    MultiSegmentTrajectoryGenerator,
)


class CubicPolynomial:
    """
    Cubic interpolation with position and velocity boundary constraints.
    """

    def __init__(self, ndof=None):
        """
        Initialize the trajectory generator.
        """
        self.ndof = ndof

    def solve(self, q0, qf, qd0, qdf, T):
        """
        Compute cubic polynomial coefficients for each DOF.

        Parameters
        ----------
        q0 : array-like, shape (ndof,)
            Initial positions.
        qf : array-like, shape (ndof,)
            Final positions.
        qd0 : array-like or None, shape (ndof,)
            Initial velocities. If None, assumed zero.
        qdf : array-like or None, shape (ndof,)
            Final velocities. If None, assumed zero.
        T : float
            Total trajectory duration.
        """
        t0, tf = 0, T
        q0 = np.asarray(q0, dtype=float)
        qf = np.asarray(qf, dtype=float)
        qd0 = np.zeros_like(q0) if qd0 is None else np.asarray(qd0, dtype=float)
        qdf = np.zeros_like(q0) if qdf is None else np.asarray(qdf, dtype=float)

        A = np.array(
            [
                [1, t0, t0**2, t0**3],
                [0, 1, 2 * t0, 3 * t0**2],
                [1, tf, tf**2, tf**3],
                [0, 1, 2 * tf, 3 * tf**2],
            ]
        )

        b = np.vstack([q0, qd0, qf, qdf])
        self.coeff = np.linalg.solve(A, b)

    def generate(self, t0=0, tf=0, nsteps=100):
        """
        Generate position, velocity, and acceleration trajectories.

        Parameters
        ----------
        t0 : float
            Start time.
        tf : float
            End time.
        nsteps : int
            Number of time samples.
        """
        t = np.linspace(t0, tf, nsteps)
        X = np.zeros((self.ndof, 3, len(t)))
        for i in range(self.ndof):  # iterate through all DOFs
            c = self.coeff[:, i]

            q = c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3
            qd = c[1] + 2 * c[2] * t + 3 * c[3] * t**2
            qdd = 2 * c[2] + 6 * c[3] * t

            X[i, 0, :] = q  # position
            X[i, 1, :] = qd  # velocity
            X[i, 2, :] = qdd  # acceleration

        return t, X


class QuinticPolynomial:
    """
    Quintic interpolation with position and velocity boundary constraints.
    """

    def __init__(self, ndof=None):
        """
        Initialize the trajectory generator.
        """
        self.ndof = ndof

    def solve(self, q0, qf, qd0, qdf, T):
        """
        Compute quintic polynomial coefficients for each DOF.

        Parameters
        ----------
        q0 : array-like, shape (ndof,)
            Initial positions.
        qf : array-like, shape (ndof,)
            Final positions.
        qd0 : array-like or None, shape (ndof,)
            Initial velocities. If None, assumed zero.
        qdf : array-like or None, shape (ndof,)
            Final velocities. If None, assumed zero.
        T : float
            Total trajectory duration.
        """
        t0, tf = 0, T
        q0 = np.asarray(q0, dtype=float)
        qf = np.asarray(qf, dtype=float)
        qd0 = np.zeros_like(q0) if qd0 is None else np.asarray(qd0, dtype=float)
        qdf = np.zeros_like(q0) if qdf is None else np.asarray(qdf, dtype=float)
        qdd0 = np.zeros_like(q0)
        qddf = np.zeros_like(q0)

        A = np.array(
            [
                [1, t0, t0**2, t0**3, t0**4, t0**5],
                [0, 1, 2 * t0, 3 * t0**2, 4 * t0**3, 5 * t0**4],
                [0, 0, 2, 6 * t0, 12 * t0**2, 20 * t0**3],
                [1, tf, tf**2, tf**3, tf**4, tf**5],
                [0, 1, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
                [0, 0, 2, 6 * tf, 12 * tf**2, 20 * tf**3],
            ]
        )

        b = np.vstack([q0, qd0, qdd0, qf, qdf, qddf])
        self.coeff = np.linalg.solve(A, b)

    def generate(self, t0=0, tf=0, nsteps=100):
        """
        Generate position, velocity, and acceleration trajectories.

        Parameters
        ----------
        t0 : float
            Start time.
        tf : float
            End time.
        nsteps : int
            Number of time samples.
        """
        t = np.linspace(t0, tf, nsteps)
        X = np.zeros((self.ndof, 3, len(t)))
        for i in range(self.ndof):  # iterate through all DOFs
            c = self.coeff[:, i]

            q = c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3 + c[4] * t**4 + c[5] * t**5
            qd = (
                c[1]
                + 2 * c[2] * t
                + 3 * c[3] * t**2
                + 4 * c[4] * t**3
                + 5 * c[5] * t**4
            )
            qdd = 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t**2 + 20 * c[5] * t**3

            X[i, 0, :] = q  # position
            X[i, 1, :] = qd  # velocity
            X[i, 2, :] = qdd  # acceleration

        return t, X
    
class Trapezoidal:
    """
    Trapezoidal interpolation with position and velocity boundary constraints.
    """

    def __init__(self, ndof=None):
        """
        Initialize the trajectory generator.
        """
        self.ndof = ndof
        self.params = {}

    def solve(self, q0, qf, qd0, qdf, T):
        """
        Compute trapezoidal parameters for each DOF.

        Parameters
        ----------
        q0 : array-like, shape (ndof,)
            Initial positions.
        qf : array-like, shape (ndof,)
            Final positions.
        qd0 : array-like or None, shape (ndof,)
            Initial velocities. If None, assumed zero.
        qdf : array-like or None, shape (ndof,)
            Final velocities. If None, assumed zero.
        T : float
            Total trajectory duration.
        """
        t0, tf = 0, T
        q0 = np.asarray(q0, dtype=float)
        qf = np.asarray(qf, dtype=float)
        b1 = (qf - q0) / tf
        b2 = 2 * (qf - q0) / tf
        v_min = np.minimum(b1, b2)
        v_max = np.maximum(b1, b2)

        tb = 0.3 * tf
        vel = (qf - q0) / (tf - tb)
        print(vel)
        vel = np.clip(vel, v_min, v_max)
        print(vel)
        alpha = vel/tb

        self.params = {
            'q0': q0, 'qf': qf, 'vel': vel, 
            'tb': tb, 'alpha': alpha, 'tf': tf
        }

    def generate(self, t0=0, tf=0, nsteps=100):
        """
        Generate position, velocity, and acceleration trajectories.

        Parameters
        ----------
        t0 : float
            Start time.
        tf : float
            End time.
        nsteps : int
            Number of time samples.
        """
        t = np.linspace(t0, tf, nsteps)
        X = np.zeros((self.ndof, 3, len(t)))
        p = self.params

        for i in range(self.ndof):  # iterate through all DOFs
            q0, qf = p['q0'][i], p['qf'][i]
            vel, alpha = p['vel'][i], p['alpha'][i]
            tb, tf_total = p['tb'], p['tf']

            for j, curr_t in enumerate(t):
                if 0 <= curr_t < tb:
                    q = q0 + (alpha * curr_t**2)/2
                    qd = alpha * curr_t
                    qdd = alpha

                elif tb <= curr_t <= (tf_total - tb):
                    q = (qf + q0 - vel*tf_total)/2 + vel* curr_t
                    qd = vel
                    qdd = 0

                else: 
                    q = qf - (alpha * tf_total**2)/2 + alpha*tf_total*curr_t - (alpha*curr_t**2)/2
                    qd = alpha*tf_total - alpha*curr_t
                    qdd = -alpha


                X[i, 0, j] = q  # position
                X[i, 1, j] = qd  # velocity
                X[i, 2, j] = qdd  # acceleration

        return t, X


def main():
    ndof = 2
    method = Trapezoidal(ndof=ndof)
    mode = "joint"

    # --------------------------------------------------------
    # Point-to-point multi-axis trajectory generator
    # --------------------------------------------------------

    # traj = MultiAxisTrajectoryGenerator(method=method, mode=mode, ndof=ndof)

    # traj.solve(q0=-30, qf=60, T=1)
    # traj.generate(nsteps=20)

    # --------------------------------------------------------
    # Via point multi-axis trajectory generator
    # --------------------------------------------------------

    traj = MultiSegmentTrajectoryGenerator(
        method=method,
        mode=mode,
        ndof=ndof,
    )
    via_points = [[-30, 30], [0, 45], [30, 15], [50, -30]]

    traj.solve(via_points, T=2)
    traj.generate(nsteps_per_segment=20)

    # plotter
    traj.plot()


if __name__ == "__main__":
    main()
