"""
mppi_and_ilqr.py

Self-contained demo of:
 - separated dynamics (f, G, B, dynamics)
 - iLQR (finite-horizon)
 - BaseMPPI (classic PI control)
 - RefinedMPPI (paper-style using H^{-1} G)

Requires: numpy
"""

import numpy as np

# --------------------------
# Dynamics (bicycle demo)
# --------------------------
L = 2.5  # wheelbase (example)

def f(x):
    """Passive drift (state derivative, not dt-multiplied).
    x = [px, py, theta, v]
    """
    px, py, theta, v = x
    beta = 0.0  # slip angle w/o steering baseline
    return np.array([
        v * np.cos(theta + beta),
        v * np.sin(theta + beta),
        0.0,
        -0.2 * v**3
    ])

def G(x):
    """Control-effectiveness matrix (4 x 2) for u = [a, delta].
    This is a simple (approximate) mapping: 'a' affects velocity directly;
    'delta' affects heading (approx v/L).
    """
    px, py, theta, v = x
    # a -> dv/dt
    g_a = np.array([0.0, 0.0, 0.0, 1.0])
    # delta -> dtheta/dt approx v/L (we neglect small nonlinear terms in position derivatives)
    g_delta = np.array([0.0, 0.0, v / L if L != 0 else 0.0, 0.0])
    return np.column_stack([g_a, g_delta])  # shape (4,2)

def B(x):
    """Diffusion matrix (4 x r) mapping Brownian noise to state-space.
    Use r = 4 (full-state independent noise) by default.
    """
    return np.eye(4)

def dynamics(x, u, dt, w=None):
    """Discrete-time stochastic dynamics:
       x_{t+1} = x_t + (f(x) + G(x) u) * dt + B(x) * w * sqrt(dt)
       w should have shape (r,) where r = B.shape[1]. If None, no noise.
    """
    x = np.asarray(x)
    u = np.asarray(u)
    drift = f(x)
    control_contrib = G(x) @ u
    x_next = x + (drift + control_contrib) * dt
    if w is not None:
        x_next = x_next + (B(x) @ np.asarray(w)) * np.sqrt(dt)
    return x_next

# --------------------------
# Helpers: cost, rollouts, jacobians
# --------------------------
def quad_cost(x, u, x_goal=None, Q=None, R=None):
    """Simple running cost function.
       q(x) = (x-x_goal)^T Q (x-x_goal)  ;  0.5 u^T R u
    """
    if Q is None:
        Q = np.diag([1.0, 1.0, 0.1, 0.1])
    if R is None:
        R = np.diag([0.1, 0.1])
    if x_goal is None:
        x_goal = np.zeros_like(x)
    dx = x - x_goal
    return float(dx.T @ Q @ dx + 0.5 * u.T @ R @ u)

def finite_difference_jacobian(func, x, u, eps=1e-5):
    """Return A,B where for discrete-time x_{t+1} = x + (f(x)+G(x)u)dt + ...
       We linearize the map F(x,u) = x_next (without noise) so A = dF/dx, B = dF/du.
       Uses finite differences.
    """
    x = np.asarray(x)
    u = np.asarray(u)
    n = x.size
    m = u.size
    base = func(x, u)
    A = np.zeros((n, n))
    Bmat = np.zeros((n, m))
    # perturb states
    for i in range(n):
        xp = x.copy()
        xp[i] += eps
        Ap = func(xp, u)
        A[:, i] = (Ap - base) / eps
    for j in range(m):
        up = u.copy()
        up[j] += eps
        Bp = func(x, up)
        Bmat[:, j] = (Bp - base) / eps
    return A, Bmat

def discrete_dynamics_map(x, u, dt):
    """Helper mapping F(x,u) = x_next (deterministic, no noise)"""
    return dynamics(x, u, dt, w=None)

# -------------------------
# Cost
# -------------------------
def make_quadratic_cost(x_goal, Q=None, R=None):
    if Q is None:
        Q = np.diag([1.0, 1.0, 0.1, 0.1])
    if R is None:
        R = np.diag([0.1, 0.1])

    def cost_fn(x, u):
        dx = x - x_goal
        return float(dx.T @ Q @ dx + 0.5 * u.T @ R @ u)
    return cost_fn

# --------------------------
# iLQR
# --------------------------
def ilqr(x0, u_init, dt, N, cost_fn, max_iter=50, reg_init=1.0, reg_factor=10.0, tol=1e-4):
    """
    Basic iLQR for finite-horizon deterministic dynamics.
    - x0: initial state
    - u_init: initial control sequence shape (N, m)
    - dt: timestep
    - N: horizon length
    - cost_fn(x,u) -> scalar (running cost). Terminal cost should be applied after forward pass.
    Returns optimized control sequence and state trajectory.
    """
    n = x0.size
    Nseq = u_init.shape[0]
    assert Nseq == N, "u_init must have shape (N, m)"
    m = u_init.shape[1]
    u = u_init.copy()
    reg = reg_init

    def forward_rollout(x0, u_seq):
        xs = np.zeros((N + 1, n))
        xs[0] = x0.copy()
        total = 0.0
        for t in range(N):
            total += cost_fn(xs[t], u_seq[t])
            xs[t + 1] = dynamics(xs[t], u_seq[t], dt, w=None)
        # optional terminal cost
        total += cost_fn(xs[N], np.zeros(m))
        return xs, total

    xs, J = forward_rollout(x0, u)
    for it in range(max_iter):
        # compute linearizations along trajectory
        A_list = []
        B_list = []
        for t in range(N):
            def map_fun(x_, u_):
                return discrete_dynamics_map(x_, u_, dt)
            A, Bmat = finite_difference_jacobian(map_fun, xs[t], u[t])
            A_list.append(A)
            B_list.append(Bmat)

        # Backward pass (LQ approximation)
        # Quadratic approx: value at final step
        V_x = 2 * (xs[N] - xs[0]) @ np.diag([1.0,1.0,0.1,0.1])  # terminal gradient (rough)
        V_x = V_x.reshape(-1)
        V_xx = 2 * np.diag([1.0,1.0,0.1,0.1])

        K_seq = np.zeros((N, m, n))
        k_seq = np.zeros((N, m))
        diverged = False

        for t in reversed(range(N)):
            A = A_list[t]
            Bmat = B_list[t]
            # cost derivatives (quadratic)
            # l = q(x,u)
            Q_x = 2 * (xs[t] - xs[0]) @ np.diag([1.0,1.0,0.1,0.1])
            Q_x = Q_x.reshape(-1)
            Q_u = (u[t] * 0.1)  # derivative of 0.5 u^T R u with R diag(0.1,0.1)
            Q_xx = 2 * np.diag([1.0,1.0,0.1,0.1])
            Q_ux = np.zeros((m, n))
            Q_uu = np.diag([0.1, 0.1]) + reg * np.eye(m)

            # Q function expansions
            # Here we do a simple LQ expansion: quu = Q_uu + B^T V_xx B
            qu = Q_u + Bmat.T @ V_x
            qx = Q_x + A.T @ V_x
            quu = Q_uu + Bmat.T @ V_xx @ Bmat
            qxx = Q_xx + A.T @ V_xx @ A
            qux = Q_ux + Bmat.T @ V_xx @ A

            # regularize
            try:
                Luu = np.linalg.cholesky(quu)
                inv_quu = np.linalg.inv(quu)
            except np.linalg.LinAlgError:
                diverged = True
                break

            k = -inv_quu @ qu
            K = -inv_quu @ qux
            # update value function
            V_x = qx + K.T @ quu @ k + K.T @ qu + qux.T @ k
            V_xx = qxx + K.T @ quu @ K + K.T @ qux + qux.T @ K
            # ensure symmetry
            V_xx = 0.5 * (V_xx + V_xx.T)
            K_seq[t] = K
            k_seq[t] = k

        if diverged:
            reg *= reg_factor
            if reg > 1e6:
                break
            continue

        # line search / forward pass
        alpha = 1.0
        accepted = False
        for _ in range(10):
            xu = np.zeros_like(u)
            xnew = np.zeros((N + 1, n))
            xnew[0] = x0.copy()
            for t in range(N):
                du = alpha * k_seq[t] + K_seq[t] @ (xnew[t] - xs[t])
                xu[t] = u[t] + du
                xnew[t + 1] = dynamics(xnew[t], xu[t], dt, w=None)
            _, Jnew = forward_rollout(x0, xu)
            if Jnew < J:
                u = xu
                xs = xnew
                J = Jnew
                accepted = True
                reg = max(reg / reg_factor, 1e-6)
                break
            else:
                alpha *= 0.5

        if not accepted:
            reg *= reg_factor

        if np.abs(J - Jnew) < tol:
            break

    return xs, u, J

# --------------------------
# BaseMPPI
# --------------------------
def base_mppi(x0, u_nominal, dt, N, K, cost_fn, lambda_=1.0, sigma=1.0, u_min=None, u_max=None):
    """
    Classic PI-MPC:
    - x0: initial state
    - u_nominal: initial nominal control sequence shape (N, m)
    - K: number of rollouts (samples)
    - sigma: stddev of additive exploration on u (scalar or array)
    - lambda_: temperature parameter
    Returns: updated u_nominal, and optionally trajectory
    """
    Nseq, m = u_nominal.shape
    assert Nseq == N
    # Pre-sample all perturbations: shape (K, N, m)
    delta_u = np.random.randn(K, N, m) * (sigma if np.isscalar(sigma) else np.array(sigma).reshape(1,1,-1))
    costs = np.zeros(K)
    trajectories = np.zeros((K, N+1, x0.size))

    for k in range(K):
        x = x0.copy()
        trajectories[k, 0] = x
        total_cost = 0.0
        for t in range(N):
            u_try = u_nominal[t] + delta_u[k, t]
            # Clip if requested
            if u_min is not None:
                u_try = np.maximum(u_try, u_min)
            if u_max is not None:
                u_try = np.minimum(u_try, u_max)
            total_cost += cost_fn(x, u_try)
            # treat delta_u as resulting from action noise, but here we sample on control
            # Implement dynamics with optional control noise set to zero
            x = dynamics(x, u_try, dt, w=None)
            trajectories[k, t+1] = x
        # terminal cost
        total_cost += cost_fn(x, np.zeros(m))
        costs[k] = total_cost

    # compute weights
    costs_min = np.min(costs)
    expw = np.exp(-(costs - costs_min) / lambda_)  # subtract min for numerical stability
    weights = expw / (np.sum(expw) + 1e-12)

    # update nominal control by weighted average of perturbations
    weighted_d = np.tensordot(weights, delta_u, axes=(0,0))  # shape (N,m)
    u_new = u_nominal + weighted_d

    return u_new, trajectories, costs, weights

# --------------------------
# RefinedMPPI (paper-style)
# --------------------------
def refined_mppi(x0,
                 u_nom,
                 dt,
                 N,
                 K,
                 cost_fn,
                 lambda_=1.0,
                 nu=1.0,
                 sigma_epsilon=1.0,
                 R=None):
    """
    Refined MPPI with likelihood-ratio corrected running cost (Eq. (31) in the paper).

    Parameters
    ----------
    x0 : (n,) array
        initial state
    u_nom : (N, m) array
        nominal control sequence
    dt : float
        timestep
    N : int
        horizon length (u_nom.shape[0] must equal N)
    K : int
        number of sampled trajectories
    cost_fn : callable (x, u) -> scalar
        base running cost q(x,u) (does not include control quadratic unless you want it in q)
    lambda_ : float
        temperature / weight param (paper uses λ)
    nu : float >= 1.0
        exploration variance multiplier used in sampling BE (nu >= 1)
    sigma_epsilon : float
        stddev for sampled epsilons (standard normal scaled)
    R : (m,m) array or None
        control cost matrix used in the 1/2 u^T R u term in q̃. If None, uses identity*0.1

    Returns
    -------
    u_new : (N,m) array
        updated nominal control sequence
    x_trajectories : (K, N+1, n) array
        sampled trajectories
    costs : (K,) array
        S_tilde (full cost-to-go) for each sampled traj (used for diagnostics)
    weights : (K,) array
        normalized importance weights
    """
    n = x0.size
    m = u_nom.shape[1]
    assert u_nom.shape[0] == N

    if R is None:
        R = np.diag([0.1] * m)

    # r = number of noise channels (B returns n x r)
    r = B(x0).shape[1]

    # Pre-sample epsilons: shape (K, N, r), standard normal scaled by sigma_epsilon
    eps = np.random.randn(K, N, r) * sigma_epsilon

    # Storage
    x_trajs = np.zeros((K, N + 1, n))
    S_tilde = np.zeros(K)  # total modified cost-to-go for each sample

    # Sampling rollouts under q_{nu, u}
    for k in range(K):
        x = x0.copy()
        x_trajs[k, 0] = x
        total = 0.0
        for t in range(N):
            u_t = u_nom[t]  # center control for sampling
            bmat = B(x)     # (n, r)
            gmat = G(x)     # (n, m)

            # sampling noise for this sample/time
            eps_k_t = eps[k, t]  # (r,)

            # scale noise for exploration: BE = scale * B (paper uses nu on Bc) 
            # # we implement BE*eps*sqrt(dt) = B * (sqrt(nu)*eps) * sqrt(dt) 
            noise_state = bmat @ (np.sqrt(nu) * eps_k_t) * np.sqrt(dt)
            

            # compute modified running cost q̃ (Eq. (31))
            # base cost q(x,u)
            q_base = cost_fn(x, u_t)

            # 1) control quadratic term: 0.5 u^T R u
            term_ctrl_quad = 0.5 * float(u_t.T @ R @ u_t)

            # 2) lambda * u^T G(...) ε sqrt(dt) -- use a general consistent form:
            #    interpret the paper's G ε as the projection of state-noise along control directions.
            #    A robust, dimensionally-correct term is:
            #       lambda * u^T [ G^T Sigma^{-1} B eps ] * sqrt(dt)
            #    where Sigma = B B^T (state-noise covariance).
            Sigma = bmat @ bmat.T  # (n, n)
            # pseudo-inverse for stability
            Sigma_inv = np.linalg.pinv(Sigma + 1e-9 * np.eye(n))
            beps = bmat @ eps_k_t  # (n,)
            # compute the scalar cross term
            cross_vec = gmat.T @ (Sigma_inv @ beps)  # (m,)
            term_cross = float(lambda_ * (u_t @ cross_vec) * np.sqrt(dt))

            # 3) exploration penalty: 0.5 * lambda * (1 - nu^{-1}) * (beps^T * (BcB_c^T)^{-1} * beps) * dt
            #    in the general case we use Sigma_inv above: (beps^T Sigma_inv beps) scaled by dt
            term_expl = 0.5 * lambda_ * (1.0 - 1.0 / float(nu)) * (float(beps.T @ (Sigma_inv @ beps)) * dt)

            # combined modified running cost (q̃)
            q_tilde = q_base + term_ctrl_quad + term_cross + term_expl

            # accumulate S_tilde (sum q̃ * dt)
            total += q_tilde * dt

            # step dynamics with the sampled noise_state
            # equivalent to: x = x + (f(x)+G(x)u) * dt + B(x) * (sqrt(nu)*eps) * sqrt(dt)
            x = x + (f(x) + gmat @ u_t) * dt + noise_state
            x_trajs[k, t + 1] = x

        # terminal cost: add phi(x_T) if desired. Here we'll reuse cost_fn at final state with zero u.
        terminal_cost = cost_fn(x, np.zeros(m))
        total += terminal_cost  # terminal cost not multiplied by dt in paper's notation (they add φ separately)
        S_tilde[k] = total

    # Importance weights
    # stable exponent: subtract min before exponent
    minS = np.min(S_tilde)
    expw = np.exp(-(S_tilde - minS) / lambda_)
    weights = expw / (np.sum(expw) + 1e-12)

    # Compute update for u using the paper's structure:
    # u_new_j = u_nom_j + H^{-1} G * E_qnu[ weight * ( eps_j / sqrt(dt) ) ]  (discrete approx)
    # We'll compute a practical version:
    u_update = np.zeros_like(u_nom)  # (N, m)
    for t in range(N):
        # approximate local matrices at nominal (or mean sampled) state
        x_nom_est = np.mean(x_trajs[:, t], axis=0)
        Gmat = G(x_nom_est)     # (n, m)
        Bmat = B(x_nom_est)     # (n, r)
        Sigma_loc = Bmat @ Bmat.T
        Sigma_inv_loc = np.linalg.pinv(Sigma_loc + 1e-9 * np.eye(n))

        # H = G^T Sigma^{-1} G  (m x m)
        H = Gmat.T @ Sigma_inv_loc @ Gmat
        # regularize H for inversion stability
        H_reg = H + 1e-6 * np.eye(m)
        H_inv = np.linalg.pinv(H_reg)

        # accumulate numerator: E[ weight * ( G^T Sigma^{-1} B eps ) ]
        numer = np.zeros(m)
        for k in range(K):
            eps_k_t = eps[k, t]
            beps = Bmat @ eps_k_t
            term = Gmat.T @ (Sigma_inv_loc @ beps)  # (m,)
            numer += weights[k] * term

        # the paper's discrete formula yields a 1/sqrt(dt) factor when mapping eps -> control correction
        delta_u = H_inv @ numer / np.sqrt(dt)
        u_update[t] = delta_u

    u_new = u_nom + u_update
    return u_new, x_trajs, S_tilde, weights

# --------------------------
# Example usage (small demo)
# --------------------------
# if __name__ == "__main__":
#     np.random.seed(0)
#     dt = 0.05
#     N = 30
#     x0 = np.array([0.0, 0.0, 0.0, 0.0])
#     x_goal = np.array([5.0, 0.0, 0.0, 0.0])

#     m = 2
#     u0 = np.zeros((N, m))

#     def cost_wrapper(x, u):
#         # small running cost to get to px ~ x_goal[0]
#         Q = np.diag([1.0, 1.0, 0.2, 0.1])
#         R = np.diag([0.05, 0.05])
#         dx = x - x_goal
#         return float(dx.T @ Q @ dx + 0.5 * u.T @ R @ u)

#     # iLQR run
#     print("Running iLQR (this may take a few seconds)...")
#     xs, u_opt, J = ilqr(x0, u0.copy(), dt, N, cost_wrapper, max_iter=25)
#     print("iLQR returned cost J =", J)

#     # Base MPPI
#     print("Running BaseMPPI...")
#     u_nom = u0.copy()
#     u_nom, trajs, costs, w = base_mppi(x0, u_nom, dt, N, K=256, cost_fn=cost_wrapper, lambda_=1.0, sigma=0.5)
#     print("BaseMPPI: mean cost across samples:", np.mean(costs))

#     # Refined MPPI
#     print("Running RefinedMPPI...")
#     u_nom2 = u0.copy()
#     u_nom2, trajs2, costs2, w2 = refined_mppi(x0, u_nom2, dt, N, K=256, cost_fn=cost_wrapper, lambda_=1.0, nu=1.0)
#     print("RefinedMPPI: mean cost across samples:", np.mean(costs2))

#     print("Done.")


# -------------------------
# MPC wrapper
# -------------------------
def run_mpc(x0,
            x_goal,
            planner='RefinedMPPI',   # 'iLQR', 'BaseMPPI', or 'RefinedMPPI'
            T=6.0,
            dt=0.05,
            K_samples=256,
            display=False):
    np.random.seed(1)
    n = x0.size
    m = 2
    N = int(np.round(T / dt))
    u_nom = np.zeros((N, m))
    cost_fn = make_quadratic_cost(x_goal)

    max_steps = 200  # safety
    x = x0.copy()
    trajectory = [x.copy()]
    total_steps = 0

    for step in range(max_steps):
        total_steps += 1
        # Plan over horizon starting from current x
        if planner == 'iLQR':
            xs, u_opt, J = ilqr(x, u_nom.copy(), dt, N, cost_fn, max_iter=20)
            u_seq = u_opt
        elif planner == 'BaseMPPI':
            u_seq, _, _, _ = base_mppi(x, u_nom.copy(), dt, N, K_samples, cost_fn, lambda_=1.0, sigma=0.5)
        elif planner == 'RefinedMPPI':
            u_seq, _, _, _ = refined_mppi(x, u_nom.copy(), dt, N, K_samples, cost_fn, lambda_=1.0, nu=1, sigma_epsilon=1.0)
        else:
            raise ValueError("Unknown planner")

        # apply first control (no process noise here; you can add noise if desired)
        u_apply = u_seq[0]
        x = dynamics(x, u_apply, dt, w=None)  # w=None => deterministic step
        trajectory.append(x.copy())

        # shift u_seq for warm start: drop first, append last-initialized zero
        u_nom = np.vstack([u_seq[1:], np.zeros((1, m))])

        # termination condition: distance to goal small
        if np.linalg.norm(x[:2] - x_goal[:2]) < 0.2 and abs(x[3] - x_goal[3]) < 0.5:
            print(f"[MPC] Reached goal at step {step}, state {x}")
            break

        if step % 10 == 0:
            # print diagnostics
            dpos = np.linalg.norm(x[:2] - x_goal[:2])
            print(f"[MPC] step {step:3d} pos {x[0]:.3f},{x[1]:.3f} v {x[3]:.3f} dist {dpos:.3f}")

    return np.array(trajectory)

# -------------------------
# Example run
# -------------------------
if __name__ == "__main__":
    start = np.array([0.0, 0.0, 0.0, 0.0])
    goal = np.array([6.0, 0.0, 0.0, 0.0])

    # Run each planner
    print("Running closed-loop MPC with iLQR...")
    traj_ilqr = run_mpc(start, goal, planner='iLQR', T=3.0, dt=0.05, K_samples=128)
    print("Running closed-loop MPC with BaseMPPI...")
    traj_base = run_mpc(start, goal, planner='BaseMPPI', T=3.0, dt=0.05, K_samples=128)
    print("Running closed-loop MPC with RefinedMPPI...")
    traj_refined = run_mpc(start, goal, planner='RefinedMPPI', T=3.0, dt=0.05, K_samples=128)

    # --- Plot static trajectories ---
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(traj_ilqr[:,0], traj_ilqr[:,1], 'b-', label='iLQR')
    ax.plot(traj_base[:,0], traj_base[:,1], 'g-', label='BaseMPPI')
    ax.plot(traj_refined[:,0], traj_refined[:,1], 'm-', label='RefinedMPPI')
    ax.plot(goal[0], goal[1], 'rx', markersize=12, label='Goal')
    ax.set_title("Closed-loop MPC trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.axis('equal')
    plt.show()

    # --- Animate trajectories ---
    import matplotlib.animation as animation
    fig2, ax2 = plt.subplots()
    ax2.set_xlim(-1, 7); ax2.set_ylim(-3, 3)
    ax2.set_aspect('equal')
    ax2.plot(goal[0], goal[1], 'rx', markersize=12, label='Goal')
    lines = {
        "iLQR": ax2.plot([], [], 'b-', label='iLQR')[0],
        "BaseMPPI": ax2.plot([], [], 'g-', label='BaseMPPI')[0],
        "RefinedMPPI": ax2.plot([], [], 'm-', label='RefinedMPPI')[0],
    }
    ax2.legend()

    maxlen = max(len(traj_ilqr), len(traj_base), len(traj_refined))
    trajs = {"iLQR": traj_ilqr, "BaseMPPI": traj_base, "RefinedMPPI": traj_refined}

    def animate(i):
        for name, traj in trajs.items():
            if i < len(traj):
                lines[name].set_data(traj[:i+1,0], traj[:i+1,1])
        return list(lines.values())

    ani = animation.FuncAnimation(fig2, animate, frames=maxlen, interval=150, blit=True)
    plt.show()