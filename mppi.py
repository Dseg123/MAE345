import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ---------- dynamics: simple bicycle-like with nonlinear drag and slip ----------
# state: [x, y, theta, v]
# control: [a, delta]  (acceleration, steering angle)
L = 0.5  # wheelbase

def dynamics(x, u, dt, w=None):
    """
    Stochastic bicycle dynamics with additive diffusion.
    w: noise vector (size n or r), scaled by sqrt(dt)
    """
    x_new = x.copy()
    a, delta = u
    v = x[3]
    beta = np.arctan(0.5 * np.tan(delta))
    x_new[0] += v * np.cos(x[2] + beta) * dt
    x_new[1] += v * np.sin(x[2] + beta) * dt
    x_new[2] += (v / L) * np.tan(delta) * dt
    x_new[3] += (a - 0.2 * v**3) * dt
    if w is not None:
        B = np.eye(4)  # diffusion matrix (can generalize)
        x_new += B @ w * np.sqrt(dt)  # correct SDE scaling
    return x_new

# finite-difference Jacobians for linearization (fx, fu)
def finite_difference_jacobians(x, u, dt, eps=1e-5):
    n = x.size
    m = u.size
    f0 = dynamics(x, u, dt)
    fx = np.zeros((n, n))
    fu = np.zeros((n, m))
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        fx[:, i] = (dynamics(xp, u, dt) - f0) / eps
    for j in range(m):
        up = u.copy(); up[j] += eps
        fu[:, j] = (dynamics(x, up, dt) - f0) / eps
    return fx, fu

# ---------- cost ----------
goal = np.array([8.0, 0.0])

def running_cost(x, u):
    pos_err = np.linalg.norm(x[:2] - goal)
    desired_heading = np.arctan2(goal[1]-x[1], goal[0]-x[0])
    heading_err = np.arctan2(np.sin(desired_heading - x[2]), np.cos(desired_heading - x[2]))
    v = x[3]
    v_err = (v - 3.0)**2
    u_cost = 0.02 * (u[0]**2 + u[1]**2)
    return 1.0*pos_err + 0.5*abs(heading_err) + 0.2*v_err + u_cost

def terminal_cost(x):
    pos_err = np.linalg.norm(x[:2] - goal)
    return 10.0 * pos_err

# ---------- rollout ----------
def rollout(x0, U, dt, noise_seq=None):
    x = x0.copy()
    xs = [x.copy()]
    cost = 0.0
    for k, u in enumerate(U):
        if noise_seq is not None:
            uk = u + noise_seq[k]
            # stochastic dynamics: noise already included in noise_seq?
            w = None
        else:
            uk = u
            w = None
        cost += running_cost(x, uk) * dt
        x = dynamics(x, uk, dt, w)
        xs.append(x.copy())
    cost += terminal_cost(xs[-1])
    return np.array(xs), cost

# ---------- Shooting baseline ----------
def random_shooting(x0, N, K=1024, sigma=(0.8, 0.3), dt=0.05):
    m = 2
    sig = np.array(sigma)
    costs = np.zeros(K)
    best_traj = None
    best_cost = np.inf
    for k in range(K):
        noise = np.random.randn(N, m) * sig
        U = noise  # mean zero shooting
        xs, costs[k] = rollout(x0, U, dt)
        if costs[k] < best_cost:
            best_cost = costs[k]
            best_traj = xs
    return best_traj, costs

# ---------- Plain MPPI ----------
def mppi_plain_update(x0, U_init, K=1024, sigma=(0.6, 0.2), dt=0.05, lam=1.0):
    N = U_init.shape[0]
    m = U_init.shape[1]
    sig = np.array(sigma)
    eps = np.random.randn(K, N, m)
    costs = np.zeros(K)
    for k in range(K):
        noise_seq = eps[k] * sig
        Uk = U_init + noise_seq
        _, costs[k] = rollout(x0, Uk, dt)
    Smin = costs.min()
    weights = np.exp(-1.0/lam * (costs - Smin))
    W = weights / np.sum(weights)
    delta = np.tensordot(W, eps * sig, axes=(0,0))
    U_new = U_init + delta
    return U_new, costs

# ---------- Refined MPPI (Eq.18 mapping), using sigma = B B.T ----------
def mppi_refined_with_diffusion(x0, U_init, K=1024, nu=1.0, dt=0.05, lam=1.0):
    N, m = U_init.shape
    r = m  # assume control and noise dimensions equal
    eps_fd = 1e-6
    
    # 1) nominal trajectory
    xs_nom = [x0.copy()]
    for k in range(N):
        xs_nom.append(dynamics(xs_nom[-1], U_init[k], dt, w=None))
    xs_nom = np.array(xs_nom)
    
    # 2) compute G and B
    G_list, B_list = [], []
    for k in range(N):
        xk, uk = xs_nom[k], U_init[k]
        n = x0.size
        
        # G_k = df/du
        G = np.zeros((n, m))
        for j in range(m):
            up = uk.copy(); up[j] += eps_fd
            G[:, j] = (dynamics(xk, up, dt) - dynamics(xk, uk, dt)) / eps_fd
        G_list.append(G)
        
        # B_k = df/dw
        B = np.zeros((n, r))
        for j in range(r):
            w = np.zeros(r); w[j] += eps_fd
            B[:, j] = (dynamics(xk, uk, dt, w=w) - dynamics(xk, uk, dt)) / eps_fd
        B[-1, -1] *= nu
        B_list.append(B)
    
    # 3) sample noise and compute control perturbations
    mapped_perturbations = np.zeros((K, N, m))
    costs = np.zeros(K)
    
    for k_sample in range(K):
        xs = [x0.copy()]
        noise_seq = np.zeros((N, m))
        for t in range(N):
            G, B = G_list[t], B_list[t]
            
            Sigma = B @ B.T
            Sigma_inv = np.linalg.inv(Sigma + 1e-8*np.eye(n))  # regularize
            
            H = G.T @ Sigma_inv @ G
            H = 0.5*(H + H.T) + 1e-8*np.eye(m)
            H_inv = np.linalg.inv(H)
            
            eps = np.random.randn(r)
            eps = np.linalg.cholesky(Sigma + 1e-8*np.eye(n)) @ eps
            
            delta_u = H_inv @ (G.T @ Sigma_inv @ B @ eps)
            noise_seq[t] = delta_u
            
            x_next = dynamics(xs[-1], U_init[t] + delta_u, dt, w=B @ eps * np.sqrt(dt))
            xs.append(x_next)
        xs = np.array(xs)
        mapped_perturbations[k_sample] = noise_seq
        
        cost = 0.0
        for t in range(N):
            cost += running_cost(xs[t], U_init[t] + noise_seq[t]) * dt
        cost += terminal_cost(xs[-1])
        costs[k_sample] = cost
    
    # 4) MPPI weighted update
    Smin = costs.min()
    weights = np.exp(-1.0/lam * (costs - Smin))
    W = weights / np.sum(weights)
    delta = np.tensordot(W, mapped_perturbations, axes=(0,0))
    U_new = U_init + delta
    return U_new, costs


# ---------- Simple iLQR ----------
def ilqr_solve(x0, U_init, N_iter=8, dt=0.05):
    U = U_init.copy()
    N = U.shape[0]
    n = x0.size
    m = U.shape[1]
    for it in range(N_iter):
        xs = [x0.copy()]
        for k in range(N):
            xs.append(dynamics(xs[-1], U[k], dt))
        xs = np.array(xs)
        
        Vx = grad_terminal(xs[-1])
        Vxx = hess_terminal(xs[-1])
        Ks = np.zeros((N, m, n))
        ks = np.zeros((N, m))
        reg = 1e-6
        diverged = False
        
        for k in reversed(range(N)):
            xk, uk = xs[k], U[k]
            fx, fu = finite_difference_jacobians(xk, uk, dt)
            lx = grad_running_state(xk, uk) * dt
            lu = grad_running_control(xk, uk) * dt
            lxx = hess_running_state(xk, uk) * dt
            luu = hess_running_control(xk, uk) * dt
            lux = np.zeros((m, n))
            Qx = lx + fx.T.dot(Vx)
            Qu = lu + fu.T.dot(Vx)
            Qxx = lxx + fx.T.dot(Vxx).dot(fx)
            Quu = luu + fu.T.dot(Vxx).dot(fu)
            Qux = lux + fu.T.dot(Vxx).dot(fx)
            Quu_reg = Quu + reg*np.eye(m)
            try:
                inv_Quu = np.linalg.inv(Quu_reg)
            except np.linalg.LinAlgError:
                diverged = True
                break
            Ks[k] = -inv_Quu @ Qux
            ks[k] = -inv_Quu @ Qu
            Vx = Qx + Ks[k].T.dot(Quu).dot(ks[k]) + Ks[k].T.dot(Qu) + Qux.T.dot(ks[k])
            Vxx = Qxx + Ks[k].T.dot(Quu).dot(Ks[k]) + Ks[k].T.dot(Qux) + Qux.T.dot(Ks[k])
            Vxx = 0.5*(Vxx + Vxx.T)
        if diverged:
            reg *= 10
            continue
        
        alpha = 1.0
        success = False
        cost_old = rollout(x0, U, dt)[1]
        for _ in range(5):
            Xnew = [x0.copy()]
            Unew = []
            for k in range(N):
                du = alpha * ks[k] + Ks[k].dot(Xnew[-1]-xs[k])
                u_try = U[k] + du
                Unew.append(u_try)
                Xnew.append(dynamics(Xnew[-1], u_try, dt))
            Unew = np.array(Unew)
            cost_new = rollout(x0, Unew, dt)[1]
            if cost_new < cost_old:
                U = Unew
                success = True
                break
            alpha *= 0.5
        if not success:
            reg *= 10
    return U

# ---------- helpers: numerical gradients/hessians ----------
def grad_running_state(x, u, eps=1e-5):
    n = x.size
    g = np.zeros(n)
    f0 = running_cost(x, u)
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        g[i] = (running_cost(xp, u) - f0) / eps
    return g

def grad_running_control(x, u, eps=1e-5):
    m = u.size
    g = np.zeros(m)
    f0 = running_cost(x, u)
    for i in range(m):
        up = u.copy(); up[i] += eps
        g[i] = (running_cost(x, up) - f0) / eps
    return g

def hess_running_state(x, u, eps=1e-4):
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ej = np.zeros(n)
            ei[i] = eps; ej[j] = eps
            fpp = running_cost(x+ei+ej, u)
            fpm = running_cost(x+ei-ej, u)
            fmp = running_cost(x-ei+ej, u)
            fmm = running_cost(x-ei-ej, u)
            H[i,j] = (fpp - fpm - fmp + fmm) / (4*eps*eps)
    return H

def hess_running_control(x, u, eps=1e-4):
    m = u.size
    H = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            ei = np.zeros(m); ej = np.zeros(m)
            ei[i] = eps; ej[j] = eps
            fpp = running_cost(x, u+ei+ej)
            fpm = running_cost(x, u+ei-ej)
            fmp = running_cost(x, u-ei+ej)
            fmm = running_cost(x, u-ei-ej)
            H[i,j] = (fpp - fpm - fmp + fmm) / (4*eps*eps)
    return H

def grad_terminal(x, eps=1e-5):
    n = x.size
    g = np.zeros(n)
    f0 = terminal_cost(x)
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        g[i] = (terminal_cost(xp) - f0) / eps
    return g

def hess_terminal(x, eps=1e-4):
    n = x.size
    H = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ej = np.zeros(n)
            ei[i] = eps; ej[j] = eps
            fpp = terminal_cost(x+ei+ej)
            fpm = terminal_cost(x+ei-ej)
            fmp = terminal_cost(x-ei+ej)
            fmm = terminal_cost(x-ei-ej)
            H[i,j] = (fpp - fpm - fmp + fmm) / (4*eps*eps)
    return H

# ---------- MPC runner ----------
def run_mpc_planner(planner_update_fn, planner_name, x0, Tsim=3.0, dt=0.05, N=40, **kwargs):
    t = 0.0
    x = x0.copy()
    traj = [x.copy()]
    U = np.zeros((N,2)); U[:,0] = 1.0
    steps = int(np.ceil(Tsim / dt))
    costs = []
    for step in range(steps):
        U, info = planner_update_fn(x, U, **kwargs)
        u0 = U[0].copy()
        x = dynamics(x, u0, dt)
        traj.append(x.copy())
        U = np.vstack([U[1:], np.zeros((1,2))])
        costs.append(running_cost(x, u0))
        if np.linalg.norm(x[:2]-goal) < 0.2:
            break
    traj = np.array(traj)
    print(f'MPC {planner_name}: executed {len(traj)-1} steps, final pos {traj[-1,:2]}, total cost {np.sum(costs):.3f}')
    return traj

# wrapper planners
def planner_plain_mppi(x, U_init, **kwargs):
    U_new, costs = mppi_plain_update(x, U_init, **kwargs)
    return U_new, {'costs': costs}

def planner_refined_mppi(x, U_init, **kwargs):
    U_new, costs = mppi_refined_with_diffusion(x, U_init, **kwargs)
    return U_new, {'costs': costs}

def planner_ilqr(x, U_init, **kwargs):
    U_new = ilqr_solve(x, U_init, **kwargs)
    return U_new, {}

# ---------- demo ----------
def run_full_demo():
    dt = 0.05; N = 40
    x0 = np.array([0.0, 0.0, 0.0, 0.5])

    # print('Running shooting baseline...')
    # X_shoot, _ = random_shooting(x0, N, K=512, sigma=(0.8,0.3), dt=dt)

    # print('Running MPC with Plain MPPI...')
    # traj_plain = run_mpc_planner(planner_plain_mppi, 'Plain MPPI', x0, Tsim=4.0, dt=dt, N=N, K=512, sigma=(0.6,0.2), lam=1.0)

    print('Running MPC with Refined MPPI...')
    traj_refined = run_mpc_planner(planner_refined_mppi, 'Refined MPPI', x0, Tsim=4.0, dt=dt, N=N, K=512, nu=5.0, lam=1.0)

    print('Running MPC with iLQR baseline...')
    traj_ilqr = run_mpc_planner(planner_ilqr, 'iLQR', x0, Tsim=4.0, dt=dt, N=N, N_iter=6)

    plt.figure(figsize=(10,6))
    if X_shoot is not None:
        plt.plot(X_shoot[:,0], X_shoot[:,1], '--', label='Shooting baseline')
    plt.plot(traj_plain[:,0], traj_plain[:,1], label='Plain MPPI')
    plt.plot(traj_refined[:,0], traj_refined[:,1], label='Refined MPPI')
    plt.plot(traj_ilqr[:,0], traj_ilqr[:,1], label='iLQR')
    plt.scatter([goal[0]],[goal[1]], marker='*', s=150, label='goal')
    plt.legend()
    plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal'); plt.title('Closed-loop MPC trajectories')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_full_demo()
