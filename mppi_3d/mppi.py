import numpy as np
import CONSTANTS as C

class MPPI:
    """
    Sampling-based MPC (MPPI) for additive Gaussian input noise:
      v_t ~ N(u_t, Sigma)

    Update (importance sampling):
      w_k ∝ exp(-(1/lambda) * S_k)
      u_t <- sum_k w_k * v_t^(k)

    --------------------------------

    sigma: (m,m) covariance for control noise
    u_min/u_max: (m,) bounds for clipping (soft constraints are also possible via cost)
    cost_fn(x_seq, u_seq, ref_seq) -> scalar cost
    dynamics_fn(x, u, dt) -> x_next
    """
    
    def __init__ (self, horizon: int, rollouts: int, lamda_: float, sigma: np.ndarray,
        u_min: np.ndarray, u_max: np.ndarray, dt: float, cost_fn, dynamics_fn, seed: int = C.SEED):
        
        self.horizon = horizon
        self.rollouts = rollouts
        self.lamda_ = lamda_
        self.sigma = sigma
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.cost_fn = cost_fn
        self.dynamics_fn = dynamics_fn
        self.rng = np.random.default_rng(seed)
        self.m = self.u_min.shape[0] # control dimension, here: 3 accelerations

        # Nominal control sequence U(N,m) = [u0, u1, .. u{n-1}]
        self.U =np.zeros((self.horizon, self.m))

        # Cholesky for fast sampling, self._L @ self._L.T ≈ self.sigma
        '''
        Cholesky decomposition is used to transform independent standard Gaussian samples into noise
        with a desired covariance, enabling correct sampling of control perturbations in MPPI
        '''
        self._L = np.linalg.cholesky(self.sigma + 1e-12 * np.eye(self.m)) # Cholesky factor

    def reset(self, u_init=None):
        if u_init is None:
            self.U[:] = 0.0
        else:
            u_init = np.array(u_init)
            self.U[:] = u_init
        
    def command(self, x0, ref_seq):
        """
         Run one MPPI iteration and return u0 to apply.
         Also shift horizon (receding horizon).
        """
        x0 = np.array(x0)
    
        # Rollouts storage
        costs = np.zeros(self.rollouts)
        V_all = np.zeros((self.rollouts, self.horizon, self.m))

        ########################################################
        # 1) Sample and Simulate

        for rollout in range(self.rollouts):
             
            # sample standard Gaussian noise
            standard_Gaussian_noise = self.rng.standard_normal((self.horizon, self.m))
            # noise eps_t ~ N(0, Sigma) # epsilon = (standard_Gaussian_noise) x (L.T)
            eps = (standard_Gaussian_noise @ self._L.T)

            # v_t = u_t + eps_t
            V = self.U + eps
            V = np.clip(V, self.u_min, self.u_max)
            V_all[rollout] = V

            # simulate trajectory
            X = np.zeros((self.horizon+1, x0.shape[0]))
            X[0]= x0
            
            for t in range(self.horizon):
                X[t+1] = self.dynamics_fn( X[t], V[t], self.dt)

            # evaluate cost
            costs[rollout] = self.cost_fn(X, V, ref_seq)

    
        ########################################################
        # 2) Compute weights
        # # w_k ∝ exp(-(1/lambda) * S_k)

        c_min = np.min(costs)
        weights = np.exp( -(costs - c_min) / max(self.lamda_, 1e-10))
        weights_norm = weights / (np.sum(weights) + 1e-10)

        ########################################################
        # 3) Update U by weighted average of sampled controls 
        # u_t <- E[w * v_t]

        self.U = np.zeros((self.horizon, self.m))
        for rollout in range(self.rollouts):
            self.U += weights_norm[rollout] * V_all[rollout]

        self.U = np.clip(self.U, self.u_min, self.u_max)

        ########################################################
        # 4) Receding horizon: output u0, shift sequence
        u0 = self.U[0].copy()
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2]  # or zeros or keep last

        return u0, costs, weights_norm


########################################################
# Costs 

# def obstacle_penalty(p_xy, walls):
#     """
#     Soft penalty for being inside an inflated rectangle around each pillar.
#     If inside -> big penalty.
#     """ 
#     x, y = p_xy

#     if is_free(x,y,walls):
#         penalty = 0
#     else:
#         penalty = 1e6

#     return penalty
def obstacle_penalty(p_xyz, collision_fn):
    """
    Generic obstacle penalty using an externally provided collision function
    """
    return 0.0 if collision_fn(p_xyz) else 1e6


def mppi_cost(X, U, ref_seq, collision_fn, w_pos=10.0, w_vel=0.5, w_u=0.05):
    """
    X: (N+1,12)   state trajectory
    U: (N,3)     accel commands
    ref_seq: (N+1,3) reference positions
    """
    cost = 0.0
    for t in range(U.shape[0]):
        p = X[t+1, 0:3]
        v = X[t+1, 3:6]
        a = U[t]

        # tracking
        e = p - ref_seq[t+1]
        cost += w_pos * (e @ e)

        # keep velocity reasonable
        cost += w_vel * (v @ v)

        # small control penalty (KL term)
        cost += w_u * (a @ a)

        # obstacles (x,y only, since your map is 2D pillars)
        cost += obstacle_penalty(p[0:3], collision_fn)

    return cost


def count_wall_hits(traj, collision_fn):
    """
    Count wall collision events based on a collision function.

    A hit is counted when the trajectory transitions:
        free -> collision
    """

    hits = 0
    was_free = True  # assume start is collision-free

    for p in traj:
        x, y , z = p[0], p[1], p[2]
        free_now = collision_fn((x, y, z))

        if was_free and not free_now:
            hits += 1

        was_free = free_now

    return hits
