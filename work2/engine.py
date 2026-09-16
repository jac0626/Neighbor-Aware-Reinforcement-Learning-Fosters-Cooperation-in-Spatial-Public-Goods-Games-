"""Reconstruction from work1 equations for new, reproducible work2 experiments.

Does not reproduce legacy script update timing or claim reproduction of published plots.
"""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit

METHODS = {'iql': 0, 'ni_global': 1, 'ni_local': 2, 'trimmed_local': 3,
           'cooperation_first': 4, 'learned': 5, 'gate_only': 6, 'selection_only': 7, 'cooperation_only': 8}
ATTACKS = {'none': 0, 'high_defect': 1, 'high_only': 2, 'flip_action': 3,
           'random_message': 4, 'burst_defect': 5, 'moderate_defect': 6, 'high_cooperate': 7}
STATES = {'reputation': 0, 'own_action': 1}
METRICS = ['cooperation', 'raw_welfare_per_agent', 'switch_rate',
           'selected_bad_fraction', 'active_selected_bad_fraction',
           'active_ni_fraction', 'mean_abs_ni', 'mean_gate']

@dataclass(frozen=True)
class Config:
    L: int = 30
    r: float = 3.6
    steps: int = 20000
    seed: int = 100
    M: int = 2
    state_mode: str = 'reputation'
    method: str = 'ni_global'
    kappa: float = 1.0
    payoff_weight: float = 1.0
    rho: float = 0.0
    attack: str = 'none'
    alpha: float = 0.8
    gamma: float = 0.9
    epsilon_initial: float = 0.5
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.99
    record_every: int = 100
    controller: tuple[float, ...] | None = None
    gate_scale: float = 1.0

    def __post_init__(self):
        if self.L < 5 or self.M not in (1, 2):
            raise ValueError('Use L>=5 to keep 12 distinct neighbors, M=1 or 2')
        if not 1 <= self.r <= 5:
            raise ValueError('This reconstruction uses payoff bounds for 1<=r<=5')
        if self.steps <= 0 or not 0 < self.record_every <= self.steps:
            raise ValueError('Invalid step/record budget')
        if self.method not in METHODS or self.attack not in ATTACKS:
            raise ValueError('Unknown method or attack')
        if self.state_mode not in STATES:
            raise ValueError('Unknown state representation')
        if 5 <= METHODS[self.method] <= 7:
            if self.controller is None or len(self.controller) != 12 or not np.isfinite(self.controller).all():
                raise ValueError('Learned methods require twelve finite controller parameters')
        if not 0 <= self.gate_scale <= 1:
            raise ValueError('gate_scale must be in [0,1]')
        if not 0 <= self.rho <= 1 or not 0 <= self.payoff_weight <= 1 or self.kappa < 0:
            raise ValueError('Invalid proportion, payoff weight or kappa')
        if not 0 < self.alpha <= 1 or not 0 <= self.gamma < 1:
            raise ValueError('Invalid learning parameters')
        if not 0 <= self.epsilon_min <= self.epsilon_initial <= 1 or not 0 < self.epsilon_decay <= 1:
            raise ValueError('Invalid exploration schedule')


def neighbors(L, M):
    offsets = [(1,0),(-1,0),(0,1),(0,-1)]
    if M == 2:
        offsets += [(2,0),(-2,0),(0,2),(0,-2),(1,1),(1,-1),(-1,1),(-1,-1)]
    return np.array([[((x+dx)%L)*L+(y+dy)%L for dx,dy in offsets]
                     for x in range(L) for y in range(L)], dtype=np.int64)


@njit(cache=True)
def raw_payoffs(actions, first_neighbors, r):
    n = len(actions)
    group_counts = np.empty(n, dtype=np.float64)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        count = 1 - actions[i]
        for j in first_neighbors[i]:
            count += 1 - actions[j]
        group_counts[i] = count
    for i in range(n):
        count = group_counts[i]
        for j in first_neighbors[i]:
            count += group_counts[j]
        result[i] = r * count / 5.0 - 5.0 * (1 - actions[i])
    return result


@njit(cache=True)
def local_controller(own_action, own_reward, messages_a, messages_r,
                     prediction, q_gap, theta, method, tie_draw, gate_scale):
    """Only observable local inputs; no sender identities, true neighbor values or labels."""
    d = len(messages_r)
    features = np.empty((d,5))
    scores = np.empty(d)
    if method == 4 or method == 8:
        for k in range(d):
            # A cooperative report outranks any defection report in [0,1].
            scores[k] = 2.0*(1-messages_a[k]) + messages_r[k]
    else:
        ordered = np.sort(messages_r)
        median = (ordered[(d-1)//2]+ordered[d//2])/2.0
        for k in range(d):
            features[k,0] = messages_r[k]-own_reward
            features[k,1] = 1.0 if messages_a[k] == own_action else -1.0
            features[k,2] = abs(messages_r[k]-median)
            features[k,3] = abs(messages_r[k]-prediction[messages_a[k]])
            features[k,4] = 1.0-messages_a[k]
            score = 0.0
            for f in range(5):
                score += theta[f]*features[k,f]
            scores[k] = messages_r[k] if method == 6 else score
    best = np.max(scores)
    ties = np.sum(scores == best)
    target = int(tie_draw*ties)
    selected = 0
    for k in range(d):
        if scores[k] == best:
            if target == 0:
                selected = k
                break
            target -= 1
    gate = 1.0
    if 5 <= method <= 7:
        if method != 7:
            logit = theta[11] + theta[10]/(1.0+abs(q_gap))
            contribution = 0.0
            for f in range(5):
                contribution += theta[5+f]*features[selected,f]
            logit += contribution
            # Stable sigmoid, retaining a continuous amplitude for finite logits.
            if logit >= 0:
                gate = 1.0/(1.0+np.exp(-logit))
            else:
                exp_logit = np.exp(logit)
                gate = exp_logit/(1.0+exp_logit)
        gate *= gate_scale
    elif method == 8:
        gate = 1.0-messages_a[selected]
    return selected, gate


@njit(cache=True)
def reported_message(action, reward, attack, step, message_rng):
    """One sender's broadcast; caller applies this only to the fixed faulty set."""
    if attack == 1:
        return 1, 1.0
    if attack == 2:
        return action, 1.0
    if attack == 3:
        return 1-action, reward
    if attack == 4:
        reported_reward = message_rng.random()
        reported_action = int(message_rng.random() >= 0.5)
        return reported_action, reported_reward
    if attack == 5 and step % 1000 < 500:
        return 1, 1.0
    if attack == 6:
        return 1, min(1.0, reward+0.2)
    if attack == 7:
        return 0, 1.0
    return action, reward


@njit(cache=True)
def _simulate(q_initial, bad, nb, r, steps, action_seed, method, kappa,
              w_p, attack, alpha, gamma, eps, eps_min, eps_decay, record_every,
              theta, gate_scale, state_mode, message_rng):
    np.random.seed(action_seed)
    n, d = nb.shape
    q = q_initial.copy()
    reputation = np.zeros(n)
    states = np.zeros(n, dtype=np.int64)
    actions = np.ones(n, dtype=np.int64)
    previous = np.ones(n, dtype=np.int64)
    rewards = np.empty(n)
    reported_r = np.empty(n)
    reported_a = np.empty(n, dtype=np.int64)
    tie_draws = np.empty(n)
    points = (steps + record_every - 1) // record_every
    trajectory = np.empty((points, 9))
    accum = np.zeros(8)
    tail = np.zeros(8)
    total = np.zeros(8)
    prediction = np.full((n,2),0.5) if 5 <= method <= 7 else np.empty((0,2))
    tail_count = 0
    record_count = 0
    out_index = 0
    for t in range(steps):
        for i in range(n):
            if state_mode == 1:
                states[i] = previous[i]
            else:
                rep_sum = reputation[i]
                for j in nb[i]:
                    rep_sum += reputation[j]
                states[i] = int(rep_sum > 0)
            # Fixed draw count regardless of method, attack, action or tie status.
            explore_draw = np.random.random()
            random_action = int(np.random.random() >= 0.5)
            greedy_tie = int(np.random.random() >= 0.5)
            tie_draws[i] = np.random.random()
            state = states[i]
            if q[i,state,0] > q[i,state,1]:
                greedy = 0
            elif q[i,state,1] > q[i,state,0]:
                greedy = 1
            else:
                greedy = greedy_tie
            actions[i] = random_action if explore_draw < eps else greedy
        payoff = raw_payoffs(actions, nb[:,:4], r)
        cooperators = 0
        switches = 0
        for i in range(n):
            cooperators += 1 - actions[i]
            switches += int(t > 0 and actions[i] != previous[i])
            reputation[i] = min(10.0, max(-10.0, reputation[i] + 1 - 2*actions[i]))
            rewards[i] = w_p * (payoff[i]-r+5)/(3*r+5) + (1-w_p)*(1-actions[i])
            reported_r[i] = rewards[i]
            reported_a[i] = actions[i]
            if bad[i]:
                reported_a[i], reported_r[i] = reported_message(
                    actions[i], rewards[i], attack, t, message_rng)
        # Q tables belong to separate agents, so these per-agent updates commute.
        for i in range(n):
            if state_mode == 1:
                next_state = actions[i]
            else:
                rep_sum = reputation[i]
                for j in nb[i]:
                    rep_sum += reputation[j]
                next_state = int(rep_sum > 0)
            s, a = states[i], actions[i]
            target = rewards[i] + gamma * max(q[i,next_state,0], q[i,next_state,1])
            q[i,s,a] += alpha * (target - q[i,s,a])
        global_gap = 0.0
        if method == 1 and kappa > 0:
            for i in range(n):
                for j in nb[i]:
                    global_gap = max(global_gap, abs(reported_r[j] - rewards[i]))
        selected_bad = 0
        active_bad = 0
        active_count = 0
        abs_ni = 0.0
        gate_sum = 0.0
        enabled = method != 0 and kappa > 0 and (method < 5 or method == 8 or gate_scale > 0)
        if enabled:
            for i in range(n):
                cap = 2.0
                if method == 3:
                    values = np.empty(d)
                    for k in range(d):
                        values[k] = reported_r[nb[i,k]]
                    values.sort()
                    # Remove values strictly above this upper order statistic.
                    cap = values[d - 1 - int(0.2*d)]
                best = -1.0e100
                local_gap = 0.0
                ties = 0
                for j in nb[i]:
                    value = reported_r[j]
                    local_gap = max(local_gap, abs(value-rewards[i]))
                    if value <= cap:
                        if value > best:
                            best, ties = value, 1
                        elif value == best:
                            ties += 1
                selected_rank = int(tie_draws[i]*ties)
                reference = nb[i,0]
                rank = 0
                for j in nb[i]:
                    if reported_r[j] == best:
                        if rank == selected_rank:
                            reference = j
                            break
                        rank += 1
                gate = 1.0
                if method >= 4:
                    messages_r = reported_r[nb[i]]
                    messages_a = reported_a[nb[i]]
                    own_prediction = prediction[i] if 5 <= method <= 7 else np.empty(0)
                    index, gate = local_controller(actions[i],rewards[i],messages_a,messages_r,
                        own_prediction,q[i,states[i],0]-q[i,states[i],1],
                        theta,method,tie_draws[i],gate_scale)
                    reference = nb[i,index]
                    best = reported_r[reference]
                gate_sum += gate
                selected_bad += int(bad[reference])
                gap = max(0.0, best-rewards[i])
                denom = (global_gap if method == 1 else local_gap) + 0.01
                update = kappa * gap / denom * gate
                if update > 0:
                    active_count += 1
                    active_bad += int(bad[reference])
                abs_ni += update
                sign = 1.0 if actions[i] == reported_a[reference] else -1.0
                q[i,states[i],actions[i]] += sign * update
        if 5 <= method <= 7:
            for i in range(n):
                a = actions[i]
                prediction[i,a] = 0.95*prediction[i,a]+0.05*rewards[i]
        metrics = np.empty(8)
        metrics[0] = cooperators/n
        metrics[1] = np.mean(payoff)
        metrics[2] = switches/n
        metrics[3] = selected_bad/n if enabled else np.nan
        metrics[4] = active_bad/n  # Aggregate numerator; divide by active fraction below.
        metrics[5] = active_count/n
        metrics[6] = abs_ni/n
        metrics[7] = gate_sum/n
        # Diagnostic fractions can be undefined; summaries preserve NaN explicitly.
        accum += metrics
        total += metrics
        record_count += 1
        if t >= int(0.8*steps):
            tail += metrics
            tail_count += 1
        if (t+1) % record_every == 0 or t+1 == steps:
            trajectory[out_index,0] = t+1
            trajectory[out_index,1:] = accum/record_count
            trajectory[out_index,5] = accum[4]/accum[5] if accum[5] else np.nan
            out_index += 1
            accum[:] = 0
            record_count = 0
        previous[:] = actions
        eps = max(eps_min, eps*eps_decay)
    tail[4] = tail[4]/tail[5]*tail_count if tail[5] else np.nan
    total[4] = total[4]/total[5]*steps if total[5] else np.nan
    return trajectory, tail/tail_count, total/steps, actions, q


def simulate(cfg):
    streams = np.random.SeedSequence(cfg.seed).spawn(4)
    initial_rng = np.random.default_rng(streams[0])
    mask_rng = np.random.default_rng(streams[1])
    action_seed = int(streams[2].generate_state(1)[0])
    message_rng = np.random.default_rng(streams[3])
    n = cfg.L**2
    q = initial_rng.uniform(-0.01, 0.01, size=(n,2,2))
    bad = np.zeros(n, dtype=np.bool_)
    bad[mask_rng.permutation(n)[:int(cfg.rho*n)]] = True
    nb = neighbors(cfg.L,cfg.M)
    values = _simulate(q,bad,nb,cfg.r,cfg.steps,action_seed,METHODS[cfg.method],cfg.kappa,
                       cfg.payoff_weight,ATTACKS[cfg.attack],cfg.alpha,cfg.gamma,
                       cfg.epsilon_initial,cfg.epsilon_min,cfg.epsilon_decay,cfg.record_every,
                       np.asarray(cfg.controller if cfg.controller is not None else np.zeros(12),dtype=np.float64),
                       cfg.gate_scale, STATES[cfg.state_mode], message_rng)
    trajectory,tail,whole,actions,q_final = values
    def metrics_dict(vector):
        return {key: (float(val) if np.isfinite(val) else None) for key,val in zip(METRICS,vector)}
    return {'config':asdict(cfg), 'tail':metrics_dict(tail), 'whole':metrics_dict(whole),
            'trajectory':trajectory, 'actions':actions, 'q':q_final,
            'bad_mask':bad, 'realized_rho':float(bad.mean()),
            'candidate_bad_fraction':float(bad[nb].mean())}
