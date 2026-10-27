"""
shared layer for the MO-WOA multi-objective comparison (phase B, axis 2).

all four algorithms read the same enumerated objective table and use the same
encoding, repair, initial population per (article, seed) and FE budget, so
objective identity holds by construction.

protocol decisions: D-1 shared P0 (paired design), D-2 terminal + cumulative
fronts, D-3 B = 1480 = 30 + 50 x 29 (surplus goes to the peers, never MO-WOA),
D-4 no tuning, D-6 strata by m = |P*|: A m>=3, B m==2, C m==1 (degenerate).
summary_size=3, repair_solution(), position bonus, knee_weights=[32, 1], the
70/30 leader split and leader preservation are copied verbatim from cell 20 of [v1_4_1].
"""

import numpy as np
from itertools import combinations

# protocol constants (decisions D-1 to D-6 above), don't change them
K_SUMMARY      = 3                       # summary_size (locked)
POP_SIZE       = 30                      # MO-WOA locked
MAX_ITER       = 50                      # MO-WOA locked
FE_BUDGET      = 1480                    # 30 + 50*29   (D-3)
KNEE_WEIGHTS   = np.array([32.0, 1.0])   # w_cov = 32 (locked)
REDUNDANCY_OFFSET = 0.2                  # delta in f2
REF_POINT      = np.array([1.1, 1.1])    # HV reference, normalized space


# per-article artefacts, computed once (BGE-M3 is slow)
class ArticleCache:
    """Everything the optimizer needs for one article. No embeddings at run time."""

    def __init__(self, article_id, adj_matrix, node_importance, position_bonus):
        self.article_id        = article_id
        self.adj               = np.asarray(adj_matrix, dtype=float)
        self.node_importance   = np.asarray(node_importance, dtype=float)
        self.position_bonus    = np.asarray(position_bonus, dtype=float)
        self.n                 = len(self.node_importance)
        # same as cell 20: graph.nodes[i]['importance'] * position_bonus[i]
        self.biased_importance = self.node_importance * self.position_bonus

    def to_dict(self):
        return dict(article_id=self.article_id, adj=self.adj,
                    node_importance=self.node_importance,
                    position_bonus=self.position_bonus)

    @staticmethod
    def from_dict(d):
        return ArticleCache(d['article_id'], d['adj'],
                            d['node_importance'], d['position_bonus'])


def build_position_bonus(n):
    """Copied from cell 28 of [v1_4_1], don't touch."""
    pb = np.ones(n)
    pb[0]   = 3.0
    pb[1:3] *= 2.5
    pb[3:5] *= 1.5
    pb[5:8] *= 0.8
    pb[8:]  *= 0.3
    return pb


# objectives, one implementation for all four algorithms
def evaluate_objectives_reference(solution, cache):
    """
    Literal copy of cell 20's evaluate_objectives(), only kept as ground truth for
    G4 (slow double loop). Returns native (coverage max, redundancy min).
    """
    solution = np.asarray(solution).astype(int)
    selected = np.where(solution == 1)[0]
    coverage = cache.biased_importance[selected].sum() if len(selected) > 0 else 0
    redundancy_raw = 0.0
    num_pairs = 0
    for i_idx, i in enumerate(selected):
        for j_idx, j in enumerate(selected):
            if j_idx > i_idx:
                redundancy_raw += cache.adj[i][j]
                num_pairs += 1
    redundancy = redundancy_raw / max(1, num_pairs) if num_pairs > 0 else 0
    redundancy = max(0, redundancy - REDUNDANCY_OFFSET)
    return coverage, redundancy


def evaluate_objectives_vec(combos, cache):
    """Vectorized: (M, k) sorted combos -> (M, 2) native (coverage, redundancy). Checked by G4."""
    combos = np.asarray(combos, dtype=int)
    M, k = combos.shape
    coverage = cache.biased_importance[combos].sum(axis=1)

    n_pairs = k * (k - 1) // 2
    if n_pairs == 0:
        redundancy = np.zeros(M)
    else:
        iu, ju = np.triu_indices(k, k=1)          # all i<j pairs within a subset
        rows = combos[:, iu]                      # (M, n_pairs)
        cols = combos[:, ju]
        redundancy_raw = cache.adj[rows, cols].sum(axis=1)
        redundancy = redundancy_raw / n_pairs
        redundancy = np.maximum(0.0, redundancy - REDUNDANCY_OFFSET)

    return np.column_stack([coverage, redundancy])


# repair operator, copied from cell 20. don't touch
def repair_solution(solution, cache, target_length=K_SUMMARY):
    solution = np.asarray(solution).astype(int).copy()
    bi = cache.biased_importance
    selected = np.where(solution == 1)[0]
    num_selected = len(selected)

    if num_selected > target_length:
        drop_order = selected[np.argsort(bi[selected])]        # lowest importance first
        for idx in drop_order[:num_selected - target_length]:
            solution[idx] = 0
    elif num_selected < target_length:
        unselected = np.where(solution == 0)[0]
        if len(unselected) > 0:
            add_order = unselected[np.argsort(-bi[unselected])]  # highest importance first
            for idx in add_order[:target_length - num_selected]:
                solution[idx] = 1

    if not solution.any():
        solution[np.argmax(bi)] = 1
    return solution


class Enumeration:
    """All C(N, k) subsets of one article: the exact true front (not a pooled one) + O(1) lookups."""

    def __init__(self, cache, k=K_SUMMARY):
        self.cache = cache
        self.k = k
        n = cache.n
        if n < k:
            raise ValueError(f"article {cache.article_id}: N={n} < k={k}")

        self.combos = np.array(list(combinations(range(n), k)), dtype=np.int16)
        self.M = len(self.combos)
        self.F_native = evaluate_objectives_vec(self.combos, cache)   # (M,2) (cov, red)

        # minimization form used by pymoo and every indicator: (-coverage, redundancy)
        self.F_min = np.column_stack([-self.F_native[:, 0], self.F_native[:, 1]])

        # O(1) lookup:  sorted index tuple -> row in F_native
        self.index_of = {tuple(c): i for i, c in enumerate(map(tuple, self.combos))}

        # normalize on the full feasible set, never on algorithm output
        cov = self.F_native[:, 0]
        red = self.F_native[:, 1]
        self.cov_min, self.cov_max = cov.min(), cov.max()
        self.red_min, self.red_max = red.min(), red.max()
        self.cov_range = self.cov_max - self.cov_min
        self.red_range = self.red_max - self.red_min

        # exact true front, deduped in objective space
        nd = non_dominated_mask(self.F_min)
        F_pf = self.F_min[nd]
        self.P_star_min = unique_rows(F_pf)                 # (m, 2) minimization
        self.P_star_norm = self.normalize(self.P_star_min)
        self.m = len(self.P_star_min)                       # |P*|  -> stratum

        # HV of the true front = max attainable HV
        self.hv_star = hypervolume(self.P_star_norm, REF_POINT)

        # true coverage extreme + true knee, for the hit-rate metrics
        self.true_cov_extreme_idx = int(np.argmax(cov))
        self.true_knee_row = self._true_knee()

    def normalize(self, F_min):
        """Map minimization objectives into per-article [0,1]^2 (both minimized)."""
        F_min = np.atleast_2d(np.asarray(F_min, dtype=float))
        cov = -F_min[:, 0]
        red = F_min[:, 1]
        cr = self.cov_range if self.cov_range > 0 else 1.0
        rr = self.red_range if self.red_range > 0 else 1.0
        g1 = (self.cov_max - cov) / cr          # 0 = best coverage
        g2 = (red - self.red_min) / rr          # 0 = least redundant
        return np.column_stack([g1, g2])

    def F_of(self, solution):
        """O(1) lookup for a binary vector, native (coverage, redundancy)."""
        key = tuple(np.where(np.asarray(solution).astype(int) == 1)[0].tolist())
        return self.F_native[self.index_of[key]]

    def row_of(self, solution):
        key = tuple(np.where(np.asarray(solution).astype(int) == 1)[0].tolist())
        return self.index_of[key]

    def _true_knee(self):
        """Knee of the true front with the locked knee_weights = [32, 1]."""
        P = self.P_star_min
        if len(P) == 1:
            return 0
        native = np.column_stack([-P[:, 0], P[:, 1]])
        omin, omax = native.min(axis=0), native.max(axis=0)
        rng = omax - omin
        rng[rng == 0] = 1
        norm = (native - omin) / rng
        norm[:, 0] = 1 - norm[:, 0]                       # invert coverage
        d = np.sqrt((KNEE_WEIGHTS * norm ** 2).sum(axis=1))
        return int(d.argmin())

    @property
    def stratum(self):
        if self.m >= 3:
            return 'A'
        elif self.m == 2:
            return 'B'
        return 'C'


# dominance utils, minimization everywhere
def non_dominated_mask(F):
    """Mask of non-dominated rows, F is (n, 2) minimization."""
    F = np.atleast_2d(np.asarray(F, dtype=float))
    n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # j dominates i  iff  all(F[j] <= F[i]) and any(F[j] < F[i])
        le = np.all(F <= F[i], axis=1)
        lt = np.any(F < F[i], axis=1)
        if np.any(le & lt):
            keep[i] = False
    return keep


def unique_rows(F, decimals=12):
    F = np.atleast_2d(np.asarray(F, dtype=float))
    if len(F) == 0:
        return F
    _, idx = np.unique(np.round(F, decimals), axis=0, return_index=True)
    return F[np.sort(idx)]


def extract_front(F):
    """
    Applied the same way to all four. MOEA/D returns its whole population incl.
    dominated members (30 vs NSGA-II's 6), without this it gets scored on junk.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    if len(F) == 0:
        return F
    return unique_rows(F[non_dominated_mask(F)])


# indicators
def hypervolume(F_norm, ref=REF_POINT):
    """Exact 2-D hypervolume by sweep. F_norm: (n,2) minimization, in [0,1]^2."""
    F = np.atleast_2d(np.asarray(F_norm, dtype=float))
    F = F[np.all(F <= ref, axis=1)]           # drop points outside the ref cuboid
    if len(F) == 0:
        return 0.0
    F = unique_rows(F[non_dominated_mask(F)])
    F = F[np.argsort(F[:, 0])]                # ascending f1
    hv, prev_f2 = 0.0, ref[1]
    for f1, f2 in F:
        if f2 < prev_f2:
            hv += (ref[0] - f1) * (prev_f2 - f2)
            prev_f2 = f2
    return float(hv)


def igd(S_norm, P_norm):
    """IGD: mean distance from each true front point to S."""
    S = np.atleast_2d(S_norm); P = np.atleast_2d(P_norm)
    if len(S) == 0:
        return float('inf')
    d = np.linalg.norm(P[:, None, :] - S[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def igd_plus(S_norm, P_norm):
    """IGD+ (Ishibuchi et al., 2015) -- weakly Pareto-compliant, unlike IGD."""
    S = np.atleast_2d(S_norm); P = np.atleast_2d(P_norm)
    if len(S) == 0:
        return float('inf')
    diff = np.maximum(S[None, :, :] - P[:, None, :], 0.0)   # (|P|, |S|, 2)
    d = np.sqrt((diff ** 2).sum(axis=2))
    return float(d.min(axis=1).mean())


def gd(S_norm, P_norm):
    """GD: mean distance from each point of S to the true front."""
    S = np.atleast_2d(S_norm); P = np.atleast_2d(P_norm)
    if len(S) == 0:
        return float('inf')
    d = np.linalg.norm(S[:, None, :] - P[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def spacing(S_norm):
    """Schott (1995). Undefined for |S| < 2 -> NaN (frequency of this is reported)."""
    S = np.atleast_2d(S_norm)
    n = len(S)
    if n < 2:
        return float('nan')
    d = np.abs(S[:, None, :] - S[None, :, :]).sum(axis=2)   # L1
    np.fill_diagonal(d, np.inf)
    di = d.min(axis=1)
    return float(np.sqrt(((di.mean() - di) ** 2).sum() / (n - 1)))


def set_coverage(A, B):
    """C(A,B) = fraction of B weakly dominated by A (Zitzler & Thiele 1998). Not symmetric, report both ways."""
    A = np.atleast_2d(A); B = np.atleast_2d(B)
    if len(B) == 0:
        return float('nan')
    if len(A) == 0:
        return 0.0
    covered = 0
    for b in B:
        le = np.all(A <= b, axis=1)
        lt = np.any(A < b, axis=1)
        if np.any(le & lt) or np.any(np.all(A == b, axis=1)):
            covered += 1
    return covered / len(B)


# FE counter, this is what enforces the budget (D-3)
class EvalLog:
    """
    Logs every new candidate evaluated, in order. 1 FE = 1 new candidate.
    Re-evaluating an unchanged individual doesn't count: pymoo never does it, and
    cell 20 recomputes ~3,100 times per run just for logging.
    """

    def __init__(self, enum, budget=FE_BUDGET):
        self.enum = enum
        self.budget = budget
        self.rows = []        # index into enum.F_native, one per new candidate

    def log(self, solution):
        self.rows.append(self.enum.row_of(solution))
        return self.enum.F_native[self.rows[-1]]

    def log_many(self, X):
        for x in X:
            self.log(x)

    @property
    def n_eval(self):
        return len(self.rows)

    def cumulative_front_min(self):
        """Non-dominated subset of the first `budget` evals, so parity is exact."""
        rows = self.rows[:self.budget]
        if not rows:
            return np.empty((0, 2))
        return extract_front(self.enum.F_min[np.array(rows)])


# shared init (D-1), same P0 for all four
def make_initial_population(cache, seed, pop_size=POP_SIZE, target_length=K_SUMMARY):
    """
    cell 20's initialize_population(), but seeded. All four get this P0 for a given
    (article, seed): no init head start, and the paired design makes Wilcoxon signed-rank legit.
    """
    rng = np.random.default_rng(seed)
    n = cache.n
    init_scores = cache.node_importance * cache.position_bonus
    s = init_scores.sum()
    probs = init_scores / s if s > 0 else np.ones(n) / n

    pop = []
    for _ in range(pop_size):
        sol = np.zeros(n, dtype=int)
        num_to_select = max(1, rng.integers(max(1, target_length - 1), target_length + 2))
        chosen = rng.choice(n, size=min(int(num_to_select), n), replace=False, p=probs)
        sol[chosen] = 1
        pop.append(repair_solution(sol, cache, target_length))
    return np.array(pop, dtype=int)


# MO-WOA, cell 20 logic rewired to the shared layer
def _crowding_distance(objectives):
    """Copied from cell 20's crowding_distance()."""
    objectives = np.atleast_2d(objectives)
    n_sol, n_obj = objectives.shape
    if n_sol <= 2:
        return np.full(n_sol, np.inf)
    dist = np.zeros(n_sol)
    for m in range(n_obj):
        order = np.argsort(objectives[:, m])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        rng_m = objectives[order[-1], m] - objectives[order[0], m]
        if rng_m == 0:
            continue
        for i in range(1, n_sol - 1):
            dist[order[i]] += (objectives[order[i + 1], m] -
                               objectives[order[i - 1], m]) / rng_m
    return dist


def _pareto_indices(objectives):
    """cell 20's compute_pareto_front in native form (coverage max, redundancy min)."""
    objectives = np.atleast_2d(objectives)
    idx = []
    for i, o1 in enumerate(objectives):
        dominated = False
        for j, o2 in enumerate(objectives):
            if i != j and (o2[0] >= o1[0] and o2[1] <= o1[1] and
                           (o2[0] > o1[0] or o2[1] < o1[1])):
                dominated = True
                break
        if not dominated:
            idx.append(i)
    return idx


def _sigmoid(x):
    """cell 20's sigmoid incl. the [-500, 500] clip. MOPSO uses it too."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def run_mowoa(enum, seed, pop_size=POP_SIZE, max_iter=MAX_ITER, budget=FE_BUDGET):
    """
    MO-WOA as in cell 20 of [v1_4_1]: binary WOA update, CD leader on the front,
    70/30 leader following, leader kept (so 29 new per gen), repair after every
    update, knee with [32, 1]. FEs: 30 + 50 x 29 = 1480 exactly.
    """
    cache = enum.cache
    rng = np.random.default_rng(seed)
    n = cache.n
    log = EvalLog(enum, budget)

    population = make_initial_population(cache, seed, pop_size)
    log.log_many(population)                                   # 30 FEs

    gen_history = []          # per-generation population objectives (minimization)

    def objs(pop):            # lookups, not counted as FEs (see EvalLog)
        return np.array([enum.F_of(s) for s in pop])

    def update_position_woa(current, guide, a, pop):
        new_pos = np.zeros(n, dtype=int)
        p = rng.random()
        b = 1.0
        A_coeff = 2 * a * rng.random() - a
        C_coeff = 2 * rng.random()

        if p < 0.5:
            if abs(A_coeff) < 1:                                # shrinking encircling
                ref = guide
            else:                                               # random search
                ref = pop[rng.integers(len(pop))]
            for i in range(n):
                D = abs(C_coeff * ref[i] - current[i])
                val = ref[i] - A_coeff * D
                new_pos[i] = 1 if rng.random() < _sigmoid(val) else 0
        else:                                                   # spiral bubble-net
            l = rng.uniform(-1, 1)
            for i in range(n):
                Dp = abs(guide[i] - current[i])
                val = Dp * np.exp(b * l) * np.cos(2 * np.pi * l) + guide[i]
                new_pos[i] = 1 if rng.random() < _sigmoid(val) else 0

        if not new_pos.any():
            new_pos[rng.integers(n)] = 1
        return repair_solution(new_pos, cache)

    for it in range(max_iter):
        a = 2.0 - it * (2.0 / max_iter)
        O = objs(population)
        gen_history.append(np.column_stack([-O[:, 0], O[:, 1]]))

        pidx = _pareto_indices(O)
        if len(pidx) == 0:
            pidx = list(range(len(population)))

        if len(pidx) > 1:
            cd = _crowding_distance(O[pidx])
            inf_mask = np.isinf(cd)
            if np.any(inf_mask):
                lead_in_pareto = rng.choice(np.where(inf_mask)[0])
            else:
                lead_in_pareto = int(np.argmax(cd))
            leader_idx = pidx[lead_in_pareto]
        else:
            leader_idx = pidx[0]
        leader = population[leader_idx]

        new_pop = []
        for i, sol in enumerate(population):
            if i == leader_idx:
                new_pop.append(sol)                             # leader kept
            else:
                guide = leader if rng.random() < 0.7 else population[rng.choice(pidx)]
                child = update_position_woa(sol, guide, a, population)
                log.log(child)                                  # 29 FEs per generation
                new_pop.append(child)
        population = np.array(new_pop)

    O = objs(population)
    gen_history.append(np.column_stack([-O[:, 0], O[:, 1]]))
    F_min_terminal = np.column_stack([-O[:, 0], O[:, 1]])

    # knee point, same as cell 20
    pidx = _pareto_indices(O)
    knee_row = None
    if len(pidx) > 0:
        P = O[pidx]
        omin, omax = P.min(axis=0), P.max(axis=0)
        rngv = omax - omin
        rngv[rngv == 0] = 1
        norm = (P - omin) / rngv
        norm[:, 0] = 1 - norm[:, 0]
        d = np.sqrt((KNEE_WEIGHTS * norm ** 2).sum(axis=1))
        knee_row = enum.row_of(population[pidx[int(d.argmin())]])

    return dict(
        name='MO-WOA',
        terminal_front=extract_front(F_min_terminal),
        cumulative_front=log.cumulative_front_min(),
        gen_history=gen_history,
        n_eval=log.n_eval,
        knee_row=knee_row,
        eval_rows=log.rows,
    )


# binary MOPSO-CD, hand-coded (pymoo's MOPSO is real-valued only)
def run_mopso(enum, seed, pop_size=POP_SIZE, budget=FE_BUDGET,
              w=0.729, c1=1.49445, c2=1.49445, v_max=6.0, archive_size=POP_SIZE):
    """
    Binary MOPSO-CD (Raquel & Naval, 2005). Used instead of the grid MOPSO
    (Coello Coello, Pulido & Lechuga, 2004) because it picks leaders by crowding
    distance like MO-WOA, so the update equation is the only thing that differs:
        particle : w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        whale    : shrinking encircling + spiral bubble-net + random search
    Same sigmoid, repair, init and budget. Clerc constriction defaults, not tuned (D-4).
    """
    cache = enum.cache
    rng = np.random.default_rng(seed)
    n = cache.n
    log = EvalLog(enum, budget)

    B = make_initial_population(cache, seed, pop_size)          # same P0 as MO-WOA (D-1)
    log.log_many(B)                                            # 30 FEs

    # continuous position seeded from the binary start (sigmoid(+/-2) ~ 0.88 / 0.12)
    X = np.where(B == 1, 2.0, -2.0).astype(float)
    V = np.zeros((pop_size, n))

    F = np.array([enum.F_of(b) for b in B])                    # native (cov, red)
    F_min = np.column_stack([-F[:, 0], F[:, 1]])

    pbest_B, pbest_Fmin = B.copy(), F_min.copy()
    arch_B, arch_Fmin = _archive_update(B, F_min, np.empty((0, n), int),
                                        np.empty((0, 2)), archive_size)
    gen_history = [F_min.copy()]

    while log.n_eval < budget:
        # leader: binary tournament on crowding distance
        cd = _crowding_distance(arch_Fmin) if len(arch_Fmin) > 1 else np.array([np.inf])
        for i in range(pop_size):
            if len(arch_B) == 1:
                g = arch_B[0]
            else:
                a1, a2 = rng.integers(len(arch_B), size=2)
                g = arch_B[a1] if cd[a1] >= cd[a2] else arch_B[a2]

            r1, r2 = rng.random(n), rng.random(n)
            V[i] = (w * V[i]
                    + c1 * r1 * (np.where(pbest_B[i] == 1, 2.0, -2.0) - X[i])
                    + c2 * r2 * (np.where(g == 1, 2.0, -2.0) - X[i]))
            V[i] = np.clip(V[i], -v_max, v_max)
            X[i] = X[i] + V[i]
            b = (rng.random(n) < _sigmoid(X[i])).astype(int)   # same sigmoid as MO-WOA
            B[i] = repair_solution(b, cache)                   # same repair as MO-WOA

        F = np.array([enum.F_of(b) for b in B])
        F_min = np.column_stack([-F[:, 0], F[:, 1]])
        log.log_many(B)                                        # 30 new candidates
        gen_history.append(F_min.copy())

        # pbest by dominance, tie -> keep new with prob 0.5
        for i in range(pop_size):
            new, old = F_min[i], pbest_Fmin[i]
            if np.all(new <= old) and np.any(new < old):
                pbest_B[i], pbest_Fmin[i] = B[i].copy(), new
            elif not (np.all(old <= new) and np.any(old < new)):
                if rng.random() < 0.5:
                    pbest_B[i], pbest_Fmin[i] = B[i].copy(), new

        arch_B, arch_Fmin = _archive_update(B, F_min, arch_B, arch_Fmin, archive_size)

    return dict(
        name='MOPSO',
        terminal_front=extract_front(arch_Fmin),               # native output is the archive
        cumulative_front=log.cumulative_front_min(),
        gen_history=gen_history,
        n_eval=log.n_eval,
        knee_row=None,
        eval_rows=log.rows,
    )


def _archive_update(B, F_min, arch_B, arch_Fmin, cap):
    """Merge population into the external archive; prune by crowding distance."""
    allB = np.vstack([arch_B, B]) if len(arch_B) else B.copy()
    allF = np.vstack([arch_Fmin, F_min]) if len(arch_Fmin) else F_min.copy()

    keep = non_dominated_mask(allF)
    allB, allF = allB[keep], allF[keep]

    _, uidx = np.unique(np.round(allF, 12), axis=0, return_index=True)
    allB, allF = allB[np.sort(uidx)], allF[np.sort(uidx)]

    if len(allF) > cap:                                        # prune the most crowded
        cd = _crowding_distance(allF)
        keep_idx = np.argsort(-cd)[:cap]
        allB, allF = allB[keep_idx], allF[keep_idx]
    return allB, allF


# pymoo adapters for NSGA-II and MOEA/D
def _pymoo_imports():
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair
    from pymoo.core.sampling import Sampling
    from pymoo.core.callback import Callback
    return Problem, Repair, Sampling, Callback


def make_pymoo_components(enum, seed, budget=FE_BUDGET, pop_size=POP_SIZE):
    Problem, Repair, Sampling, Callback = _pymoo_imports()
    cache = enum.cache
    log = EvalLog(enum, budget)

    class SummaryProblem(Problem):
        def __init__(self):
            # needs vtype=bool
            super().__init__(n_var=cache.n, n_obj=2, xl=0, xu=1, vtype=bool)

        def _evaluate(self, X, out, **kw):
            Xi = np.asarray(X).astype(int)
            F = np.array([enum.F_of(x) for x in Xi])
            log.log_many(Xi)                                   # every row = 1 new FE
            out["F"] = np.column_stack([-F[:, 0], F[:, 1]])    # minimization form

    class SummaryRepair(Repair):
        def _do(self, problem, X, **kw):
            # has to return bool. with int64, BitflipMutation flips via bitwise NOT
            # (~0 = -1, ~1 = -2), the encoding gets corrupted and the run still
            # finishes without error (cardinality sums go negative). hit this in testing
            return np.array([repair_solution(x, cache) for x in np.asarray(X).astype(int)]).astype(bool)

    class SharedSampling(Sampling):
        """D-1: every algorithm gets MO-WOA's importance-biased P0."""
        def _do(self, problem, n_samples, **kw):
            P0 = make_initial_population(cache, seed, pop_size)
            if n_samples <= len(P0):
                return P0[:n_samples].astype(bool)
            extra = make_initial_population(cache, seed + 100000,
                                            n_samples - len(P0))
            return np.vstack([P0, extra]).astype(bool)

    class GenCallback(Callback):
        def __init__(self):
            super().__init__()
            self.gen_history = []
        def notify(self, algorithm):
            self.gen_history.append(algorithm.pop.get("F").copy())

    return SummaryProblem(), SummaryRepair(), SharedSampling(), GenCallback(), log


def run_pymoo(enum, seed, which, budget=FE_BUDGET, pop_size=POP_SIZE):
    """
    which: 'NSGA-II' | 'MOEA/D'
    pymoo only checks termination at generation boundaries, so it can't stop at
    exactly 1480 (NSGA-II with ('n_evals', 1500) used 1524). Peers overshoot to
    the next boundary, so the surplus goes to them, never MO-WOA. The cumulative
    front is cut at the first 1480 evals, so parity is exact there.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.termination.max_eval import MaximumFunctionCallTermination
    from pymoo.optimize import minimize

    problem, repair, sampling, cb, log = make_pymoo_components(enum, seed, budget, pop_size)

    common = dict(sampling=sampling, crossover=TwoPointCrossover(),
                  mutation=BitflipMutation(), repair=repair)

    if which == 'NSGA-II':
        algo = NSGA2(pop_size=pop_size, eliminate_duplicates=True, **common)
    elif which == 'MOEA/D':
        ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size - 1)
        algo = MOEAD(ref_dirs=ref_dirs, n_neighbors=15, **common)
    else:
        raise ValueError(which)

    res = minimize(problem, algo, MaximumFunctionCallTermination(budget),
                   seed=seed, callback=cb, verbose=False, save_history=False)

    # MOEA/D returns its whole population, extract_front cleans that up (same for all four)
    F_terminal = np.atleast_2d(res.pop.get("F"))

    return dict(
        name=which,
        terminal_front=extract_front(F_terminal),
        cumulative_front=log.cumulative_front_min(),
        gen_history=cb.gen_history,
        n_eval=log.n_eval,
        knee_row=None,
        eval_rows=log.rows,
    )


# one (article, seed) -> all four algorithms
ALGORITHMS = ['MO-WOA', 'NSGA-II', 'MOPSO', 'MOEA/D']


def run_all(enum, seed):
    return {
        'MO-WOA':  run_mowoa(enum, seed),
        'NSGA-II': run_pymoo(enum, seed, 'NSGA-II'),
        'MOPSO':   run_mopso(enum, seed),
        'MOEA/D':  run_pymoo(enum, seed, 'MOEA/D'),
    }


def score_run(result, enum, tier='terminal'):
    """Compute every indicator for one run. tier: 'terminal' | 'cumulative' (D-2)."""
    F = result[f'{tier}_front']
    S = enum.normalize(F) if len(F) else np.empty((0, 2))
    P = enum.P_star_norm

    hv = hypervolume(S)
    out = {
        'algorithm': result['name'],
        'tier': tier,
        'HV': hv,
        'HV_pct': 100.0 * hv / enum.hv_star if enum.hv_star > 0 else float('nan'),
        'IGD': igd(S, P),
        'IGD+': igd_plus(S, P),
        'GD': gd(S, P),
        'Spacing': spacing(S),
        'cardinality': len(S),
        'n_eval': result['n_eval'],
    }
    # only possible because we know the true front
    if len(F):
        pf = unique_rows(enum.P_star_min)
        hits = sum(1 for f in F if np.any(np.all(np.isclose(pf, f), axis=1)))
        out['front_recovery'] = hits / len(pf)
        cov_ext = enum.F_min[enum.true_cov_extreme_idx]
        out['found_cov_extreme'] = bool(np.any(np.all(np.isclose(F, cov_ext), axis=1)))
    else:
        out['front_recovery'] = 0.0
        out['found_cov_extreme'] = False
    return out


def hv_convergence(result, enum):
    """HV-vs-generation, normalized as a % of the true optimal HV."""
    return [100.0 * hypervolume(enum.normalize(extract_front(Fg))) / enum.hv_star
            if enum.hv_star > 0 else float('nan')
            for Fg in result['gen_history']]


# MO-WOA+, the fixed variant. canonical MO-WOA above stays untouched.
# pilot diagnosis (real data):
#   retention: terminal 75.70% vs cumulative 87.28% HV, 11.58 pts thrown away.
#     no archive, generational replacement drops what it found
#   search: S-transfer saturates. as a->0, A->0, val -> leader[i] in {0,1},
#     sigmoid(val) -> {0.50, 0.73}, so every bit is a near coin-flip and
#     repair_solution() collapses the offspring onto greedy top-3. diversity dies,
#     search only adds ~4.6 pts in 1480 FEs
# fixes, each switchable for the ablation:
#   use_archive: external archive, cap 30 like MOPSO, non-dominated, crowding-pruned
#     [Coello Coello et al., 2004]
#   transfer='V': V-shaped transfer [Mirjalili & Lewis, 2013; binary WOA: Hussien
#     et al., 2019]. step magnitude -> flip prob of the parent's bit, step->0 keeps the parent

def _v_transfer(step):
    """V-shaped transfer: |tanh(step)|. V(0)=0 (keep parent bit), V(large)->1."""
    return np.abs(np.tanh(step))


def _nds_crowding_select(B_all, F_all, k):
    """(mu+lambda) environmental selection: non-dominated sorting + crowding (NSGA-II)."""
    Nn = len(F_all)
    remaining = np.arange(Nn)
    chosen = []
    while len(chosen) < k and len(remaining) > 0:
        mask = non_dominated_mask(F_all[remaining])
        front = remaining[mask]
        if len(chosen) + len(front) <= k:
            chosen.extend(front.tolist())
            remaining = remaining[~mask]
        else:
            cd = _crowding_distance(F_all[front])
            need = k - len(chosen)
            chosen.extend(front[np.argsort(-cd)[:need]].tolist())
            break
    chosen = np.array(chosen[:k])
    return B_all[chosen], F_all[chosen]


def run_mowoa_final(enum, seed, transfer='V', guide_mode='per_whale_archive',
                    survival='elitist', use_archive=True,
                    ls_probes=0, ls_mode='directed',
                    pop_size=POP_SIZE, max_iter=MAX_ITER, budget=FE_BUDGET,
                    archive_size=POP_SIZE):
    """
    Ablation-switchable MO-WOA+. Update equations, repair, init (D-1), budget (D-3)
    and knee weights are the same as cell 20. Switches:
      transfer   : 'S' (cell 20 sigmoid) | 'V' (V-shaped flip)
      guide_mode : 'single_leader' (cell 20) | 'per_whale_archive' (MOPSO-CD gbest rule)
      survival   : 'generational' (cell 20) | 'elitist' (mu+lambda NDS+crowding, like NSWOA)
      use_archive: external archive, cap 30, crowding-pruned
    S / single_leader / generational / no archive reproduces run_mowoa() bit-exactly (G7b).
    """
    cache = enum.cache
    rng = np.random.default_rng(seed)
    n = cache.n
    log = EvalLog(enum, budget)

    population = make_initial_population(cache, seed, pop_size)
    log.log_many(population)

    def objs(pop):
        return np.array([enum.F_of(s) for s in pop])

    F = objs(population)
    F_min = np.column_stack([-F[:, 0], F[:, 1]])
    gen_history = [F_min.copy()]

    arch_B = np.empty((0, n), dtype=int); arch_F = np.empty((0, 2))
    if use_archive:
        arch_B, arch_F = _archive_update(population, F_min, arch_B, arch_F, archive_size)

    def offspring_of(current, guide, a, pop):
        p = rng.random(); b = 1.0
        A_coeff = 2 * a * rng.random() - a
        C_coeff = 2 * rng.random()
        if transfer == 'S':
            new_pos = np.zeros(n, dtype=int)
            if p < 0.5:
                ref = guide if abs(A_coeff) < 1 else pop[rng.integers(len(pop))]
                for i in range(n):
                    D = abs(C_coeff * ref[i] - current[i])
                    val = ref[i] - A_coeff * D
                    new_pos[i] = 1 if rng.random() < _sigmoid(val) else 0
            else:
                l = rng.uniform(-1, 1)
                for i in range(n):
                    Dp = abs(guide[i] - current[i])
                    val = Dp * np.exp(b * l) * np.cos(2 * np.pi * l) + guide[i]
                    new_pos[i] = 1 if rng.random() < _sigmoid(val) else 0
        else:
            new_pos = current.copy()
            if p < 0.5:
                ref = guide if abs(A_coeff) < 1 else pop[rng.integers(len(pop))]
                for i in range(n):
                    D = abs(C_coeff * ref[i] - current[i])
                    if rng.random() < _v_transfer(A_coeff * D):
                        new_pos[i] = 1 - new_pos[i]
            else:
                l = rng.uniform(-1, 1)
                for i in range(n):
                    Dp = abs(guide[i] - current[i])
                    if rng.random() < _v_transfer(Dp * np.exp(b * l) * np.cos(2 * np.pi * l)):
                        new_pos[i] = 1 - new_pos[i]
        if not new_pos.any():
            new_pos[rng.integers(n)] = 1
        return repair_solution(new_pos, cache)

    def ls_probe():
        """1-swap probe on an archive member. Cardinality stays at k so repair never
        fires, it's the only move that skips repair's importance bias.
        random: uniform swap. directed: drop the selected sentence most similar to
        the others, add one of the 3 least similar unselected ones."""
        if len(arch_B) == 0:
            return None
        if len(arch_B) == 1:
            base = arch_B[0]
        else:
            cdp = _crowding_distance(arch_F)
            a1, a2 = rng.integers(len(arch_B), size=2)
            base = arch_B[a1] if cdp[a1] >= cdp[a2] else arch_B[a2]
        x = np.asarray(base).astype(int).copy()
        sel = np.where(x == 1)[0]
        unsel = np.where(x == 0)[0]
        if len(sel) == 0 or len(unsel) == 0:
            return None
        if ls_mode == 'random':
            out_i = rng.choice(sel)
            in_i = rng.choice(unsel)
        else:
            contrib = np.array([cache.adj[i, sel].sum() for i in sel])
            out_i = sel[int(np.argmax(contrib))] if rng.random() < 0.7 else rng.choice(sel)
            remaining = sel[sel != out_i]
            scores = cache.adj[unsel][:, remaining].sum(axis=1)
            k = min(3, len(unsel))
            in_i = unsel[rng.choice(np.argsort(scores)[:k])]
        x[out_i] = 0
        x[in_i] = 1
        return x

    for it in range(max_iter):
        a = 2.0 - it * (2.0 / max_iter)
        F = objs(population)
        F_min = np.column_stack([-F[:, 0], F[:, 1]])
        pidx = _pareto_indices(F)
        if len(pidx) == 0:
            pidx = list(range(len(population)))

        # leader / guides
        if guide_mode == 'single_leader':
            if len(pidx) > 1:
                cd = _crowding_distance(F[pidx])
                inf_mask = np.isinf(cd)
                lp = (rng.choice(np.where(inf_mask)[0]) if np.any(inf_mask)
                      else int(np.argmax(cd)))
                leader_idx = pidx[lp]
            else:
                leader_idx = pidx[0]
            leader = population[leader_idx]
        else:
            leader_idx = 0
            leader = None

        def draw_guide():
            if guide_mode == 'per_whale_archive' and use_archive and len(arch_B) > 0:
                if len(arch_B) == 1:
                    return arch_B[0]
                cd = _crowding_distance(arch_F)
                a1, a2 = rng.integers(len(arch_B), size=2)
                return arch_B[a1] if cd[a1] >= cd[a2] else arch_B[a2]
            if guide_mode == 'per_whale_archive':          # archive off: pop front
                if len(pidx) == 1:
                    return population[pidx[0]]
                cd = _crowding_distance(F[pidx])
                a1, a2 = rng.integers(len(pidx), size=2)
                return population[pidx[a1] if cd[a1] >= cd[a2] else pidx[a2]]
            return leader

        # 29 offspring, leader stays in its slot like cell 20 (needed for the
        # bit-exact rng stream at baseline switches)
        new_pop = []
        children = []
        if ls_probes > 0 and use_archive and len(arch_B) > 0:
            nonleader = [i for i in range(len(population)) if i != leader_idx]
            probe_slots = set(rng.choice(nonleader,
                                         size=min(ls_probes, len(nonleader)),
                                         replace=False).tolist())
        else:
            probe_slots = set()
        for i, sol in enumerate(population):
            if i == leader_idx:
                new_pop.append(sol)
                continue
            child = None
            if i in probe_slots:
                child = ls_probe()
            if child is None:
                g = draw_guide() if rng.random() < 0.7 else population[rng.choice(pidx)]
                child = offspring_of(sol, g, a, population)
            log.log(child)
            new_pop.append(child)
            children.append(child)
        children = np.array(children)
        Fc = objs(children)
        Fc_min = np.column_stack([-Fc[:, 0], Fc[:, 1]])

        # survival
        if survival == 'generational':
            population = np.array(new_pop)
        else:                                              # elitist (mu+lambda)
            B_all = np.vstack([population, children])
            F_all = np.vstack([F_min, Fc_min])
            population, _ = _nds_crowding_select(B_all, F_all, pop_size)

        Fq = objs(population)
        Fq_min = np.column_stack([-Fq[:, 0], Fq[:, 1]])
        gen_history.append(Fq_min.copy())
        if use_archive:
            arch_B, arch_F = _archive_update(children, Fc_min, arch_B, arch_F,
                                             archive_size)

    # terminal output + knee
    if use_archive:
        F_terminal = arch_F; knee_B, knee_F = arch_B, arch_F
    else:
        Fq = objs(population)
        F_terminal = np.column_stack([-Fq[:, 0], Fq[:, 1]])
        pidx = _pareto_indices(Fq)
        knee_B, knee_F = population[pidx], F_terminal[pidx]

    knee_row = None
    if len(knee_F) > 0:
        native = np.column_stack([-knee_F[:, 0], knee_F[:, 1]])
        omin, omax = native.min(axis=0), native.max(axis=0)
        rngv = omax - omin; rngv[rngv == 0] = 1
        norm = (native - omin) / rngv
        norm[:, 0] = 1 - norm[:, 0]
        d = np.sqrt((KNEE_WEIGHTS * norm ** 2).sum(axis=1))
        knee_row = enum.row_of(knee_B[int(d.argmin())])

    return dict(name=f"MO-WOA+[{transfer},{guide_mode[:8]},{survival[:4]}"
                     f"{',arch' if use_archive else ''}]",
                terminal_front=extract_front(F_terminal),
                cumulative_front=log.cumulative_front_min(),
                gen_history=gen_history, n_eval=log.n_eval,
                knee_row=knee_row, eval_rows=log.rows)
