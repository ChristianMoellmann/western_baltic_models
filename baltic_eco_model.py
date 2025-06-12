import numpy as np
from scipy.optimize import minimize

class BalticEcoEconModel:
    '''
    Age-structured ecological-economic model for three Baltic Sea species:
    cod (C), sprat (S), herring (H).
    '''
    def __init__(self, params):
        # Unpack parameters
        self.ages = np.arange(1, params['A_max']+1)
        self.dt = 1.0  # annual time step
        self.r = params['discount_rate']

        # Biological parameters per species
        self.w = params['weight_at_age']      # weight_i[sp][age]
        self.gamma = params['maturity']       # maturity_i[sp][age]
        self.q = params['catchability']       # catchability_i[sp][age]
        self.M_nat = params['M_natural']      # natural mortality funcs
        self.pred_mort = params['predation']  # predation mortality dict

        # Economic parameters per species (to be filled)
        self.p_bar = params['p_bar']   # price intercept
        self.eta = params['eta']       # demand elasticity
        self.c0 = params['c0']         # cost scale
        self.chi = params['chi']       # cost exponent

        # Recruitment scenarios
        self.recruit = params['recruitment']  # functions mapping stock to recruits

    def compute_SSB(self, N):
        '''Spawning stock biomass for each species. N[sp][age]'''        
        ssb = {}
        for sp in N:
            ssb[sp] = np.sum(self.w[sp] * self.gamma[sp] * N[sp])
        return ssb

    def compute_X(self, N):
        '''Efficient biomass for cost calculations.'''
        X = {}
        for sp in N:
            X[sp] = np.sum(self.w[sp] * self.q[sp] * N[sp])
        return X

    def inverse_demand(self, H, sp):
        '''p = p_bar * H^-eta'''        
        return self.p_bar[sp] * np.maximum(H,1e-6) ** (-self.eta[sp])

    def cost(self, H, X, sp):
        '''C = c0 * X^-chi * H'''        
        return self.c0[sp] * X**(-self.chi[sp]) * H

    def step(self, N, F):
        '''Advance one year: N and F are dicts indexed by species.'''        
        N_next = {sp: np.zeros_like(N[sp]) for sp in N}
        ssb = self.compute_SSB(N)['C']  # cod SSB influences predation

        # Recruitment at age 1 (or age2 for cod)
        N_next['C'][1] = self.recruit['C'](ssb)
        N_next['S'][0] = self.recruit['S']()
        N_next['H'][0] = self.recruit['H']()

        # Aging + survival + fishing
        for sp in ['C','S','H']:
            for a in range(1, len(self.ages)):
                Z = self.M_nat[sp](a+1)  + self.pred_mort.get((sp,a+1), 0)
                surv = np.exp(-Z) * np.exp(-F[sp] * self.q[sp][a])
                N_next[sp][a] += N[sp][a-1] * surv
            # Plus group
            Z_last = self.M_nat[sp](len(self.ages))  + self.pred_mort.get((sp,len(self.ages)), 0)
            surv_last = np.exp(-Z_last) * np.exp(-F[sp] * self.q[sp][-1])
            N_next[sp][-1] += N[sp][-1] * surv_last
        return N_next

    def simulate(self, F_traj, N0, T):
        '''Simulate for T years with fishing mortalities F_traj[t][sp].'''
        N = {sp: np.array(N0[sp], dtype=float) for sp in N0}
        results = {'N': [], 'H': [], 'profit': [], 'CS': []}
        for t in range(T):
            F = F_traj[t]
            X = self.compute_X(N)
            H = {sp: F[sp] * X[sp] for sp in X}  # catch
            p = {sp: self.inverse_demand(H[sp], sp) for sp in H}
            C = {sp: self.cost(H[sp], X[sp], sp) for sp in H}
            profit = {sp: p[sp]*H[sp] - C[sp] for sp in H}
            CS = {sp: self.p_bar[sp]/(1 - self.eta[sp]) * H[sp]**(1 - self.eta[sp]) - p[sp]*H[sp] for sp in H}

            results['N'].append({sp:N[sp].copy() for sp in N})
            results['H'].append(H)
            results['profit'].append(profit)
            results['CS'].append(CS)
            N = self.step(N, F)
        return results

    def welfare(self, results):
        '''Compute discounted welfare from simulation results.'''
        W = 0.0
        for t, (cs, prof) in enumerate(zip(results['CS'], results['profit'])):
            flow = sum(cs[sp] + prof[sp] for sp in cs)
            W += flow / ((1 + self.r)**t)
        return W


if __name__ == "__main__":
    # Biological parameters from Voss et al. (2022) Supplementary Material
    cod_M_nat = np.array([0.0, 0.65, 0.65, 0.65, 0.65, 0.215, 0.215, 0.215])
    herr_M_nat = np.array([0.177, 0.162, 0.1567, 0.1473, 0.1369, 0.1317, 0.1289, 0.1294])
    sprat_M_nat = np.array([0.213, 0.22, 0.2219, 0.2193, 0.2187, 0.2187, 0.215, 0.215])

    pred_herr = np.array([1.41e-3, 8.06e-4, 5.75e-4, 4.95e-4, 4.67e-4, 4.32e-4, 3.85e-4, 2.87e-4])
    pred_sprat = np.array([1.91e-3, 1.22e-3, 1.07e-3, 1.02e-3, 9.82e-4, 9.82e-4, 9.82e-4, 0.0])

    weight_C = np.array([0.0095, 0.057, 0.145, 0.261, 0.392, 0.526, 0.654, 0.772])
    weight_H = np.array([0.0111, 0.019, 0.0279, 0.0284, 0.0398, 0.0408, 0.0432, 0.0521])
    weight_S = np.array([0.0047, 0.009, 0.0096, 0.0098, 0.011, 0.0117, 0.0117, 0.0111])

    mat_C = np.array([0.0774, 0.361, 0.726, 0.903, 0.962, 0.983, 0.991, 0.995])
    mat_H = np.array([0.0, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0])
    mat_S = np.array([0.17, 0.93, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    q_C = np.array([0.00033, 0.087, 0.545, 0.861, 0.956, 0.979, 0.979, 0.986])
    q_H = np.array([0.182, 0.377, 0.622, 0.818, 0.9241, 0.971, 0.989, 0.996])
    q_S = np.array([0.182, 0.377, 0.622, 0.818, 0.924, 0.971, 0.989, 0.996])

    # Recruitment averages 2014-2018 from Table S1
    rec_vals = {'C': 61.6, 'H': 22003, 'S': 98862}

    params = {
        'A_max': 8,
        'discount_rate': 0.05,  # placeholder
        'weight_at_age': {'C': weight_C, 'H': weight_H, 'S': weight_S},
        'maturity': {'C': mat_C, 'H': mat_H, 'S': mat_S},
        'catchability': {'C': q_C, 'H': q_H, 'S': q_S},
        'M_natural': {
            'C': lambda a: cod_M_nat[min(a-1, len(cod_M_nat)-1)],
            'H': lambda a: herr_M_nat[a-1],
            'S': lambda a: sprat_M_nat[a-1],
        },
        'predation': {
            **{('H', i+1): pred_herr[i] for i in range(len(pred_herr))},
            **{('S', i+1): pred_sprat[i] for i in range(len(pred_sprat))},
        },
        # Economic parameters to be filled from main publication
        'p_bar': {'C': 27.43, 'H': 1.38, 'S': 7.26},
        'eta': {'C': 0.65, 'H': 0.33, 'S': 0.60},
        'c0': {'C': 6.60, 'H': 0.93, 'S': 26.67},
        'chi': {'C': 0.43, 'H': 0.21, 'S': 0.70},
        'recruitment': {
            'C': lambda ssb: rec_vals['C'],
            'H': lambda : rec_vals['H'],
            'S': lambda : rec_vals['S'],
        }
    }

    model = BalticEcoEconModel(params)
    print('Parameter dictionary constructed from supplemental data.')
