import numpy as np
import pandas as pd



class Species:
    """
    Represents a single fish species with age-structured dynamics.
    Parameters loaded from tables.
    """
    def __init__(self, name, param_tables):
        self.name = name
        gr = param_tables['growth']
        ages = gr['Age'].astype(int).values
        self.ages = ages
        self.n_ages = len(ages)
        self.natural_mortality = param_tables['mortality'][name].astype(float).values
        self.maturity = param_tables['maturity'][name].astype(float).values
        self.weight_at_age = param_tables['growth'][name].astype(float).values
        self.selectivity = param_tables['selectivity'][name].astype(float).values
        econ = param_tables['economics']
        self.price = float(econ.loc[econ['Species']==name, 'Price_per_ton'])
        self.cost = float(econ.loc[econ['economics']['Species']==name, 'Cost_per_unit_F']) if 'Cost_per_unit_F' in econ else float(econ.loc[econ['Species']==name, 'Cost_per_unit_F'])
        rec = param_tables['recruitment']
        sub = rec[rec['Species']==name].iloc[0]
        if sub['Function']=='Beverton-Holt':
            self.recruitment_params = {
                'type': 'BH', 'alpha': float(sub['alpha']),
                'beta': float(sub['beta']), 'gamma': float(sub.get('gamma',0.0))
            }
        else:
            self.recruitment_params = {
                'type': 'loglinear', 'a': float(sub['a']),
                'b': float(sub['b']), 'temp_coef': float(sub['temp_coef'])
            }

    def spawning_stock_biomass(self, numbers):
        return np.sum(numbers * self.weight_at_age * self.maturity)

    def recruitment(self, ssb, env_idx):
        p = self.recruitment_params
        if p['type']=='BH':
            R = (p['alpha']*ssb)/(1+(p['alpha']/p['beta'])*ssb)
            return R * np.exp(p['gamma']*env_idx)
        else:
            return np.exp(p['a'] + p['b']*np.log(ssb) + p['temp_coef']*env_idx)


class FisheriesModel:
    """
    Multi-species age-structured fisheries model with optimization.
    """
    def __init__(self, species_list, years, climate):
        self.species = {s.name:s for s in species_list}
        self.years = np.array(years)
        self.climate = climate
        self.n_at_age = {n:np.zeros((len(years), sp.n_ages)) for n,sp in self.species.items()}
        self.catch_at_age = {n:np.zeros((len(years), sp.n_ages)) for n,sp in self.species.items()}

    def set_initial(self, init):
        for n,nums in init.items(): self.n_at_age[n][0,:]=nums

    def run(self, F):
        self.F = F
        for t,year in enumerate(self.years[:-1]):
            for n,sp in self.species.items():
                N = self.n_at_age[n][t]
                env = self.climate.get(year,0.0)
                ssb = sp.spawning_stock_biomass(N)
                R = sp.recruitment(ssb, env)
                Z = sp.natural_mortality + sp.selectivity*F[n]
                Nn = np.zeros_like(N)
                Nn[1:-1] = N[:-2]*np.exp(-Z[:-2])
                Nn[-1] = N[-2]*np.exp(-Z[-2]) + N[-1]*np.exp(-Z[-1])
                Nn[0] = R
                self.n_at_age[n][t+1]=Nn
                C = (sp.selectivity*F[n]/Z*(1-np.exp(-Z))*N*sp.weight_at_age)
                self.catch_at_age[n][t]=C

    def revenue(self, species):
        catch = self.catch_at_age[species].sum(axis=1)[:-1]
        return catch * self.species[species].price

    def optimize_cod_revenue(self, F_range=np.linspace(0,1,51)):
        """
        Find uniform F_at_age for Cod maximizing mean annual revenue over horizon.
        """
        best = {'F':None,'rev':-np.inf}
        # keep Herring F fixed
        F_base = {n:np.array([0.2]*self.species[n].n_ages) for n in self.species}
        for f in F_range:
            F_test = F_base.copy()
            F_test['Cod'] = np.full(self.species['Cod'].n_ages, f)
            # reset populations
            model = FisheriesModel(list(self.species.values()), self.years, self.climate)
            model.set_initial({n:self.n_at_age[n][0] for n in self.species})
            model.run(F_test)
            rev = model.revenue('Cod').mean()
            if rev>best['rev']:
                best={'F':f,'rev':rev}
        return best

if __name__=='__main__':
    doc='Including climate change scenarios in an ecological 26 May 25.dotx'
    tables=load_tables_from_dotx(doc)
    keys=['climate','growth','mortality','maturity','selectivity','recruitment','economics']
    pt=dict(zip(keys,tables))
   
    # build climate
    cdf=pt['climate']; years=np.arange(2020,2081)
    row=cdf[cdf['Climate']=='RCP 4.5'].iloc[0]
    interp=np.interp(years,[2040,2080],[float(row['2040_MMSY']),float(row['2080_MMSY'])])
    climate=dict(zip(years,interp))
    
    # init species & model
    h=Species('Herring',pt); c=Species('Cod',pt)
    model=FisheriesModel([h,c],years,climate)
    init={'Herring':np.array(pt['growth']['Initial_numbers_Herring'],float),
          'Cod':    np.array(pt['growth']['Initial_numbers_Cod'],float)}
    model.set_initial(init)
    # optimize
    result=model.optimize_cod_revenue()
    print(f"Optimal uniform F for Cod: {result['F']:.3f}, mean annual revenue: {result['rev']:.2f}")
