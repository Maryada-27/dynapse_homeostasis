'''
The brain of code
'''
import numpy as np
import lib.dynapse2_util  as DYN2util
from network_config import netConfig


class Adam:
    '''
    Adam implementation, NOT IN USE
    '''
    def __init__(self, beta1=0.9, beta2=0.999):
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.eps = 1e-8
        self.t = 0

    def __call__(self, delta):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(delta)
            self.v = np.zeros_like(delta)

        self.m = self.m * self.beta1 + (1-self.beta1) * delta
        self.v = self.v * self.beta2 + (1-self.beta2) * delta ** 2
        mhat = self.m / (1-self.beta1 ** self.t)
        vhat = self.v / (1-self.beta2 ** self.t)
        return mhat / (np.sqrt(vhat) + self.eps)


class Momentum:
    '''
    Update with Momentum
    '''
    def __init__(self, momentum, lr):
        self.m = momentum
        self.lr = lr
        self.change = None

    def __call__(self, delta):
        if self.change is None:
            self.change = delta
        self.change = self.change * self.m + delta * (1-self.m)
        return self.change * self.lr


class Homeostasis:
    '''
    Homeostasis rule
    '''
    def __init__(self, learning_rate, E_target, I_target):
        self.lr = learning_rate
        self.E_target = E_target
        self.I_target = I_target
        self.optimizer = Momentum(0.3, learning_rate)
        self.coarse_u_bound = 5
        self.coarse_l_bound = 2#3
        self.fine_u_bound = 250
        self.fine_l_bound = 20
        self.delta_bound = 5

    
    @staticmethod
    def stochastic_rounding(x):
        '''
        Instead of round, use stochastic rounding
        '''
        integer_part = np.floor(x)
        decimal_part = x - integer_part > np.random.random(size=x.shape)
        return (integer_part + decimal_part).astype(int)

    def global_update_weights(self, myConfig, source, target, delta_w):
        '''
        Update weights
        '''
        chip_id = netConfig.allocated_chips[target]# destination
        core_id = netConfig.allocated_cores[target]
        w_name = netConfig.connectivity[source][target]['w_name']
        weight = DYN2util.get_parameter(myConfig.chips[chip_id].cores[core_id].parameters, w_name)
        fineW = weight[1]
        delta_w = min(self.delta_bound, abs(delta_w)) * np.sign(delta_w)
        coarseW = weight[0]
        print(f'{DYN2util.bcolors.OKBLUE}{source=},{target=} :: {delta_w=}{DYN2util.bcolors.ENDC}')
    
        if fineW + delta_w < self.fine_l_bound :
            print(f'{DYN2util.bcolors.WARNING}Need to switch coase scale as well{DYN2util.bcolors.ENDC}')

            coarse_updated  = max(coarseW-1,self.coarse_l_bound)
            if coarse_updated == coarseW: 
                fine_updated = self.fine_l_bound
            else:
                fine_overflow = (fineW -self.fine_l_bound) + delta_w
                fine_updated = self.fine_u_bound + fine_overflow
            print(f'Setting for {chip_id=} {core_id=} ({coarse_updated},{fine_updated})')
            DYN2util.set_parameter(myConfig.chips[chip_id].cores[core_id].parameters,w_name,coarse_updated,fine_updated)
        elif fineW + delta_w > self.fine_u_bound: 
            print(f'{DYN2util.bcolors.WARNING}Need to switch coase scale as well{DYN2util.bcolors.ENDC}')
            coarse_updated  = min(coarseW+1,self.coarse_u_bound)
            if coarse_updated == coarseW: 
                fine_updated = self.fine_u_bound
            else:
                fine_overflow = fineW + delta_w
                fine_updated = fine_overflow - self.fine_u_bound
                fine_updated = max(self.fine_l_bound,fine_updated)
            print(f'Setting for {chip_id=} {core_id=} ({coarse_updated},{fine_updated})')
            DYN2util.set_parameter(myConfig.chips[chip_id].cores[core_id].parameters,w_name,coarse_updated,fine_updated)
        elif self.fine_l_bound <= fineW + delta_w <= self.fine_u_bound and self.coarse_l_bound <= coarseW <= self.coarse_u_bound:
            fine_updated = fineW + delta_w
            print(f'Setting for {chip_id=} {core_id=} ({coarseW},{fine_updated})')
            DYN2util.update_fine(myConfig.chips[chip_id].cores[core_id].parameters,w_name,fine_updated)
        else:
            print(f'{DYN2util.bcolors.WARNING}No solution{DYN2util.bcolors.ENDC}')
        
    def calculate_update_weights(self, myConfig, AvgE : float, AvgI: float,staticInh : bool = False,staticExc : bool = False):
        '''
        Compute weight update over average firing rates
        '''
        # print(f'{bcolors.OKGREEN} INFO: >> {AvgE=} Hz and {AvgI=} Hz {bcolors.ENDC}')
        if staticExc:
            d_Wee = 0
            d_Wie = - max(1, AvgE) * (self.E_target - AvgE)
        else:
            d_Wee = + max(1, AvgE) * (self.I_target - AvgI)
            d_Wie = - max(1, AvgE) * (self.E_target - AvgE)
        #-------------- Source Inhibitory - ----------------
        if staticInh:
            d_Wei = 0
            d_Wii = 0
        else:
            d_Wei = - max(1, AvgI) * (self.I_target - AvgI)
            d_Wii = + max(1, AvgI) * (self.E_target - AvgE)
        weight_updates = np.multiply(self.lr, [d_Wee, d_Wei, d_Wie, d_Wii])  
        # weight_updates = self.optimizer(np.array([d_Wee,d_Wei,d_Wie,d_Wii]))

        round_weight_updates = self.stochastic_rounding(weight_updates)
        print(f'[Wee,Wei,Wie,Wii]:: = {weight_updates=}, {round_weight_updates=}')
        if np.any(round_weight_updates):
            self.global_update_weights(myConfig, 'Pyr', 'Pyr', round_weight_updates[0])
            self.global_update_weights(myConfig, 'PV', 'Pyr', round_weight_updates[1])
            self.global_update_weights(myConfig, 'Pyr', 'PV', round_weight_updates[2])
            self.global_update_weights(myConfig, 'PV', 'PV', round_weight_updates[3])


    def calculate_update_weights_avg_over_repeat(self, myConfig, AvgE =[], AvgI=[],staticInh : bool = False,staticExc : bool = False):
        '''
        Compute average weight update for multiple repeats and update weights 
        '''
        # print(f'{bcolors.OKGREEN} INFO: >> {AvgE=} Hz and {AvgI=} Hz {bcolors.ENDC}')
        d_Wee, d_Wei, d_Wie, d_Wii = [], [], [], []
        for avg_e, avg_i in zip(AvgE, AvgI):
            if staticExc:
                d_Wee.append(0)
                d_Wie.append(- max(1, avg_e) * (self.E_target - avg_e))
            else:
                d_Wee.append(+ max(1, avg_e) * (self.I_target - avg_i)) 
                d_Wie.append(- max(1, avg_e) * (self.E_target - avg_e))
            if staticInh:
                d_Wii.append(0)
                d_Wei.append(0)
            else:
                d_Wii.append(+ max(1, avg_i) * (self.E_target - avg_e))
                d_Wei.append(- max(1, avg_i) * (self.I_target - avg_i))
        
        weight_updates = np.multiply(self.lr, [np.mean(d_Wee), np.mean(d_Wei), np.mean(d_Wie), np.mean(d_Wii)])
        round_weight_updates = self.stochastic_rounding(weight_updates)
        print(f'[Wee,Wei,Wie,Wii]:: = {weight_updates=}, {round_weight_updates=}')
        if np.any(round_weight_updates):
            self.global_update_weights(myConfig, 'Pyr', 'Pyr', round_weight_updates[0])
            self.global_update_weights(myConfig, 'PV', 'Pyr', round_weight_updates[1])
            self.global_update_weights(myConfig, 'Pyr', 'PV', round_weight_updates[2])
            self.global_update_weights(myConfig, 'PV', 'PV', round_weight_updates[3])

    def calculate_update_weights_OLH(self, myConfig, AvgE =[], AvgI=[]):
        '''
        Owen Mackwood, Laura B Naumann, Henning Sprekeler
        DOI: 10.7554/eLife.59715 
        Compute average weight update for multiple repeats and update weights 
        '''
        # print(f'{bcolors.OKGREEN} INFO: >> {AvgE=} Hz and {AvgI=} Hz {bcolors.ENDC}')
        d_Wee, d_Wei, d_Wie, d_Wii = [], [], [], []
        for avg_e, avg_i in zip(AvgE, AvgI):
            d_Wie.append(max(1, avg_e) * (avg_e - self.E_target ))
            d_Wei.append(max(1, avg_i) * (avg_e - self.E_target))  # ideally it should be local
            d_Wii.append(0)
            d_Wee.append(0)

        weight_updates = np.multiply(self.lr, [np.mean(d_Wee), np.mean(d_Wei), np.mean(d_Wie), np.mean(d_Wii)])
        round_weight_updates = self.stochastic_rounding(weight_updates)
        print(f'[Wee,Wei,Wie,Wii]:: = {weight_updates=}, {round_weight_updates=}')
        if np.any(round_weight_updates):
            self.global_update_weights(myConfig, 'Pyr', 'Pyr', round_weight_updates[0])
            self.global_update_weights(myConfig, 'PV', 'Pyr', round_weight_updates[1])
            self.global_update_weights(myConfig, 'Pyr', 'PV', round_weight_updates[2])
            self.global_update_weights(myConfig, 'PV', 'PV', round_weight_updates[3])