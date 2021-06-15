#Following Bayesian Optimization searches for points with lower confidence band

from mint.mint import *
import matplotlib.pyplot as plt
import os
import time

import torch
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils.transforms import normalize, unnormalize, standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

class Botorch(Minimizer):
    def __init__(self):
        super(Botorch, self).__init__()


    def preprocess(self):
        #aqcuisition function
        self.acq_name = 'UCB' #'UCB', 'qNEI', or 'qEI'

        #reshape boundary condition
        self.boundsBO = torch.tensor(self.bounds).T

        #batch size for optimizing acquisition function
        self.BATCH_SIZE = 1

        #number of trials and iterations of the optimization
        self.N_TRIALS = 1
        self.N_BATCH = self.max_iter

        self.debug = True


    def acq_function(self, model, train_x, train_obj):
        """
        choose an acquisition function
        check https://botorch.org/api/acquisition.html for other options
        """
        if self.acq_name == 'UCB':
            return UpperConfidenceBound(model, beta=0.1)
        elif self.acq_name == 'qNEI':
            qmc_sampler = SobolQMCNormalSampler(num_samples=250)
            return qNoisyExpectedImprovement(model=model, X_baseline=train_x,
                                             sampler=qmc_sampler, prune_baseline=True)
        elif self.acq_name == 'qEI':
            return qExpectedImprovement(model=model, best_f=train_obj.max().detach().numpy())
        else:
            raise TypeError('wrong name for the acquisition function')

    def reshape_training_data(self, x, y):
        """
        reshape the input and objective for optimization.
        x: normalized to [0, 1]
        """
        train_obj = torch.tensor(y).T
        train_obj.requires_grad=True

        x = normalize(x, self.boundsBO)
        train_x = x.repeat(train_obj.shape[0],1)
        train_x.requires_grad=True

        return train_x, train_obj

    def initialize_model(self, train_x, train_obj, state_dict=None):
        """
        define models for objectives
        """
        model = SingleTaskGP(train_x, train_obj ).to(train_x)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        #load state dict if it is possed
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model

    def optimize_function(self, acq_func):
        """
        Optimize the acquisition function, and returns a new candidate
        """
        candidates, val = optimize_acqf(
            acq_function=acq_func,
            bounds=normalize(self.boundsBO, self.boundsBO),
            q=self.BATCH_SIZE,
            num_restarts=5,
            raw_samples=20,#, used for initialization heuristic
            options={"batch_limit":5, "maxiter":200},
        )
        pred_x = candidates.detach()

        print(' expected value: ', val.detach().numpy(), end=' ')
        return pred_x


    def minimize(self, get_objective, x):
        """
        Main optimization code.
        """

        for trial in range(1, self.N_TRIALS + 1):

            print(f"\nTrial {trial:>2} of {self.N_TRIALS} ")
            x_best_all, obj_best_all = [], []
            x_all, obj_all = [], []

            #get initial training data and initialize the model
            obj = self.target.objective_acquisition
            print('initial parameter: ', x, ', ', obj)
            train_x, train_obj = self.reshape_training_data(torch.tensor(x), obj)
            train_obj_st = standardize(train_obj.detach())
            mll, model = self.initialize_model(train_x, train_obj_st)
            print('model: '+str(model))

            for iteration in range(1, self.N_BATCH + 1):
                print(iteration,'/',self.N_BATCH,'th BATCH')
                # fit the model
                fit_gpytorch_model(mll)

                #optimize the acquizition function and get a new cantidate point
                pred_x = self.optimize_function(self.acq_function(model, train_x, train_obj_st))

                #get new objectives of the new point from the device
                pred_x = unnormalize(pred_x, bounds=self.boundsBO)
                print('next candidate: ', pred_x[0].numpy())
                obj = get_objective(pred_x[0].detach().tolist())

                try:
                    #save results
                    x_all.append(pred_x[0].tolist())
                    obj_all.append(obj.mean())

                    # update training points
                    new_x, new_obj = self.reshape_training_data(pred_x, obj)
                    train_x = torch.cat([train_x, new_x])
                    train_obj = torch.cat([train_obj, new_obj])

                    # reinitialize the models so they are ready for fitting on next iteration
                    #use the current state dict to speed up fitting
                    train_obj_st = standardize(train_obj.detach())
                    mll, model = self.initialize_model(train_x, train_obj_st, model.state_dict())
                    print('model: '+str(model))
                    print('mll: '+str(mll))

                except:
                    pass

            if self.debug == True:
                #get the best value
                x_all = np.array(x_all)
                idx = obj_all.index(max(obj_all))
                x_best = x_all[idx,:]
                obj_best = obj_all[idx]
                print("--")
                print('best estimate: x: ', x_best, ' obj: ', obj_best)
                print("--")
                l1_norm = np.linalg.norm(x_best - self.mi.pvs_optimum_value[0], ord=1)
                print('l1_norm: ', l1_norm)

                #save the best value
                x_best_all.append(x_best)
                obj_best_all.append(obj_best)

        #best value after all trials
        x_best_all = np.array(x_best_all)
        idx_b = obj_best_all.index(max(obj_best_all))
        x_final = x_best_all[idx_b,:]
        obj_final = obj_best_all[idx_b]

        if self.debug == True:
            print("--")
            print('FINAL best estimate: x: ', x_final,
                  ' obj: ', obj_final)
            print("--")

            plt.scatter(x_all[:,0], x_all[:,1], c=obj_all)
            plt.xlabel('dev 1')
            plt.ylabel('dev 2')
            plt.title(str(len(x_all))+' explored points')
            plt.savefig(os.getcwd()+'/explored_points.png')

        return x_final


