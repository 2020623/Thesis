
from   utilFunctions.structtype import structtype
import sys
import numpy as np
import copy
import math


def df_simplex(*args):
        
    np.random.seed(1)

    if (len(args) < 3):
        sys.exit('At least four inputs are required.')
    
    if (len(args) > 4):
        sys.exit('At most five inputs are required.')
    
    # declare Vars
    obj = args[0]
    A   = args[1]
    y   = args[2]
    if len(args) == 4:
        opts = args[3]
    
    # set options
    eps_opt = 0e0
    max_n_f = 100*(A.shape[0] + 1)
    max_it  =   math.inf
    min_f   = 0e0
    f = []
    alpha_ini = max(5e-1, eps_opt)
    is_alpha_ini_given = False
    verbosity = True
    if (len(args) == 4):
        if ( np.isscalar(opts) or not type(opts) !='utilFunctions.structtype.structtype'):
            sys.exit('The third input (which is optional) must be a structure.')
        
        opts_field = ['eps_opt', 'max_n_f', 'min_f', 'f0','verbosity'] 
        for i in range(len(opts_field)):
            if str(opts_field[i]) =='eps_opt':
                eps_opt = opts.eps_opt
                if (not np.isscalar(eps_opt) or not isnumeric(eps_opt) or not np.isreal(eps_opt) or eps_opt < 0e0):
                    sys.exit('In the options, eps_opt must be a non-negative number.')
                
            elif str(opts_field[i]) == 'max_n_f':
                max_n_f = math.floor(opts.max_n_f)
                if (not isnumeric(max_n_f) or not np.isreal(max_n_f) or not np.isscalar(max_n_f) or max_n_f<1e0):
                    sys.exit('In the options, max_n_f must be a number greater than or equal to 1.')
                
            elif str(opts_field[i]) == 'max_it':
                max_it = math.floor(opts.max_it)
                if (not np.isscalar(max_it) or not isnumeric(max_it) or not np.isreal(max_it) or max_it<1e0):
                    sys.exit('In the options, max_it must be a number greater than or equal to 1.')
                
            elif str(opts_field[i]) == 'min_f':
                min_f = opts.min_f
                if (not np.isscalar(min_f) or not isnumeric(min_f) or not np.isreal(min_f)):
                    sys.exit('In the options, min_f must be a real number.')
                
            elif str(opts_field[i]) == 'f0':
                f = opts.f0
                if (not np.isscalar(f) or not isnumeric(f) or not np.isreal(f)):
                    sys.exit('In the options, f0 must be a real number.')
                
            elif str(opts_field[i]) == 'alpha_ini':
                alpha_ini = opts.alpha_ini
                if (not np.isscalar(alpha_ini) or not isnumeric(alpha_ini) or not np.isreal(alpha_ini) or alpha_ini<0e0):
                    sys.exit('In the options, alpha_ini'' must be a non-negative number.')
                
                is_alpha_ini_given = True
            elif str(opts_field[i]) == 'verbosity':
                verbosity = opts.verbosity
                if (not np.isscalar(verbosity) or not (type(verbosity) == type(True))):
                    sys.exit('In the options, verbosity must be a logical.')
            else:
                sys.exit('Not valid field name in the structure of options.')

    n = A.shape[1]
    b_sampling     = [0]*(2*n)
    alpha_sampling = [0]*(2*n)
    ind_sampling   = [False]*(2*n)

    ind = np.random.permutation(n).tolist() # shuffle variables 

    ind_i = -1
    tau = 9E-1
    j = np.argmax(y)
    
    x = A @ y
    if not f :
        f = obj(x.tolist())
        n_f = 1
    else:
        n_f = 0
    
    if (verbosity):
        print('=====================================================================================')
        print('DF-SIMPLEX starts:')
        print('==================')
        print('it = ',0,', f = ',"{:.4e}".format(f))
    
    # search directions will be normalized
    # line search parameters
    if (not is_alpha_ini_given):
        doubleItem = np.eye(n, dtype=np.double)[j]
        npMatrix = (np.identity(n) - doubleItem)
        normtrail = max(np.linalg.norm(A @ np.array(npMatrix).T, axis = 0))
        alpha_max = alpha_ini* normtrail
    else:
        alpha_max = alpha_ini
    
    alpha_vec = [max(alpha_max,eps_opt)]*(n) # vector of initial stepsizes
    gamma = 1e-6
    theta = 5e-1 # stepsize reduction factor
    delta = 5e-1 # reciprocal of the stepsize expansion factor
    
    compute_j = False
    allow_stop = (alpha_max<=eps_opt)
    skip_d = [False] * (2*n)

    sampling = structtype(b = [], v_d = None, alpha = None,j = None )
    df_simplex_info = structtype(y = None, n_f = None, it = None, flag = None )

    if (f <= min_f):
        it = 0
        indicesM = [i for i, x in enumerate(ind_sampling) if x == True]
        trailVector = []
        for IteM in indicesM:
            trailVector.append(b_sampling[IteM])
        sampling.b = trailVector
        sampling.v_d = []
        sampling.alpha = []
        sampling.j = j
        flag = 3
        if (verbosity):
            print('target objective value obtained')
        df_simplex_info.y = y
        df_simplex_info.n_f = n_f
        df_simplex_info.it = it
        df_simplex_info.flag = flag
        return
    
    if (max_n_f <= 0e0):
        it = 0
        indicesM = [i for i, x in enumerate(ind_sampling) if x == True]
        trailVector = []
        for IteM in indicesM:
            trailVector.append(b_sampling[IteM])
        sampling.b = trailVector
        sampling.v_d = []
        sampling.alpha = []
        sampling.j = j
        flag = 1
        if (verbosity):
            print('maximum number of function evaluations reached')
        df_simplex_info.y = y
        df_simplex_info.n_f = n_f
        df_simplex_info.it = it
        df_simplex_info.flag = flag
        return
    
    if (max_it <= 0e0):
        it = 0
        indicesM = [i for i, x in enumerate(ind_sampling) if x == True]
        trailVector = []
        for IteM in indicesM:
            trailVector.append(b_sampling[IteM])
        sampling.b = trailVector
        sampling.v_d = []
        sampling.alpha = []
        sampling.j = j
        flag = 2
        if (verbosity):
            print('maximum number of iterations reached')
        df_simplex_info.y = y
        df_simplex_info.n_f = n_f
        df_simplex_info.it = it
        df_simplex_info.flag = flag
        return
    
    it = 1
    while True:
        
        if (n_f >= max_n_f):
            flag = 1
            break
        # select index i
        if (ind_i < (n - 1)):
            ind_i += 1
            if ind_i == 46:
                stop = 1
            i = ind[ind_i]
            if (i == j):
                ind_i += 1
                if (ind_i < (n - 1)):
                    i = ind[ind_i]
                else:
                    compute_j = True
        else:
            compute_j = True
        
        if (compute_j):
            alpha_vec[j] = min(alpha_vec)
            # check stopping condition
            alpha_max = max(alpha_vec)
            if (allow_stop and alpha_max<=eps_opt):
                flag = 0
                break
            if (it >= max_it):
                flag = 2
                break
            if (verbosity):
                print('it = ',it,', f = ',"{:.4e}".format(f),', n_f = ',n_f,', alpha_max = ',"{:.4e}".format(alpha_max))
            
            it = it + 1

            ind = np.random.permutation(n).tolist() # shuffle variables 
            
            # select a new index j
            j_max = np.argmax(y)
            if (y[j] < tau*y[j_max]):
                j = j_max
            compute_j = False
            
            # select index i
            ind_i = 0
            i = ind[ind_i]
            if (i == j):
                ind_i = ind_i + 1
                i = ind[ind_i]
            
            allow_stop = (alpha_max <= eps_opt)
            
        if (y[i] > 0e0 or y[j] > 0e0):  # so that at least one direction between
                                        # (e_i-e_j) and (e_j-e_i) is feasible
            linesearch_i = True
            expansion_i  = False
            if (y[i] == 0e0):
                if ( not skip_d[2*i]):
                    h1 = i
                    h2 = j
                    which_dir_i = True # d = e_i - e_j
                    first_linesarch_i = False
                else:
                    linesearch_i = False
                
            elif (y[j] == 0e0):
                if ( not skip_d[2*i + 1]):
                    h1 = j
                    h2 = i
                    which_dir_i = False # d = e_j - e_i
                    first_linesarch_i = False
                else:
                    linesearch_i = False
            
            else: # randomly choose the first direction to use
                if ( np.random.random() < 5e-1):  
                    if ( not skip_d[2*i]):
                        h1 = i
                        h2 = j
                        which_dir_i = True # d = e_i - e_j
                        first_linesarch_i = True
                    elif (not skip_d[2*i + 1]):
                        h1 = j
                        h2 = i
                        which_dir_i = False # d = e_j - e_i
                        first_linesarch_i = False
                    else:
                        linesearch_i = False
                else:
                    if ( not skip_d[2*i + 1]):
                        h1 = j
                        h2 = i
                        which_dir_i = False # d = e_j - e_i
                        first_linesarch_i = True
                    elif (not skip_d[2*i]):
                        h1 = i
                        h2 = j
                        which_dir_i = True # d = e_i - e_j
                        first_linesarch_i = False
                    else:
                        linesearch_i = False
            
            if (linesearch_i):
                d_x = np.subtract(A[:,h1], A[:,h2])
                norm_d_x = np.linalg.norm(d_x[np.newaxis].T)  
                if (norm_d_x > 0e0):
                    d_i = 1e0/norm_d_x
                    d_x = d_x/norm_d_x
                    ind_i_sampling = (2*i + 1) - (1 if which_dir_i else 0)
                else:
                    linesearch_i = False
            
            # backtracking procedure
            while (linesearch_i and n_f < max_n_f):
                alpha_max_feas_i = norm_d_x * y[h2]
                alpha_trial = min(alpha_max_feas_i, alpha_vec[i])
                s_i = alpha_trial * d_i
                y_trial = copy.deepcopy(y) 
                y_trial[h1] = y_trial[h1] + s_i
                y_trial[h2] = y_trial[h2] - s_i
                x_trial = x + (alpha_trial * d_x)
                f_trial = obj(x_trial.tolist())
                n_f = n_f + 1
                if (f_trial <= f-gamma*alpha_trial*alpha_trial):
                    expansion_i = True
                    linesearch_i = False
                else:
                    b_sampling[ind_i_sampling] = f_trial - f
                    alpha_sampling[ind_i_sampling] = alpha_trial
                    ind_sampling[ind_i_sampling] = True
                    if (theta*alpha_vec[i] <= eps_opt):
                        skip_d[ind_i_sampling] = True
                    
                    if (first_linesarch_i and n_f < max_n_f):
                        h3 = h1
                        h1 = h2
                        h2 = h3
                        d_x = -d_x
                        which_dir_i = not which_dir_i
                        ind_i_sampling = (2*i + 1) - (1 if which_dir_i else 0)
                        first_linesarch_i = False
                    else:
                        linesearch_i = False
            
            # expansion procedure
            # we now produce a new point and we start a new collection
            # of samples that first includes the point where we come from
            # and the point not accepted in the expansion (if any)
            if (expansion_i):
                allow_stop = False
                if (alpha_trial<alpha_max_feas_i and n_f<max_n_f and f_trial>min_f):
                    y_next = copy.deepcopy(y_trial)
                    x_next = copy.deepcopy(x_trial)
                    f_next = f_trial
                    f_prev = f
                    alpha_prev = alpha_trial
                    first_expansion = True
                    while (expansion_i and alpha_trial<alpha_max_feas_i and n_f<max_n_f):
                        alpha_trial = min(alpha_max_feas_i,alpha_trial/delta)
                        s_i = alpha_trial*d_i
                        y_trial[h1] = y[h1] + s_i
                        y_trial[h2] = y[h2] - s_i
                        x_trial = x + alpha_trial*d_x
                        f_trial = obj(x_trial.tolist())
                        n_f = n_f + 1
                        if (f_trial <= f-gamma*alpha_trial*alpha_trial):
                            f_prev = f_next
                            y_next = copy.deepcopy(y_trial)
                            x_next = copy.deepcopy(x_trial)
                            f_next = f_trial
                            alpha_prev = alpha_trial
                            first_expansion = False
                        elif (f_trial <= min_f):
                            expansion_i = False
                            y_next = copy.deepcopy(y_trial)
                            x_next = copy.deepcopy(x_trial)
                            f_next = f_trial
                            alpha_prev = alpha_trial
                            flag = 3
                        else:
                            expansion_i = False
                            b_sampling[ind_i_sampling] = f_trial - f_next
                            b_sampling[(2*i + 1) - (0 if which_dir_i else 1)] = f_prev - f_next
                            alpha_sampling[ind_i_sampling] = alpha_trial - alpha_prev
                            if (first_expansion):
                                alpha_sampling[(2*i + 1)-(0 if which_dir_i else 1)] = alpha_prev
                            else:
                                alpha_sampling[(2*i + 1)-(0 if which_dir_i else 1)] = (1e0-delta)*alpha_prev
                            
                            ind_sampling = [False] * (2*n)
                            ind_sampling[ind_i_sampling] = True
                            ind_sampling[(2*i + 1)-(0 if which_dir_i else 1)] = True
                    
                    if ( not ind_sampling[ind_i_sampling]): # if this occurs, it means that the stepsize has been expanded,
                                                            # but the previous while loop ended because
                                                            # alpha_trial=alpha_max_feas_i, or n_f=n_f_max, or f_trial<=min_f
                        b_sampling[(2*i + 1)-(0 if which_dir_i else 1)] = f_prev - f_next
                        alpha_sampling[(2*i + 1)-(0 if which_dir_i else 1)] = (1e0-delta)*alpha_trial
                        ind_sampling = [False for i in range(2*n)]
                        ind_sampling[(2*i + 1)-(0 if which_dir_i else 1)] = True
                    
                    y = copy.deepcopy(y_next)
                    x = copy.deepcopy(x_next)
                    f = f_next
                    alpha_vec[i] = alpha_prev
                else:   # if this occurs, it means that the stepsize has not been expanded
                        # because alpha_trial=alpha_max_feas_i, n_f=n_f_max or f_trial>min_f
                    b_sampling[(2*i + 1)-(0 if which_dir_i else 1)] = f - f_trial
                    alpha_sampling[(2*i + 1)-(0 if which_dir_i else 1)] = alpha_trial
                    ind_sampling = [False for i in range(2*n)]
                    ind_sampling = [False] * (2*n)
                    ind_sampling[(2*i + 1)-(0 if which_dir_i else 1)] = True
                    y = copy.deepcopy(y_trial)
                    x = copy.deepcopy(x_trial)
                    f = f_trial
                    alpha_vec[i] = max(alpha_trial,eps_opt)
                
                if (f <= min_f):
                    flag = 3
                    break
                
                skip_d = [False for i in range(2*n)] 
                skip_d = [False] * (2*n) 
                skip_d[(2*i + 1)-(0 if which_dir_i else 1)] = True
            else:
                if (alpha_vec[i]> eps_opt):
                    alpha_vec[i] = max(theta*alpha_vec[i],eps_opt)
                    indicesM = [i for i, x in enumerate(ind_sampling) if x == True]
                    for IteM in indicesM:
                        skip_d[IteM] = False
                    skip_d[(2*i + 1)-(0 if which_dir_i else 1)] = False
            
        else:
            if (alpha_vec[i]> eps_opt):
                alpha_vec[i] = max(theta*alpha_vec[i],eps_opt)
                skip_d[(2*i + 1)] = False
                skip_d[2*i] = False
    
    indicesM = [i for i, x in enumerate(ind_sampling) if x == True]
    trailVectorB = []
    for IteM in indicesM:
        trailVectorB.append(b_sampling[IteM])
    sampling.b = trailVectorB
    v_d_temp_1 = np.copy((np.argwhere(ind_sampling)[:,0] + 1.5)/2e0)
    v_d_temp_2 = np.copy(np.round(v_d_temp_1) * np.sign(np.round(v_d_temp_1) - v_d_temp_1))
    sampling.v_d = np.copy(np.array((np.abs(v_d_temp_2) * np.sign(v_d_temp_2)), dtype="int"))
    trailVectorAl = []
    for IteM in indicesM:
        trailVectorAl .append(alpha_sampling[IteM])
    sampling.alpha =  [X if X>0 else 0 for X in trailVectorAl]
    sampling.j = int(j)
    
    if (verbosity): 
        print('it = ',it,', f = ',"{:.4e}".format(f),', n_f = ',n_f,', alpha_max = ',"{:.4e}".format(alpha_max))
        if (flag == 0):
            print('optimality condition satisfied with the desired tolerance')
        elif (flag == 1):
            print('maximum number of function evaluations reached')
        elif (flag == 2):
            print('maximum number of iterations reached')
        elif (flag == 3):
            print('target objective value obtained')
        else:
            print('maximum cpu time exceeded')
    
    df_simplex_info.y = y
    df_simplex_info.n_f = n_f
    df_simplex_info.it = it
    df_simplex_info.flag = flag
    
    return x, f, df_simplex_info, sampling

def isnumeric(obj):
    try:
        obj + 0
        return True
    except TypeError:
        return False
    


