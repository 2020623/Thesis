import math
import sys
import numpy as np
import scipy
import df_simplex as DFs
from utilFunctions.structtype import structtype

def ord(*args): 

    np.random.seed(1)

    if (len(args) < 3):
        sys.exit('At least four inputs are required.')
    
    if (len(args) > 4):
        sys.exit('At most five inputs are required.')

    # declare Vars
    obj = args[0]
    A   = args[1]
    i0  = args[2]
    if len(args) == 4:
        opts = args[3]
	
    # set options
    n_atoms = A.shape[1]
    eps_opt = 0e0
    max_n_f = 100 * (A.shape[0] + 1)
    max_it  =   math.inf
    min_f   = float(0)
    n_atoms_in = n_atoms
    ind_y_A = []
    use_simplex_grad = True
    verbosity = True
    
    if (len(ind_y_A) != 0):
        if (len(ind_y_A) != n_atoms_in):
            sys.exit('In the options, the length of ','set_initial_atoms',' must be equal to ','n_initial_atoms','.')
        
        if (len(np.unique(ind_y_A)) != n_atoms_in):
            sys.exit('In the options, ','set_initial_atoms',' cannnot contain duplicates.')
        
        if (not(any([ind_y_Ai == i0 for ind_y_Ai in ind_y_A]))):
            sys.exit('In the options, ''set_initial_atoms'' must contain ''i0''.')
	
    if (len(ind_y_A) == 0):
        indRrail = np.random.permutation(n_atoms)
        ind_y_A = np.concatenate(([0], indRrail[indRrail != 0]))
        Index = np.argwhere(np.array(ind_y_A) == i0)[0][0]
        ind_y_A[Index] = n_atoms - 1
        ind_y_A[0] = i0
        ind_y_A = ind_y_A.tolist()
    else:
        Index = [i for i in range(len(ind_y_A)) if ind_y_A[i] == i0][0]
        ind_y_A[Index] = ind_y_A[0]
        ind_y_A[0]     = i0 
    
    atoms_in = np.in1d(range(n_atoms), ind_y_A)
    ind_atoms_out = np.where(np.array(atoms_in) == False)[0]
    n_atoms_out = n_atoms - n_atoms_in
    A_k = A[:, ind_y_A]
    y = [0] * (n_atoms_in)
    y[0] = 1e0

    f = obj((A_k @ y))
    n_f = 1

    it = 0
    
    f_sample_best = math.inf
    y_sample_best = [0] * (n_atoms)

    flag = -1
    sol_found_sampling = False
    
    # line search parameters
    mu_hat = max(1e-4, eps_opt)
    gamma = 1e-6
    theta = 5e-1 # stepsize reduction factor
    delta = 5e-1 # reciprocal of the stepsize expansion factor
    
    if (verbosity):
        print('=====================================================================================')
        print('ORD starts:')
        print('===========')
        print('------------------------------------------------------------------------------------')
        print('                                                |     local minimization details    ')
        print('  it         f         n_f   |A^k|     mu^k     |   n_it     n_f       eps     flag')
        print('------------------------------------------------------------------------------------')
        print('  ',0,'  ',"{:.4e}".format(f),'    ',n_f,'   ',n_atoms_in,'              |')
    
    ord_info = structtype(y = None, n_f = None, it = None, flag = None )
    if (f <= min_f):
        x = A[:,i0]
        flag = 3
        if (verbosity):
            print('target objective value obtained')
        
        ord_info.y   = y
        ord_info.n_f = n_f
        ord_info.it  = it
        ord_info.flag = flag
        return
    
    if (max_n_f <= 0e0):
        x = A[:,i0]
        flag = 1
        if (verbosity):
            print('maximum number of function evaluations reached')
        
        ord_info.y = y
        ord_info.n_f = n_f
        ord_info.it = it
        ord_info.flag = flag
        return
    
    if (max_it <= 0e0):
        x = A[:,i0]
        flag = 2
        if (verbosity):
            print('maximum number of iterations reached')
        ord_info.y    = y
        ord_info.n_f  = n_f
        ord_info.it   = it
        ord_info.flag = flag
        return
    
    flag_local = None
    v_added    = None
    sampling        = structtype(b = [], v_d = None, alpha = None,j = None )
    df_simplex_opts = structtype(eps_opt = None, max_n_f = None, min_f = None, f0 = None, verbosity = None)
    
    while True:
        
        #==================================================================
        # OPTIMIZE PHASE
        #==================================================================
        if (n_atoms_in > 1):
            # set DF-SIMPLEX parameters and call the solver
            if (it > 1):
                if (flag_local <= 5e-1) and (not v_added) and (mu_hat <= eps_opt_local):
                    alpha_ini_local = eps_opt_local
                    eps_opt_local = max(25e-2*eps_opt_local, eps_opt)
                    max_n_f_local = min(3*(n_atoms_in+1), max_n_f-n_f)
                else:
                    if (flag_local <= 5e-1):
                        alpha_ini_local = eps_opt_local
                        max_n_f_local = min(5*(n_atoms_in+1),max_n_f-n_f)
                    else:
                        alpha_ini_local = max(sampling.alpha)
                        max_n_f_local = min(2*max_n_f_local,max_n_f-n_f)
                    eps_opt_local = max(75e-2*eps_opt_local,eps_opt)
                df_simplex_opts.alpha_ini = alpha_ini_local
            elif (it == 1):
                eps_opt_local = max(5e-1,eps_opt)
                if (abs(flag_local) <= 5e-1):
                    alpha_ini_local = eps_opt_local
                elif (flag_local >= 5e-1):
                    alpha_ini_local = max(sampling.alpha)
                    max_n_f_local = min(10*(n_atoms_in+1),max_n_f-n_f)
                
                df_simplex_opts.alpha_ini = alpha_ini_local
            else:
                eps_opt_local = eps_opt
                max_n_f_local = min(2*(n_atoms_in+1), max_n_f-n_f)
            if (max_n_f_local > 0):
                df_simplex_opts.eps_opt = eps_opt_local
                df_simplex_opts.max_n_f = max_n_f_local
                df_simplex_opts.min_f = min_f
                df_simplex_opts.f0 = f
                df_simplex_opts.verbosity = True
                [x,f,df_simplex_info,sampling] = DFs.df_simplex(obj, A_k,y,df_simplex_opts)
                y = df_simplex_info.y
                inner_n_f = df_simplex_info.n_f
                inner_it = df_simplex_info.it
                flag_local = df_simplex_info.flag
            else:
                inner_n_f  = 0           
                inner_it   = 0
                flag_local = 1
        else:
            y = 1e0
            inner_it = 0
            inner_n_f = 0
            sampling.b = []
            sampling.v_d = []
            sampling.alpha = []
            flag_local = -1
            if (it>0 and flag_local<=5e-1 and mu_hat<=eps_opt_local):
                eps_opt_local = max(25e-2*eps_opt_local,eps_opt)
            else:
                eps_opt_local = max(75e-2*eps_opt_local,eps_opt)
        
        n_f = n_f + inner_n_f
        it  = it + 1
                
        if (flag_local == 3):
            mu = 0
            flag = 3
            break
        
        #==================================================================
        # REFINE PHASE
        #==================================================================
        
        v_added = False
        mu = 0e0
        if (n_atoms_in < n_atoms):
            max_norm_d_x = 0e0
            while ( not v_added and n_atoms_out>0 and n_f<max_n_f):
                atom_out_to_add_k = np.random.randint(0, n_atoms_out)
                atom_to_add_k = ind_atoms_out[atom_out_to_add_k]
                ind_atoms_out.remove(ind_atoms_out[atom_out_to_add_k])
                n_atoms_out = n_atoms_out - 1
                
                d_x = A[:,atom_to_add_k] - x
                norm_d_x = np.linalg.norm(d_x)
                if (norm_d_x > 0e0):
                    max_norm_d_x = max(max_norm_d_x, np.linalg.norm(d_x))
                    f_trial = obj(x+mu_hat*d_x)
                    n_f = n_f + 1
                    if (f_trial <= f - gamma*mu_hat*mu_hat):
                        mu = mu_hat
                        f_next = f_trial
                        if (n_f < max_n_f):
                            mu_trial = mu
                            expansion = True
                            while (expansion):
                                mu_trial = min(mu_trial/delta,1e0)
                                f_trial = obj(x+mu_trial*d_x)
                                n_f = n_f + 1
                                if (f_trial <= f - gamma*mu_trial*mu_trial):
                                    mu = mu_trial
                                    f_next = f_trial
                                    expansion = (mu_trial < 1e0)
                                else:
                                    expansion = False
                                    if (f_trial < f_sample_best):
                                        f_sample_best = f_trial
                                        dat_sp = np.append((1e0-mu_trial)*np.array(y), mu_trial)
                                        col_sp = np.zeros_like(dat_sp,dtype="int")
                                        row_sp = np.append(ind_y_A, atom_to_add_k)
                                        y_sample_best = np.array(scipy.sparse.csr_matrix((dat_sp, (row_sp, col_sp)), shape=(n_atoms,1)).todense())
                                        y_sample_best = [y_sample_best[i][0] for i in range(y_sample_best.shape[0])]
                                        if (f_sample_best <= min_f):
                                            f = f_sample_best
                                            y = np.copy(y_sample_best)
                                            # x = np.dot(A,y)
                                            x = A @ y
                                            sol_found_sampling = True
                                            flag = 3
                                            break
                        
                        v_added = True
                    elif (f_trial < f_sample_best):
                        f_sample_best = f_trial
                        dat_sp = np.append((1e0-mu_hat)*np.array(y), mu_hat)
                        col_sp = np.zeros_like(dat_sp,dtype="int")
                        row_sp = np.append(ind_y_A, atom_to_add_k)
                        y_sample_best = np.array(scipy.sparse.csr_matrix((dat_sp, (row_sp, col_sp)), shape=(n_atoms,1)).todense()) # (n_atoms,1)
                        y_sample_best = [y_sample_best[i][0] for i in range(y_sample_best.shape[0])]
                        if (f_sample_best <= min_f):
                            f = f_sample_best
                            y = np.copy(y_sample_best)
                            # x = np.dot(A,y)
                            x = A @ y
                            sol_found_sampling = True
                            flag = 3
                            break
                
                if (flag == 3):
                    break
            
            if (flag == 3):
                break
            
            if (not v_added):
                if (mu_hat*max_norm_d_x<=eps_opt and eps_opt_local<=eps_opt and flag_local<=5e-1):
                    flag = 0
                    break
                else:
                    mu_hat = theta*mu_hat
            
        elif (eps_opt_local<=eps_opt and flag_local<=5e-1):
            flag = 0
            break
        
        if (n_f>=max_n_f or it >= max_it or (v_added and f_next<=min_f)):
            if (v_added):
                ind_y_A.append(atom_to_add_k)
                y = ((1e0-mu)*np.array(y)).tolist()
                y.append(mu)
                f = f_next
                n_atoms_in += 1
            
            if (f <= min_f):
                flag = 3
            elif (n_f >= max_n_f):
                flag = 1
            else:
                flag = 2
            break
        

        #==================================================================
        # DROP PHASE
        #==================================================================
        
        if (n_atoms_in>1 and np.any(np.array(y)<=0e0)):
            
            if ((type(use_simplex_grad) == type(True) and use_simplex_grad) or \
                (isnumeric(use_simplex_grad) and (use_simplex_grad>=n_atoms_in+1))):
                
                # compute the simplex gradient for the restriced problem
                # as the least-squares solution g of the linear system M*g = b
                #------------------------------------------------------------------
                # (1) build (part of) the matrix M from the output of DF-SIMPLEX
                #     (also (part of) the vector b is an output of DF-SIMPLEX)
                if (len(sampling.alpha) != 0):
                    M =  Mvalue(n_atoms_in, sampling.v_d)
                    M[:,sampling.j] = -1e0 * np.sign(np.array(sampling.v_d))
                    M =  Mvalue1(sampling.alpha,M, A_k)
                    alpha_approx = max(sampling.alpha)
                else:
                    M = []
                    alpha_approx = 1e-6
                
                # (2) find polling samples (in the transormed space) to be added
                v_d_to_addTr = [not flag for flag in ismember(range(n_atoms_in),np.abs([i - 1 for i in np.abs(np.array(sampling.v_d))]))]
                v_d_to_add1   = [i for i,flag in enumerate(v_d_to_addTr) if flag]
                v_d_to_add   = [i for i in v_d_to_add1 if y[i]<1e0]
                if (n_f + len(v_d_to_add) < max_n_f):
                    if (len(v_d_to_add) != 0):
                        M_to_add = Mvalue2(n_atoms_in, v_d_to_add)
                        M_to_add[:,sampling.j] = -1e0
                        M_to_add = alpha_approx*M_to_add
                        nrm = M_to_add @ A_k.T
                        norm_d_x = [np.linalg.norm(nrm[i,:]) for i in range(nrm.shape[0])]
                        is_norm_d_x_zero = [i<=0e0 for i in norm_d_x]
                        [1e0 if norm_d_x[i]<=0e0 else norm_d_x[i] for i in range(len(norm_d_x))]
                        mylist = []
                        for i in range(M_to_add.shape[1]):
                            mylist.append(np.divide(M_to_add[:,i], np.array(norm_d_x)).tolist())
                        M_to_add = np.array(mylist).T
                    else:
                        M_to_add = -alpha_approx*np.ones(n_atoms_in)
                        norm_d_x = np.linalg.norm(M_to_add,A_k.T)
                        if (norm_d_x > 0e0):
                            M_to_add = np.array([np.divide(M_to_add[:,i], np.array(norm_d_x)).tolist() for i in range(M_to_add.shape[1])])
                            is_norm_d_x_zero = False
                        else:
                            is_norm_d_x_zero = True
                    
                    # (3) compute the objective function at the new points and add these values to b
                    size_M_to_add = M_to_add.shape[0]
                    # b_to_add = [0 for i in range(size_M_to_add)]
                    b_to_add = [0] * (size_M_to_add)
                    for i in range(size_M_to_add):
                        if (not is_norm_d_x_zero[i]):
                            b_to_add[i] = obj((x+ A_k @ M_to_add[i,:].T).tolist()) - f
                            n_f = n_f + 1
                    Mi = M.tolist()
                    M_to_addi = M_to_add.tolist()
                    [Mi.append(M_to_addi[i]) for i in range(len(M_to_addi))]
                    M = np.array(Mi)
                    [sampling.b.append(b_to_add[i]) for i in range(len(M_to_addi))]
                    
                    try:
                        # (4) compute the simplex gradient
                        # g_approx = np.dot(np.linalg.pinv(np.dot(M.T,M)), np.dot(M.T, np.array(sampling.b)))
                        g_approx = np.linalg.pinv(M.T @ M) @ M.T @ np.array(sampling.b)
                        #------------------------------------------------------------------
                        
                        # gx = np.dot(g_approx.T ,y) # approximation of gradient-vector product
                        gx = g_approx.T @ y # approximation of gradient-vector product
                        
                        # (5) find atoms to remove
                        Flags = np.dot(gx,np.ones(n_atoms_in))
                        ind_atoms_to_remove = [i for i,item in enumerate(y) if (item<=0e0 and Flags[i] <= g_approx[i])]
                    
                    except Exception as ME:
                        # something went wrong when computing the simplex gradient
                        print('WARNING: An error occured when computing the simplex gradient ', \
                            ME.args[0] , 'Simplex gradient will not be used in this iteration.')
                        ind_atoms_to_remove = [i for i,item in enumerate(y) if item<=0e0]
                
                elif (n_f < max_n_f):
                    # not enough function evaluations at our disposal to compute the simplex gradient
                    ind_atoms_to_remove = [i for i,item in enumerate(y) if item<=0e0]
                else:
                    # the maximum number of function evaluations is reached
                    if (v_added):
                        x = A_k @ y
                    flag = 1
                    break          
            else:
                ind_atoms_to_remove = [i for i,item in enumerate(y) if item<=0e0]
        
        else:
            ind_atoms_to_remove = []
        
        # add atoms
        if (v_added):
            atoms_in[atom_to_add_k] = True
            A_k = np.append(A_k, A[:,atom_to_add_k][np.newaxis].T,1)
            ind_y_A.append(atom_to_add_k)
            y = ((1e0-mu)*np.array(y)).tolist()
            y.append(mu)
            f = f_next
            n_atoms_in +=1
            
        # remove atoms
        n_atoms_to_remove = len(ind_atoms_to_remove)
        if (n_atoms_to_remove > 0):
            atoms_in = IndicesM(atoms_in, ind_y_A, ind_atoms_to_remove)
            A_k = deleteC(A_k, ind_atoms_to_remove)
            y   = deleteL(y, ind_atoms_to_remove)
            ind_y_A = deleteL(ind_y_A, ind_atoms_to_remove)
            n_atoms_in = n_atoms_in - n_atoms_to_remove
        
        ind_atoms_out = [i for i,flag in enumerate(atoms_in) if not flag]
        n_atoms_out = n_atoms - n_atoms_in
        #==================================================================
                
        # iteration prints
        if (verbosity):
            if len(str(inner_it)) == 1:
                inner_it = '0' + str(inner_it)
            if len(str(inner_n_f)) == 2:
                inner_n_f = '0' + str(inner_n_f)
            print('  ',it,'  ',"{:.4e}".format(f),'  ',n_f,'    ',n_atoms_in,'     ',format(mu, ".4f"),' |    ',inner_it,'     ',inner_n_f,'   ',format(eps_opt_local, ".4f"),'    ',flag_local)
    
    # build x from y
    if (not sol_found_sampling):
        if (f_sample_best < f):
            f = f_sample_best
            y = np.copy(y_sample_best)
            x = A @ y
        else:
            y_temp = np.copy(y)
            y = [0] * (n_atoms)
            y = yMultiSymbols(y, ind_y_A, y_temp)
    
    # final iteration prints
    if (verbosity):
        if len(str(inner_it)) == 1:
            inner_it = '0' + str(inner_it)
        if len(str(inner_n_f)) == 2:
            inner_n_f = '0' + str(inner_n_f)
        print('  ',it,'  ',"{:.4e}".format(f),' ',n_f,'    ',n_atoms_in,'     ',format(mu, ".4f"),' |    ',inner_it,'     ',inner_n_f,'   ',format(eps_opt_local, ".4f"),'    ',flag_local)
    
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
    
    ord_info.y = y
    ord_info.n_f = n_f
    ord_info.it = it
    ord_info.flag = flag
    
    return x, f, ord_info

# ------------ Util Functions ------------------
def ismember(listOne,listTwo):
    flags =[]
    for item in listOne:
        index = np.where(np.array(listTwo)==item)[0]
        if index.size == 0:
            flags.append(False)
        else:
            flags.append(True)
    return flags

def Amatrix(Amat, indices):
    myList1 = []
    myList2 = []
    for i in indices:
        list = Amat[:,i]
        myList1.append(list.tolist())
    for i in range(Amat.shape[0]):
        list = [myList1[j][i] for j in range(Amat.shape[1])]
        myList2.append(list)
    return np.array(myList2)

def isnumeric(obj):
    try:
        obj + 0
        return True
    except TypeError:
        return False

def Mvalue(n_atoms_in_, sampling_v_d):
    arrange  = np.tile(np.arange(1,n_atoms_in_+1), (len(sampling_v_d), 1))
    absValue = np.abs(np.array(sampling_v_d))
    signValue = np.sign(np.array(sampling_v_d))
    list1 = []
    for i in range(arrange.shape[0]):
        list = [1/signValue[i] if arrange[i,j] == absValue[i] else 0 for j in range(arrange.shape[1])]
        list1.append(list)
    return np.array(list1)

def Mvalue1(sampling_alpha,M_, A_k_):
    mat = np.dot(M_,A_k_.T)
    list1 = []
    list2 = []
    norms = []
    for i in range(M_.shape[0]):
        lista = [sampling_alpha[i]* M_[i,j] for j in range(M_.shape[1])]
        list1.append(lista)
    list1 = np.array(list1)
    for i in range(mat.shape[0]):
        nrm = np.linalg.norm(mat[i,:])
        norms.append(nrm)
    for i in range(list1.shape[0]):
        listb = [list1[i,j]/norms[i] for j in range(list1.shape[1])]
        list2.append(listb)
    return np.array(list2)

def Mvalue2(n_atoms_in_, v_d_to_add_):
    arrange   = np.tile(np.arange(1,n_atoms_in_+1), (len(v_d_to_add_), 1))
    absValue  = np.array([i+1 for i in v_d_to_add_]) 
    oensVAL   = - np.ones(n_atoms_in_)
    list1     = []
    for i in range(arrange.shape[0]):
        list = [1 if arrange[i,j] == absValue[i] else 0 for j in range(arrange.shape[1])]
        list1.append(list)
    list1.append(oensVAL.tolist())
    return np.array(list1)

def IndicesM(atoms_in_, ind_y_A_, ind_atoms_to_remove_):
    for i in ind_atoms_to_remove_:
        indexI = ind_y_A_[i]
        atoms_in_[indexI] = False
    return atoms_in_

def deleteC(A_, ind_atoms_to_remove_):
    list1 = []
    list2 = []
    for i in range(A_.shape[1]):
        if i not in ind_atoms_to_remove_:
            list1.append(i)
    for i in list1:
        list2.append(A_[:,i].tolist())
    ak = np.array(list2).T
    return ak

def deleteL(y_, ind_atoms_to_remove_):
    list1 = []
    list2 = []
    for i in range(len(y_)):
        if i not in ind_atoms_to_remove_:
            list1.append(i)
    for i in list1:
        list2.append(y_[i])
    return list2

def yMultiSymbols(y_, ind_y_A_, y_temp_):
    counter = 0
    for i in ind_y_A_:
        y_[i] = y_temp_[counter]
        counter +=1
    return y_