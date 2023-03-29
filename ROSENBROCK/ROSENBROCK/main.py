import time 
import numpy      as     np 
from   scipy      import sparse
from   df_simplex import df_simplex
from   ord        import ord

np.random.seed(1)

# def operations(firstTerm, secTerm, operate):
#     '''
#     Operation used in obj function
#     '''
#     if operate == 'power':
#         return [x**secTerm for x in firstTerm]
#     elif operate == 'minus':
#         return [secTerm - x1 for x1 in firstTerm]
#     elif operate == 'Lminus':   
#         return [x1 - x2 for x1, x2 in zip(firstTerm, secTerm)]
#     elif operate == 'multiply':  
#         return [x * secTerm for x in firstTerm]

# Define an objective function:
#------------------------------
# Extended Rosenbrock function
n = 50     # dimension
c = 1e1    # parameter of the objective function

# def objFunc(x):
#     x = x[:n]
#     Atrail =  operations(x[1::2],operations(x[::2], 2, 'power'), 'Lminus')
#     A      =  operations(operations( Atrail, 2, 'power'), c, 'multiply')
#     B      =  operations(operations( x[::2], 1E0, 'minus'), 2, 'power')
#     assert(len(A) == len(B))
#     return sum([A[i] + B[i] for i in range(len(B))]) # or len(A)

def objFunc(x):
    x = np.array(x)
    return np.sum(c*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)

obj = objFunc

# Define an n-by-m matrix of atoms, where each column is an n-dimensional atom:
#---------------------------------------------------------------------------------
# atoms randomly generated
m = 100  # number of atoms
A = 1E1 * np.random.random((n, m)) # matrix of atoms

# Choose an atom to be used as starting point:
#---------------------------------------------
i0 = np.random.randint(1,m) # index of the atom to be used as starting point


# call ORD (in case needed)
#--------------------------------------------------------------------------
start=time.time()
x_ord, f_ord, ord_info  = ord(obj, A, i0)
end=time.time()
# write statistics to the screen
print('********************** FINAL RESULTS **********************')
print('Algorithm: ORD')
print('f = ', "{:.4e}".format(f_ord))
print('objective function evaluations = ' , ord_info.n_f)
print('iterations = ' , ord_info.it)
print('flag = ' ,ord_info.flag)
print('**************** END OF ORD ********************************')
print("\n")
print("CPU TIME:", end-start)

# start = time.time()
# x_df_simplex, f_df_simplex, df_simplex_info, _ = df_simplex(obj, A, np.double([0 if i != i0 else 1 for i in range(m)]))
# end=time.time()
# print('********************** FINAL RESULTS **********************')
# print('Algorithm: DF-SIMPLEX')
# print('f = ', "{:.4e}".format(f_df_simplex))
# print('objective function evaluations = ', df_simplex_info.n_f)
# print('iterations = ', df_simplex_info.it)
# print('flag = ' ,  df_simplex_info.flag)
# print('******************* END OF THE CODE ************************') 
# print("\n")
# print("CPU TIME:", end-start)
