"""
This is random material, do not read it :)
"""


def _vec_to_log_pdm(vec, d):
    """
    """
    # get indexes:
    ind = np.tril_indices(d, 0)
    # initialize:
    mat = np.zeros((d, d))
    mat[ind] = vec
    # take exponential of the diagonal to ensure positivity:
    mat[np.diag_indices(d)] = np.exp(np.diagonal(mat))
    #
    return np.dot(mat, mat.T)


def _log_pdm_to_vec(pdm, d):
    """
    """
    # decompose:
    mat = np.linalg.cholesky(pdm)
    # take log of diagonal:
    mat[np.diag_indices(d)] = np.log(np.diagonal(mat))
    #
    return mat[np.tril_indices(d, 0)]



def _temp_vec_kde_pdf(x, samples, weights):
    """
    Utility function to compute the KDE
    """
    X = np.subtract(x[np.newaxis, :, :], samples[:, np.newaxis, :])
    _temp = np.dot(weights, np.exp(-0.5*(X*X).sum(axis=2)))
    return np.log(_temp)



C_p1, C_p12 = chain_2.cov(pars=param_names), chain_12.cov(pars=param_names)
C_Pi = chain_prior.cov(pars=param_names)
theta_1 = chain_2.getMeans(pars=[chain_1.index[name]
                           for name in param_names])
theta_12 = chain_12.getMeans(pars=[chain_12.index[name]
                             for name in param_names])

KL_eig, KL_eigv = utils.KL_decomposition(C_p1, C_p12)

KL_eig

prior_factor = 1000.0
temp_C_p1 = utils.QR_inverse(utils.QR_inverse(C_p1) +prior_factor*utils.QR_inverse(C_Pi))
temp_C_p12 = utils.QR_inverse(utils.QR_inverse(C_p12) +prior_factor*utils.QR_inverse(C_Pi))

temp_theta_12 = np.dot(temp_C_p12,np.dot(theta_12,utils.QR_inverse(C_p12))+np.dot(theta_1,prior_factor*utils.QR_inverse(C_Pi)))


KL_eig_2, KL_eigv_2 = utils.KL_decomposition(temp_C_p1, temp_C_p12)
KL_eig_2

plt.plot(KL_eig)
plt.plot(KL_eig_2)

helper_stat(theta_1-theta_12, KL_eig,KL_eigv, 1.05)
helperplot(KL_eig,KL_eigv)

helper_stat(theta_1-temp_theta_12, KL_eig_2,KL_eigv_2, 1.05)
helperplot(KL_eig_2,KL_eigv_2)




def helper_stat(shift, eig, eigv, lower_cutoff):
    upper_cutoff = 100.
    _filter = np.logical_and(eig > lower_cutoff, eig < upper_cutoff)
    Q_UDM = np.sum(np.dot(eigv.T, shift)[_filter]**2./(eig[_filter]-1.))
    dofs = np.sum(_filter)
    P = scipy.stats.chi2.cdf(Q_UDM, dofs)
    return utils.from_confidence_to_sigma(P)

def helperplot(eig, eigv):
    fish = np.sum(eigv*eigv/eig, axis=1)
    fish = ((eigv*eigv/eig).T/fish).T
    im1 = plt.imshow(fish, cmap='viridis')
    num_params = len(fish)
    for i in range(num_params):
        for j in range(num_params):
            if fish[j,i]>0.5:
                col = 'k'
            else:
                col = 'w'
            plt.text(i, j, np.round(fish[j,i],2), va='center', ha='center', color=col)
    # label on the axis:
    plt.xlabel('KL mode ($\\lambda^a-1$)');
    plt.ylabel('Parameters');
    # x axis:
    ticks  = np.arange(num_params)
    labels = [ str(t+1)+'\n ('+str(l)+')' for t,l in zip(ticks,np.round(eig-1.,2))]
    plt.xticks(ticks, labels, horizontalalignment='center')
    labels = [ '$'+chain_12.getParamNames().parWithName(name).label+'$' for name in param_names ]
    plt.yticks(ticks, labels, horizontalalignment='right')
    plt.show()
