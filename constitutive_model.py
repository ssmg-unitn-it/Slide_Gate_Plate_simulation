import fenics as fe

# deformation tensor in small strain theory
def eps(u):
    return fe.sym(fe.grad(u))


# Thermal deformation
def eps_T(Temp, alpha, ref_Temp):
    return alpha*(Temp-ref_Temp)*fe.Identity(2)


# elastic deformation
def eps_e(u, Temp, eps_p, alpha, ref_Temp):
    return eps(u) - eps_T(Temp, alpha, ref_Temp) - eps_p


# positive elastic strain
def eigenSplit_pos(tensor):
    v00 = (tensor[0,0] + tensor[1,1])/2 + 
        fe.sqrt(
            ((tensor[0,0]-tensor[1,1])/2)**2 + 
            (tensor[1,0])**2
        )
    v01 = 0
    v10 = 0
    v11 = (tensor[0,0] + tensor[1,1])/2 - 
        fe.sqrt(((tensor[0,0]-tensor[1,1])/2)**2+
        (tensor[1,0])**2)
    
    w00 = 1/fe.sqrt(1+((v00-tensor[0,0])/tensor[1,0])**2)
    w01 = 1/fe.sqrt(1+((v11-tensor[0,0])/tensor[1,0])**2)
    w10 = w00*(v00-tensor[0,0])/tensor[1,0]
    w11 = w01*(v11-tensor[0,0])/tensor[1,0]
    
    w = ([w00,w01],[w10,w11])
    w = fe.as_tensor(w)
    w_tr = ([w00,w10],[w01,w11])
    w_tr = fe.as_tensor(w_tr)
    
    v00 = fe.conditional(fe.gt(v00,0.0),v00,0.0)
    v11 = fe.conditional(fe.gt(v11,0.0),v11,0.0)
    
    v = ([v00,v01],[v10,v11])
    v = fe.as_tensor(v)
    
    return w*v*w_tr


# p stress invariant = hydrostatic stress
# this function is needed only for the post-processing
def finvp(sig):
    return - 1/3 * fe.tr(sig)


# q stress invariant = deviatoric stress
# this function is needed only for the post-processing
def finvq(sig, ntol):
    dev = sig + finvp(sig)*fe.Identity(2)

    # ntol in J2 avoids numerical issues
    J2 = 0.5*fe.tr(dev*dev) + ntol

    return fe.sqrt(3*J2)


# cos(3*theta) stress invariant = cos[3*(Lode_angle)]
# this function is needed only for the post-processing
def finv_cos_3theta(sig, ntol):
    dev = sig + finvp(sig)*fe.Identity(2)

    # ntol in J2 avoids numerical issues
    J2 = 0.5*fe.tr(dev*dev) + ntol

    J3 = fe.det( dev )

    return (3*(3**0.5)/2)*J3/(J2**(3/2))


# normalized BP yield function
def YieldFunction(sig,pc,pc0,Omega,Mbp,alpbp,mbp,betbp,gambp,ntol):

    invp = - 1/3 * fe.tr(sig)
    dev = sig + invp*fe.Identity(2)
    
    # ntol in J2 avoids numerical issues
    J2 = 0.5*fe.tr(dev*dev) + ntol*pc0
    J3 = fe.det(dev)
    
    invq = fe.sqrt(3*J2)
    
    cos3theta = (3*(3**0.5)/2)*J3*(J2**(-3/2))
    
    cbp = Omega*pc0
    
    Phi = (invp + cbp)/(pc + cbp)
    
    finvp_val = - Mbp*pc*fe.sqrt(
        (Phi-abs(Phi)**(mbp-1)*Phi)*(2*(1-alpbp)*Phi+alpbp)
    )
    
    # the term under square root in finvp_val exists only if Phi in [0;1]
    finvp = fe.conditional( 
        fe.lt(Phi, 1.0), 
        fe.conditional(
            fe.gt(Phi, 0.0), 
            finvp_val, 
            fe.Constant(1e5)),
        fe.Constant(1e5)
    )
    
    gtheta_inverse=fe.cos(betbp*3.14/6-1/3*(fe.acos(gambp*cos3theta)))
    
    F = finvp + invq*gtheta_inverse
  
    return F/pc0


# yield function derivative
def dFdsig(sig, pc, pc0, Omega, Mbp, alpbp, mbp, betbp, gambp, ntol):
    vsig = fe.variable(sig)
    return fe.diff(
        YieldFunction(vsig,pc,pc0,Omega,Mbp,alpbp,mbp,betbp,gambp,ntol),
        vsig )


# pc mechanical hardening law
def pcM(Temp, Baralp, Ak, delta0):
    return Ak*Baralp/(1+delta0*Baralp)

    
# Macaulay brackets    
def Macaulay(var, negative=False):
    if negative:
        return (var - abs(var))/2
    else:
        return (var + abs(var))/2    


# tensor norm     
def tensorNorm(tensor):
    return fe.sqrt(fe.inner(tensor, tensor))
    

# positive signum function of scalar variable 
# (0 if negative, 1 if positive)    
def positive_signum(var, ntol):
    return Macaulay(var)/abs(var + ntol)


# crack driving force assuring irreversibility of damage
def H_new(u, Temp, eps_p, H_old, lmbda, mu, Gc, lsp, alpha, ref_Temp):
    
    # elastic deformation
    elastic_strain = eps_e(u, Temp, eps_p, alpha, ref_Temp)
       
    # positive part of strain eigenvalues split 
    eps_e_p = eigenSplit_pos(elastic_strain)
    
    # plane stress approximation Hp
    lmbda_star = 2*lmbda*mu/(lmbda+2*mu)
    
    # damaged elastic strain energy (tension part)
    psiD_new=0.5*lmbda_star*(Macaulay(fe.tr(elastic_strain)))**2+\
        mu*fe.tr( eps_e_p*eps_e_p )
    
    # crack driving force normalization to allow eventual extension
    psiD_new = psiD_new/Gc

    # Irreversibility condition is imposed using ufl operators
    return fe.conditional( fe.lt(H_old, psiD_new), psiD_new, H_old )


# degradation function
def gdeg(d, ntol):
    return pow(1-d,2) + ntol


# stress calculation
def sig(u, Temp, dam, eps_p, lmbda, mu, alpha, ref_Temp, ntol):
    elastic_strain = eps_e(u, Temp, eps_p, alpha, ref_Temp)

    # plane stress approximation Hp
    lmbda_star = 2*lmbda*mu/(lmbda+2*mu)
    
    return gdeg(dam, ntol)*( 
            lmbda_star*fe.tr( elastic_strain )*fe.Identity(2) + 
            2.0*mu*elastic_strain 
        )