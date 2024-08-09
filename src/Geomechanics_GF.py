# Function to compute the GIST green function of out-zone stress
# Written for Python 3.8
# Copyright 2022 ExxonMobil Upstream Research Company
# Authors: Yanhua Yuan email: yanhua.yuan@exxonmobil.com (based on Lei Jin's document 6, equations 3-20)
# Package Depenencies - numpy

import numpy as np 

def compute_green_function_gist(Source_coor, Points_Coor, poisson_ratio):
    # Source_Coor: source location [pore pressure location, 3*nmesh]
    # Points_Coor: receiver location [stress evaluation location, 3*nmesh]
    
    # Define relative distance terms 
    DX = Points_Coor[0] - Source_coor[0]
    DY = Points_Coor[1] - Source_coor[1]
    DZ_minus = Points_Coor[2] - Source_coor[2]
    DZ_plus = Points_Coor[2] + Source_coor[2]
    R_minus = np.sqrt(DX**2 + DY**2 + DZ_minus**2) # eq. 1
    R_plus = np.sqrt(DX**2 + DY**2 + DZ_plus**2) # eq. 2

    # Shared terms
    nu = poisson_ratio
    fac = 1. / (8.*np.pi) / (1.-nu)
    fac_nu_1 = (1.-2.*nu)
    fac_nu_2 = (3.-4.*nu)
    c = Source_coor[2]
    z = Points_Coor[2]
    R1 = R_minus
    R2 = R_plus
     
    # Source: A point force acting along z (downward) (eqs. 3-8)
    G_z_xx = fac * ( fac_nu_1*DZ_minus/R1**3 
                    - 3.*DX**2*DZ_minus/R1**5 
                    + fac_nu_1*(3.*DZ_minus-4.*nu*DZ_plus)/R2**3
                    - (3.*fac_nu_2*DX**2*DZ_minus - 6.*c*DZ_plus*(fac_nu_1*z-2.*nu*c))/R2**5 
                    - 30.*c*DX**2*z*DZ_plus/R2**7
                    - 4.*(1.-nu)*fac_nu_1/R2/(R2+z+c) * (1. - DX**2/R2/(R2+z+c) - DX**2/R2**2)
                   )

    G_z_yy = fac * ( fac_nu_1*DZ_minus/R1**3 
                    - 3.*DY**2*DZ_minus/R1**5 
                    + fac_nu_1*(3.*DZ_minus-4.*nu*DZ_plus)/R2**3
                    - (3.*fac_nu_2*DY**2*DZ_minus - 6.*c*DZ_plus*(fac_nu_1*z-2.*nu*c))/R2**5 
                    - 30.*c*DY**2*z*DZ_plus/R2**7
                    - 4.*(1.-nu)*fac_nu_1/R2/(R2+z+c) * (1. - DY**2/R2/(R2+z+c) - DY**2/R2**2)
                   )

    G_z_zz = fac * ( - fac_nu_1*DZ_minus/R1**3 
                    + fac_nu_1*DZ_minus/R2**3
                    - 3.*DZ_minus**3/R1**5 
                    - (3.*fac_nu_2*z*DZ_plus**2 - 3.*c*DZ_plus*(5.*z-c))/R2**5 
                    - 30.*c*z*DZ_plus**3/R2**7
                   )

    G_z_xy = DX*DY*fac * ( - 3.*DZ_minus/R1**5 
                          - 3.*fac_nu_2*DZ_minus/R2**5
                          + 4.*(1.-nu)*fac_nu_1/R2**2/(R2+z+c) * (1./(R2+z+c)+1./R2)
                          - 30.*c*z*DZ_plus/R2**7
                         )

    G_z_xz = DX*fac * ( - fac_nu_1/R1**3 
                       + fac_nu_1/R2**3 
                       - 3.*DZ_minus**2/R1**5 
                       - (3.*fac_nu_2*z*DZ_plus - 3.*c*(3.*z+c))/R2**5 
                       - 30.*c*z*DZ_plus**2/R2**7
                      )

    G_z_yz = DY*fac * ( - fac_nu_1/R1**3 
                       + fac_nu_1/R2**3 
                       - 3.*DZ_minus**2/R1**5 
                       - (3.*fac_nu_2*z*DZ_plus - 3.*c*(3.*z+c))/R2**5 
                       - 30.*c*z*DZ_plus**2/R2**7
                      )
    
    # Source: A point force acting along x (eqs. 9-14)
    G_x_xx = DX*fac * ( - fac_nu_1/R1**3 
                       + fac_nu_1*(5.-4.*nu)/R2**3 
                       - 3.*DX**2/R1**5 
                       - 3.*fac_nu_2*DX**2/R2**5 
                       - 4.*(1.-nu)*fac_nu_1/R2/(R2+z+c)**2 * (3.-DX**2*(3.*R2+z+c)/R2**2/(R2+z+c))
                       + 6.*c/R2**5 * (3.*c - (3.-2.*nu)*DZ_plus + 5.*DX**2*z/R2**2)
                      )

    G_x_yy = DX*fac * ( fac_nu_1/R1**3 
                       + fac_nu_1*fac_nu_2/R2**3 
                       - 3.*DY**2/R1**5 
                       - 3.*fac_nu_2*DY**2/R2**5 
                       - 4.*(1.-nu)*fac_nu_1/R2/(R2+z+c)**2 * (1.-DY**2*(3.*R2+z+c)/R2**2/(R2+z+c))
                       + 6.*c/R2**5 * (c - fac_nu_1*DZ_plus + 5.*DY**2*z/R2**2)
                      )

    G_x_zz = DX*fac * ( fac_nu_1/R1**3 
                       - fac_nu_1/R2**3 
                       - 3.*DZ_minus**2/R1**5 
                       - 3.*fac_nu_2*DZ_plus**2/R2**5 
                       + 6.*c/R2**5 * (c + fac_nu_1*DZ_plus + 5.*z*DZ_plus**2/R2**2)
                      )

    G_x_xy = DY*fac * ( - fac_nu_1/R1**3 
                       + fac_nu_1/R2**3 
                       - 3.*DX**2/R1**5 
                       - 3.*fac_nu_2*DX**2/R2**5 
                       - 4.*(1.-nu)*fac_nu_1/R2/(R2+z+c)**2 * (1.-DX**2*(3.*R2+z+c)/R2**2/(R2+z+c))
                       - 6.*c*z/R2**5 * (1. - 5.*DX**2/R2**2)
                      )

    G_x_xz = fac * ( - fac_nu_1*DZ_minus/R1**3 
                    + fac_nu_1*DZ_minus/R2**3
                    - 3.*DX**2*DZ_minus/R1**5 
                    - 3.*fac_nu_2*DX**2*DZ_plus/R2**5 
                    - 6.*c/R2**5 * (z*DZ_plus - fac_nu_1*DX**2 - 5.*DX**2*z*DZ_plus/R2**2)
                   )

    G_x_yz = DX*DY*fac * ( - 3.*DZ_minus/R1**5 
                          - 3.*fac_nu_2*DZ_plus/R2**5 
                          + 6.*c/R2**5 * (fac_nu_1 - 5.*z*DZ_plus/R2**2)
                         )

    # Source: A point force acting along y (eqs. 15-20)
    G_y_yy = DY*fac * ( - fac_nu_1/R1**3 
                       + fac_nu_1*(5.-4.*nu)/R2**3 
                       - 3.*DY**2/R1**5 
                       - 3.*fac_nu_2*DY**2/R2**5 
                       - 4.*(1.-nu)*fac_nu_1/R2/(R2+z+c)**2 * (3.-DY**2*(3.*R2+z+c)/R2**2/(R2+z+c))
                       + 6.*c/R2**5 * (3.*c - (3.-2.*nu)*DZ_plus + 5.*DY**2*z/R2**2)
                      )

    G_y_xx = DY*fac * ( fac_nu_1/R1**3 
                       + fac_nu_1*fac_nu_2/R2**3 
                       - 3.*DX**2/R1**5 
                       - 3.*fac_nu_2*DX**2/R2**5 
                       - 4.*(1.-nu)*fac_nu_1/R2/(R2+z+c)**2 * (1.-DX**2*(3.*R2+z+c)/R2**2/(R2+z+c))
                       + 6.*c/R2**5 * (c - fac_nu_1*DZ_plus + 5.*DX**2*z/R2**2)
                      )

    G_y_zz = DY*fac * ( fac_nu_1/R1**3 
                       - fac_nu_1/R2**3 
                       - 3.*DZ_minus**2/R1**5 
                       - 3.*fac_nu_2*DZ_plus**2/R2**5 
                       + 6.*c/R2**5 * (c + fac_nu_1*DZ_plus + 5.*z*DZ_plus**2/R2**2)
                      )

    G_y_xy = DX*fac * ( - fac_nu_1/R1**3 
                       + fac_nu_1/R2**3 
                       - 3.*DY**2/R1**5 
                       - 3.*fac_nu_2*DY**2/R2**5 
                       - 4.*(1.-nu)*fac_nu_1/R2/(R2+z+c)**2 * (1.-DY**2*(3.*R2+z+c)/R2**2/(R2+z+c))
                       - 6.*c*z/R2**5 * (1. - 5.*DY**2/R2**2)
                      )

    G_y_yz = fac * ( - fac_nu_1*DZ_minus/R1**3 
                    + fac_nu_1*DZ_minus/R2**3
                    - 3.*DY**2*DZ_minus/R1**5 
                    - 3.*fac_nu_2*DY**2*DZ_plus/R2**5 
                    - 6.*c/R2**5 * (z*DZ_plus - fac_nu_1*DY**2 - 5.*DY**2*z*DZ_plus/R2**2)
                   )

    G_y_xz = DX*DY*fac * ( - 3.*DZ_minus/R1**5 
                          - 3.*fac_nu_2*DZ_plus/R2**5 
                          + 6.*c/R2**5 * (fac_nu_1 - 5.*z*DZ_plus/R2**2)
                         )

    return np.array([
        [G_x_xx, G_x_yy, G_x_zz, G_x_xy, G_x_xz, G_x_yz],
        [G_y_xx, G_y_yy, G_y_zz, G_y_xy, G_y_xz, G_y_yz],
        [G_z_xx, G_z_yy, G_z_zz, G_z_xy, G_z_xz, G_z_yz]
    ])
