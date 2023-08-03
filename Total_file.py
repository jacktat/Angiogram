import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from functionsImport_phantom import *
matplotlib.use('tkAgg')
import time
start = time.time()
from scipy.optimize import lsq_linear
import pydicom as dicom
import matplotlib.image as mpimg
import plotly.graph_objects as go
from PIL import Image

#note: IM7 and IM8

# input
alpha =         np.radians( -29.3)     
beta  =          np.radians(25.5)       
alpha2 =       np.radians( 30.7)        
beta2 =        np.radians(23.6)        

# Angiogram #settings
d_s1 = 720
d_sd1 =   976  
d_d1 = d_sd1 - d_s1  # Patient to detector
d_s2 = 720 #   
d_sd2 =   976  
d_d2 = d_sd2 - d_s2
d_p = 0.390625/1.3556
px = 512 -1 

def rotation(alpha, beta):
    M = np.array([(np.cos(alpha), 0, np.sin(alpha)),
                  (np.sin(-beta) * np.sin(alpha), np.cos(-beta), np.sin(-beta) * -np.cos(alpha)),
                  (-np.cos(-beta) * np.sin(alpha), np.sin(-beta), np.cos(alpha) * np.cos(-beta))])
    return M

M = rotation(alpha,beta)
M2 = rotation(alpha2,beta2)

# original source and detector.
S_o1 = np.array([0, 0, -d_s1])
S_o2 = np.array([0, 0, -d_s2])
D_o1 = np.array([0, 0, d_d1])
D_o2 = np.array([0, 0, d_d2])
# define detector corners P, S, R, Q with P bottom left, and rotating counter-clockwise for the rest.
P_o1 = np.array([-px / 2 * d_p, -px / 2 * d_p, d_d1])  # bottomleft corner of detector
SS_o1 = np.array([px / 2 * d_p, -px / 2 * d_p, d_d1])  # bottomright corner of detector
R_o1 = np.array([px / 2 * d_p, px / 2 * d_p, d_d1])  # top right corner of detector
Q_o1 = np.array([-px / 2 * d_p, px / 2 * d_p, d_d1])  # top left corner of detector
P_o2 = np.array([-px / 2 * d_p, -px / 2 * d_p, d_d2])  # bottomleft corner of detector
SS_o2 = np.array([px / 2 * d_p, -px / 2 * d_p, d_d2])  # bottomright corner of detector
R_o2 = np.array([px / 2 * d_p, px / 2 * d_p, d_d2])  # top right corner of detector
Q_o2 = np.array([-px / 2 * d_p, px / 2 * d_p, d_d2])  # top left corner of detector

P1 = np.matmul(M, P_o1)
P2 = np.matmul(M2, P_o2)
SS_1 = np.matmul(M, SS_o1)
SS_2 = np.matmul(M2, SS_o2)
R1 = np.matmul(M, R_o1)
R2 = np.matmul(M2, R_o2)
Q1 = np.matmul(M, Q_o1)
Q2 = np.matmul(M2, Q_o2)
S = np.matmul(M, S_o1)  # Position of Xray Source
D = np.matmul(M, D_o1)  # Centerpoint of Detector
S2 = np.matmul(M2, S_o2)  # Position of Xray Source
D2 = np.matmul(M2, D_o2)  # Centerpoint of Detector

r2 = np.array([0, 1, 0])
r2 = np.matmul(M, r2)
r3 = np.array([1, 0, 0])
r3 = np.matmul(M, r3)
r2_2 = M2@np.array([0, 1, 0])   #there was a massive mistake with calculation of r2_2 because r2 and r3 were overwritten
r3_2 = M2@np.array([1, 0, 0])   #there was a massive mistake with calculation of r2_2 because r2 and r3 were overwritten

r2_base = np.array([0, 1])
r3_base = np.array([1, 0])

# note these coords are just for plotting detector edge midpoints
topmid_3d = M @ np.array([0, (px/2) * d_p, D_o1[2]])
botmid_3d = M @ np.array([0, -(px/2) * d_p, D_o1[2]])
rightmid_3d = M @ np.array([-(px/2) * d_p, 0, D_o1[2]])
leftmid_3d = M @ np.array([(px/2) * d_p, 0, D_o1[2]])
topmid_3d_2 = M2 @ np.array([0, (px/2) * d_p, D_o2[2]])
botmid_3d_2 = M2 @ np.array([0, -(px/2) * d_p, D_o2[2]])
rightmid_3d_2 = M2 @ np.array([-(px/2) * d_p, 0, D_o2[2]])
leftmid_3d_2 = M2 @ np.array([(px/2) * d_p, 0, D_o2[2]])

def projection_1(x, y, z):
    point = np.array([x, y, z])
    dirV = point - S  # S - point
    A = np.array([(r2[0] / np.linalg.norm(r2), r3[0] / np.linalg.norm(r3), - dirV[0] / np.linalg.norm(dirV)),
                  (r2[1] / np.linalg.norm(r2), r3[1] / np.linalg.norm(r3), - dirV[1] / np.linalg.norm(dirV)),
                  (r2[2] / np.linalg.norm(r2), r3[2] / np.linalg.norm(r3), - dirV[2] / np.linalg.norm(dirV))])
    b = np.array([(S[0] - D[0]),
                  (S[1] - D[1]),
                  (S[2] - D[2])])
    result = np.linalg.solve(A, b)
    return result

def projection_2(x2, y2, z2):
    point_2 = np.array([x2, y2, z2])
    dirV_2 = point_2 - S2  # S2 - point_2
    A2 = np.array(
        [(r2_2[0] / np.linalg.norm(r2_2), r3_2[0] / np.linalg.norm(r3_2), - dirV_2[0] / np.linalg.norm(dirV_2)),
         (r2_2[1] / np.linalg.norm(r2_2), r3_2[1] / np.linalg.norm(r3_2), - dirV_2[1] / np.linalg.norm(dirV_2)),
         (r2_2[2] / np.linalg.norm(r2_2), r3_2[2] / np.linalg.norm(r3_2), - dirV_2[2] / np.linalg.norm(dirV_2))])
    b2 = np.array([(S2[0] - D2[0]),
                   (S2[1] - D2[1]),
                   (S2[2] - D2[2])])
    result2 = np.linalg.solve(A2, b2)
    return result2

view1 = np.array([
  [        153.59    ,  -127.78     ,       0],
  [        156.73    ,  -136.59     ,  9.3392],
  [        161.14    ,   -145.4     ,  19.172],
  [        166.17    ,  -154.22     ,  29.302],
  [        169.32    ,   -162.4     ,  38.052],
  [        172.47    ,  -169.95     ,  46.219],
  [        174.36    ,  -178.13     ,  54.601],
  [        178.13    ,  -185.06     ,  62.472],
  [        182.54    ,  -195.13     ,  73.444],
  [        184.43    ,  -202.05     ,  80.606],
  [        184.43    ,  -208.98     ,  87.517],
  [        182.54    ,  -212.75     ,  91.731],
  [        179.39    ,  -214.64     ,  95.394],
  [         173.1    ,  -215.27     ,  101.71],
  [        167.43    ,  -214.01     ,   107.5],
  [        161.77    ,  -212.12     ,  113.46],
  [        153.59    ,  -210.87     ,  121.72],
  [        144.14    ,  -207.72     ,  131.66],
  [        137.85    ,  -204.57     ,  138.68]
])

view2 = np.array([
  [      -61.057   ,   -103.86   ,         0 ],
  [       -57.28   ,    -113.3   ,    10.149 ],
  [      -52.874   ,   -122.74   ,    20.548 ],
  [      -47.838   ,   -134.07   ,    32.923 ],
  [      -42.803   ,   -143.51   ,    43.602 ],
  [      -35.249   ,   -153.59   ,    56.167 ],
  [      -27.696   ,   -159.88   ,     65.98 ],
  [      -21.401   ,   -169.32   ,    77.305 ],
  [      -15.107   ,   -176.88   ,    87.118 ],
  [      -7.5534   ,   -183.17   ,    96.932 ],
  [       3.1473   ,   -190.72   ,       110 ],
  [       10.701   ,   -196.39   ,    119.43 ],
  [       20.142   ,   -202.68   ,    130.75 ],
  [       28.955   ,   -206.46   ,    140.32 ],
  [       39.026   ,    -211.5   ,    151.56 ],
  [       53.503   ,   -214.01   ,    166.23 ],
  [       64.204   ,   -210.24   ,    177.55 ],
  [       70.498   ,   -207.72   ,    184.32 ],
  [       79.311   ,   -202.05   ,    194.77 ]
])


proj1_2DX =  view1[:,0]
proj1_2DY =  view1[:,1]
ss        =  view1[:,2]
proj2_2DX =  view2[:,0]
proj2_2DY =  view2[:,1]
ss2       =  view2[:,2]

noPoints = 100#125
step_t = (ss[-1] / (noPoints))
step_s = (ss2[-1] / (noPoints))
t = list(np.arange(0, ss[-1], step_t)) # t is parameter for projection 1
s = list(np.arange(0, ss2[-1], step_s)) # s is parameter for projection 2

best_order_fit = 10 # order of best fit

# Make empty lists for legendre polynomial fits
fit_list_x_proj1 = []
x_coord_fit_proj1 = []
fit_list_y_proj1 = []
y_coord_fit_proj1 = []
fit_list_x_proj2 = []
x_coord_fit_proj2 = []
fit_list_y_proj2 = []
y_coord_fit_proj2 = []

# --- Fit legendre polynomial for X of projection 1 with best order ---
#errorX_proj1 = fit_list_x_proj1[0]  # SSR

legendre_coefs_x_proj1 = np.polynomial.legendre.legfit(ss, proj1_2DX, best_order_fit) #Least squares fit of Legendre series to data, coefficients
x_coord_fit_proj1 = np.polynomial.legendre.legval(t, legendre_coefs_x_proj1) #Evaluate a Legendre series at points x


legendre_coefs_y_proj1 = np.polynomial.legendre.legfit(ss, proj1_2DY, best_order_fit)
y_coord_fit_proj1 = np.polynomial.legendre.legval(t, legendre_coefs_y_proj1)

#second fit
legendre_coefs_x_proj2 = np.polynomial.legendre.legfit(ss2, proj2_2DX, best_order_fit) #Least squares fit of Legendre series to data, coefficients
x_coord_fit_proj2 = np.polynomial.legendre.legval(s, legendre_coefs_x_proj2) #Evaluate a Legendre series at points x


legendre_coefs_y_proj2 = np.polynomial.legendre.legfit(ss2, proj2_2DY, best_order_fit)
y_coord_fit_proj2 = np.polynomial.legendre.legval(s, legendre_coefs_y_proj2)

# ___plotting___#
plotSwitch = 1
if plotSwitch:
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(S[0], S[1], S[2], color='k')  # source
    ax.scatter(D[0], D[1], D[2], color='b')  # detectormid
    ax.scatter(S2[0], S2[1], S2[2], color='k')  # source
    ax.scatter(D2[0], D2[1], D2[2], color='b')  # detectormid

    x_vals_detector = [[P1[0], SS_1[0], R1[0], Q1[0], P1[0]], [P2[0], SS_2[0], R2[0], Q2[0], P2[0]]]
    y_vals_detector = [[P1[1], SS_1[1], R1[1], Q1[1], P1[1]], [P2[1], SS_2[1], R2[1], Q2[1], P2[1]]]
    z_vals_detector = [[P1[2], SS_1[2], R1[2], Q1[2], P1[2]], [P2[2], SS_2[2], R2[2], Q2[2], P2[2]]]
    plt.plot(x_vals_detector[0], y_vals_detector[0], z_vals_detector[0], c='k')
    plt.plot(x_vals_detector[1], y_vals_detector[1], z_vals_detector[1], c='k')

    ax.scatter(topmid_3d[0], topmid_3d[1], topmid_3d[2], c='yellow', alpha=0.4)
    ax.scatter(botmid_3d[0], botmid_3d[1], botmid_3d[2], c='k', alpha=0.4)
    ax.scatter(rightmid_3d[0], rightmid_3d[1], rightmid_3d[2], c='k', alpha=0.4)
    ax.scatter(leftmid_3d[0], leftmid_3d[1], leftmid_3d[2], c='k', alpha=0.4)
    ax.scatter(topmid_3d_2[0], topmid_3d_2[1], topmid_3d_2[2], c='yellow', alpha=0.4)
    ax.scatter(botmid_3d_2[0], botmid_3d_2[1], botmid_3d_2[2], c='k', alpha=0.4)
    ax.scatter(rightmid_3d_2[0], rightmid_3d_2[1], rightmid_3d_2[2], c='k', alpha=0.4)
    ax.scatter(leftmid_3d_2[0], leftmid_3d_2[1], leftmid_3d_2[2], c='k', alpha=0.4)


    ax.set_aspect('equal')
    plt.xlabel('x-axis', fontsize=20)
    plt.ylabel('y-axis', fontsize=20)

    fig2 = plt.figure("1st view, red")

    ax3 = fig2.add_axes([0, 0, 1, 1])
    plt.xlim([-(px / 2), (px / 2)])
    plt.ylim([-(px / 2), (px / 2)])
    phantom_IM7 = mpimg.imread('phantom_IM7.jpg')
    extent = [-px/2, px/2, -px/2, px/2]
    ax3.imshow(phantom_IM7, extent=extent, aspect='equal', cmap='bone', alpha=0.8)


    ax2 = fig2.add_axes([0, 0, 1, 1])

    plt.xlim([-(px/2), (px/2)])
    plt.ylim([-(px/2), (px/2)])
    plt.plot(x_coord_fit_proj1, y_coord_fit_proj1, label='Legendre fit_proj', c='r',alpha=0.9)

    for i in range(len(view1)):
        ax2.scatter(proj1_2DX[i],proj1_2DY[i],color='b')

    fig3 = plt.figure("2d view, yelow")

    ax3 = fig3.add_axes([0, 0, 1, 1])
    plt.xlim([-(px/2), (px/2)])
    plt.ylim([-(px/2), (px/2)])


    for i in range(len(view2)):
        ax3.scatter(proj2_2DX[i],proj2_2DY[i],color='b')



    plt.plot(x_coord_fit_proj2, y_coord_fit_proj2, label='Legendre fit_proj', c='r')
    ax3.legend()
    ax2.legend()
    plt.show()

""" 
def reconstruct3D(Point,alpha1,beta1,alpha2,beta2,d_s,d_sd,d_p,px):
    epipolarpoint_local_2D_func = []
    betaF =   beta1
    alphaF =  alpha1
    beta2F =  beta2
    alpha2F = alpha2
    d_sF = d_s
    d_sdF = d_sd
    d_dF = d_sdF - d_sF  # Patient to detector
    d_pF = d_p  # 0.390625  # Pixel spacing
    pxF = px  # 512  # Pixels
    MF = rotation(alphaF, betaF)
    M2F = rotation(alpha2F, beta2F)
    S_oF = np.array([0, 0, -d_sF])
    D_oF = np.array([0, 0, d_dF])
    P_oF = np.array([-pxF / 2 * d_pF, -pxF / 2 * d_pF, d_dF])  # bottomleft corner of detector
    SS_oF = np.array([pxF / 2 * d_pF, -pxF / 2 * d_pF, d_dF])  # bottomright corner of detector
    R_oF = np.array([pxF / 2 * d_pF, pxF / 2 * d_pF, d_dF])  # top right corner of detector
    Q_oF = np.array([-pxF / 2 * d_pF, pxF / 2 * d_pF, d_dF])  # top left corner of detector
    P1F = np.matmul(MF, P_oF)
    P2F = np.matmul(M2F, P_oF)
    SS_1F = np.matmul(MF, SS_oF)
    SS_2F = np.matmul(M2F, SS_oF)
    R1F = np.matmul(MF, R_oF)
    R2F = np.matmul(M2F, R_oF)
    Q1F = np.matmul(MF, Q_oF)
    Q2F = np.matmul(M2F, Q_oF)
    SF = np.matmul(MF, S_oF)  # Position of Xray Source
    DF = np.matmul(MF, D_oF)  # Centerpoint of Detector
    S2F = np.matmul(M2F, S_oF)  # Position of Xray Source
    D2F = np.matmul(M2F, D_oF)  # Centerpoint of Detector
    r2F = np.array([0, 1, 0])  # basic x and y directional vectors to define the detector.
    r2F = np.matmul(MF, r2F)
    r3F = np.array([1, 0, 0])
    r3F = np.matmul(MF, r3F)
    r2_2F = np.array([0, 1, 0])
    r2_2F = np.matmul(M2F, r2_2F)
    r3_2F = np.array([1, 0, 0])
    r3_2F = np.matmul(M2F, r3_2F)
    reconstructed_points_func = []
    r4F = S2F - SF
    sign_change_location_totalF = []
    r5F = Point - S2F
    AAF = np.array([(r2_2F[0] / np.linalg.norm(r2_2F), r3_2F[0] / np.linalg.norm(r3_2F), -r4F[0] / np.linalg.norm(r4F),
                    -r5F[0] / np.linalg.norm(r5F)),
                   (r2_2F[1] / np.linalg.norm(r2_2F), r3_2F[1] / np.linalg.norm(r3_2F), -r4F[1] / np.linalg.norm(r4F),
                    -r5F[1] / np.linalg.norm(r5F)),
                   (r2_2F[2] / np.linalg.norm(r2_2F), r3_2F[2] / np.linalg.norm(r3_2F), -r4F[2] / np.linalg.norm(r4F),
                    -r5F[2] / np.linalg.norm(r5F))])
    BBF = np.array([S2F[0] - D2F[0], S2F[1] - D2F[1], S2F[2] - D2F[2]])
    n1F = np.cross(r2_2F, r3_2F)  # normal on detectorplane 1
    n2F = np.cross(r4F, r5F)  # normal on epipolar plane
    r_epi_f = np.cross(n1F,
                     n2F)  # cross product of the  two normals of the planes. This equals to the directional vector of the epipolar line.
    boundsF = ([- pxF / 2 * d_pF, - pxF / 2 * d_pF, -np.inf, -np.inf], [pxF / 2 * d_pF, pxF / 2 * d_pF, np.inf, np.inf])
    resF = lsq_linear(AAF, BBF, bounds=boundsF, method='bvls', lsmr_tol='auto', verbose=0)
    lam2_f = resF.x[0]  # dit zijn de lamdas voor een punt op de snijlijn van de 2 vlakken
    lam3_f = resF.x[1]

    x_epipolar_point_func = D2F[0] + (lam2_f / np.linalg.norm(r2_2F)) * r2_2F[0] + (lam3_f / np.linalg.norm(r3_2F)) * r3_2F[0]
    y_epipolar_point_func = D2F[1] + (lam2_f / np.linalg.norm(r2_2F)) * r2_2F[1] + (lam3_f / np.linalg.norm(r3_2F)) * r3_2F[1]
    z_epipolar_point_func = D2F[2] + (lam2_f / np.linalg.norm(r2_2F)) * r2_2F[2] + (lam3_f / np.linalg.norm(r3_2F)) * r3_2F[2]
    L1F = calculate_lambda_JT([x_epipolar_point_func,y_epipolar_point_func,z_epipolar_point_func],r_epi_f,P2F,SS_2F)
    L2F = calculate_lambda_JT([x_epipolar_point_func,y_epipolar_point_func,z_epipolar_point_func],r_epi_f,P2F,Q2F)
    L3F = calculate_lambda_JT([x_epipolar_point_func,y_epipolar_point_func,z_epipolar_point_func],r_epi_f,Q2F,R2F)
    L4F = calculate_lambda_JT([x_epipolar_point_func,y_epipolar_point_func,z_epipolar_point_func],r_epi_f,R2F,SS_2F)
    lamda_listF = [L1F,L2F,L3F,L4F]
    lamda_listF = sorted(lamda_listF)
    min_lambda_epiF = lamda_listF[1]
    max_lamda_epiF = lamda_listF[2]
    aF = np.linspace( min_lambda_epiF, max_lamda_epiF , 100)
    x_epipolarlineF = np.zeros(len(aF))
    y_epipolarlineF = np.zeros(len(aF))
    z_epipolarlineF = np.zeros(len(aF))

    for k in range(len(aF)):
        x_valueF = x_epipolar_point_func + ( aF[k] * r_epi_f[0] / np.linalg.norm(r_epi_f) )
        y_valueF = y_epipolar_point_func + ( aF[k] * r_epi_f[1] / np.linalg.norm(r_epi_f) )
        z_valueF = z_epipolar_point_func + ( aF[k] * r_epi_f[2] / np.linalg.norm(r_epi_f) )

        x_epipolarlineF[k] = x_valueF
        y_epipolarlineF[k] = y_valueF
        z_epipolarlineF[k] = z_valueF

        epipolarpoint_local_2D_func.append(project3Dto2D(P2F, SS_2F, Q2F,[x_valueF, y_valueF, z_valueF])[0])
    dirV_epi_2DF = epipolarpoint_local_2D_func[0] - epipolarpoint_local_2D_func[1]
    dirV_gF = np.dot(dirV_epi_2DF, np.array(((0, -1), (1,0))))
    prev_delta_yF = 0
    sign_change_locationF = []

    for point in range(len(legVx2_2D)):  # note: this loop outputs coordianates of intersection under variable "sign_change_location"
        intersection_d_epiF = intersection_two_lines_2D(np.array([legVx2_2D[point], legVy2_2D[point]]), dirV_gF,
                                                       epipolarpoint_local_2D_func[0], dirV_epi_2DF)
        delta_yF = legVy2_2D[point] - intersection_d_epiF[1]
        sign_delta_yF = np.sign(delta_yF)
        if point != 0 and sign_delta_yF != prev_delta_yF:
            sign_change_locationF.append(np.array(([legVx2_2D[point], legVy2_2D[point]])))
        prev_delta_yF = sign_delta_yF
    if len(sign_change_locationF) == 1:
        subListF = []
        epi_curve_intersect3DF = project2Dto3D_JT(M2F, d_dF, d_pF, sign_change_locationF[0])
        intersectionF = xRayIntersection(SF, Point, S2F, epi_curve_intersect3DF,
                                        domain=[(-300, 300), (-400, 300),
                                                (-800, 200)])
        intersectionF = np.around(intersectionF.astype(np.double), 4)
        subListF.append(intersectionF)

        reconstructed_points_func.append(subListF)
        reconstructed_points_func.append(intersectionF)
        pointTransferred = project3Dto2D(P2F, SS_2F, Q2F, intersectionF)
    elif len(sign_change_locationF) > 1:
        # print("len sign change=",len(sign_change_location))
        subListF = []
        for i in range(len(sign_change_locationF)):
            epi_curve_intersect3DF = project2Dto3D_JT(M2F, d_dF, d_pF, sign_change_locationF[i])
            intersectionF = xRayIntersection(SF, Point, S2F,
                                            epi_curve_intersect3DF,
                                            domain=[(-300, 300), (-400, 300),
                                                    (-800, 200)])
            intersectionF = np.around(intersectionF.astype(np.double), 4)
            subListF.append(intersectionF)
            total_reconstructed_points.append(intersectionF)
        reconstructed_points_func.append(subListF)
    return reconstructed_points_func
"""

legVx =  x_coord_fit_proj1
legVy =  y_coord_fit_proj1
legVx2 = x_coord_fit_proj2
legVy2 = y_coord_fit_proj2

# transfer the 2D reparametrised coordiantes to 3D by adding z_value at the detector plane , then rotate with matmul(M)
#legVx *= -d_p  #was niet *-1
legVx = [value * -d_p for value in legVx]                       #note let op  deze line en die hierboven
#legVy *= d_p                            #todo morgen checken: ik heb dus al deze onzin hieronder vberandered van b *= d_p het ding naar y=y*d_p
legVy = [value * d_p for value in legVy]
legVz = d_d1 * np.ones_like(legVx)
centerline1 = []
legVx_2D = [value / -d_p for value in legVx] #legVx / -d_p
legVy_2D = [value / d_p for value in legVy]#legVy / d_p
#legVx2 *= -d_p
legVx2 = [value * -d_p for value in legVx2]
legVy2 = [value * d_p for value in legVy2]
legVz2 = d_d2 * np.ones_like(legVx2)

centerline2 = []
legVx2_2D = [value / -d_p for value in legVx2] #legVx2 / -d_p
legVy2_2D = [value / d_p for value in legVy2] #legVy2 / d_p
#legVx2_2D = legVx2 / -d_p
#legVy2_2D =legVy2 / d_p

for i in range(len(legVx)):
    #compare = project2Dto3D_JT(M,d_d1,d_p,np.array([legVx_2D[i],legVy_2D[i]]))
    point = np.array([legVx[i], legVy[i], legVz[i]])
    pointRot = np.matmul(M, point)
    centerline1.append(pointRot)

for i in range(len(legVx2)):
    point2 = np.array([legVx2[i], legVy2[i], legVz2[i]])
    pointRot2 = np.matmul(M2, point2)
    centerline2.append(pointRot2)

# ____________ Epipolar planes and lines __________ #
reconstructed_points = []
epipolar_total = []
total_reconstructed_points = []
for point in range(len(centerline1)):
    r5 = centerline1[point] - S2
    r4 = S2 - S
    AA = np.array([(r2_2[0] / np.linalg.norm(r2_2), r3_2[0] / np.linalg.norm(r3_2), -r4[0] / np.linalg.norm(r4),
                    -r5[0] / np.linalg.norm(r5)),
                   (r2_2[1] / np.linalg.norm(r2_2), r3_2[1] / np.linalg.norm(r3_2), -r4[1] / np.linalg.norm(r4),
                    -r5[1] / np.linalg.norm(r5)),
                   (r2_2[2] / np.linalg.norm(r2_2), r3_2[2] / np.linalg.norm(r3_2), -r4[2] / np.linalg.norm(r4),
                    -r5[2] / np.linalg.norm(r5))])
    BB = np.array([S2[0] - D2[0], S2[1] - D2[1], S2[2] - D2[2]])
    n1 = np.cross(r2_2, r3_2)  # normal on detectorplane 1
    n2 = np.cross(r4, r5)  # normal on epipolar plane
    r_epi = np.cross(n1,
                     n2)  # cross product of the  two normals of the planes. This equals to the directional vector of the epipolar line.
    # epipolar plane directional vectors; 1 is vector pointing from one source to the other
    bounds = ([- px / 2 * d_p, - px / 2 * d_p, -np.inf, -np.inf], [px / 2 * d_p, px / 2 * d_p, np.inf, np.inf])
    res = lsq_linear(AA, BB, bounds=bounds, method='bvls', lsmr_tol='auto', verbose=0)
    lam2 = res.x[0]  # dit zijn de lamdas voor een punt op de snijlijn van de 2 vlakken
    lam3 = res.x[1]

    x_epipolar_point = D2[0] + (lam2 / np.linalg.norm(r2_2)) * r2_2[0] + (lam3 / np.linalg.norm(r3_2)) * r3_2[0]
    y_epipolar_point = D2[1] + (lam2 / np.linalg.norm(r2_2)) * r2_2[1] + (lam3 / np.linalg.norm(r3_2)) * r3_2[1]
    z_epipolar_point = D2[2] + (lam2 / np.linalg.norm(r2_2)) * r2_2[2] + (lam3 / np.linalg.norm(r3_2)) * r3_2[2]
    # calculation of lamda range which limits epipolar line to detector
    L1 = calculate_lambda_JT([x_epipolar_point, y_epipolar_point, z_epipolar_point], r_epi, P2, SS_2)
    L2 = calculate_lambda_JT([x_epipolar_point, y_epipolar_point, z_epipolar_point], r_epi, P2, Q2)
    L3 = calculate_lambda_JT([x_epipolar_point, y_epipolar_point, z_epipolar_point], r_epi, Q2, R2)
    L4 = calculate_lambda_JT([x_epipolar_point, y_epipolar_point, z_epipolar_point], r_epi, R2, SS_2)
    lamda_list = [L1, L2, L3, L4]
    lamda_list = sorted(lamda_list)
    min_lambda_epi = lamda_list[1]
    max_lamda_epi = lamda_list[2]
    a = np.linspace(min_lambda_epi, max_lamda_epi, 2)
    x_epipolarline = np.zeros(len(a))
    y_epipolarline = np.zeros(len(a))
    z_epipolarline = np.zeros(len(a))


    NoIntersectionsPerPoint = np.zeros(len(centerline1))
    sign_change_location_total = []
    epipolarpoint_local_2D = []
    for k in range(len(a)):
        x_value = x_epipolar_point + (a[k] * r_epi[0] / np.linalg.norm(r_epi))
        y_value = y_epipolar_point + (a[k] * r_epi[1] / np.linalg.norm(r_epi))
        z_value = z_epipolar_point + (a[k] * r_epi[2] / np.linalg.norm(r_epi))

        x_epipolarline[k] = x_value
        y_epipolarline[k] = y_value
        z_epipolarline[k] = z_value
        epipolarpoint_local_2D.append(project3Dto2D(P2, SS_2, Q2, [x_value, y_value, z_value])[0])
        epipolar_total.append(project3Dto2D(P2, SS_2, Q2, [x_value, y_value, z_value])[0])
    dirV_epi_2D = epipolarpoint_local_2D[0] - epipolarpoint_local_2D[
        1]  # note first transfer epipolar line in vector representation to 2D
    dirV_g = np.dot(dirV_epi_2D, np.array(((0, -1), (1,
                                                     0))))  # note (g is perpendicular line on epi) we will define directionalVector perpendicular to dirVector from epiline

    prev_delta_y = 0
    sign_change_location = []

    for point in range(
            len(legVx2_2D)):  # note: this loop outputs coordianates of intersection under variable "sign_change_location"
        intersection_d_epi = intersection_two_lines_2D(np.array([legVx2_2D[point], legVy2_2D[point]]), dirV_g,
                                                       epipolarpoint_local_2D[0], dirV_epi_2D)
        delta_y = legVy2_2D[point] - intersection_d_epi[1]
        sign_delta_y = np.sign(delta_y)
        if point != 0 and sign_delta_y != prev_delta_y:
            sign_change_location.append(np.array(([legVx2_2D[point], legVy2_2D[point]])))
            sign_change_location_total.append(np.array(([legVx2_2D[point], legVy2_2D[point]])))
        prev_delta_y = sign_delta_y
    # print(len(sign_change_location))

    # notitie: vanaf hier heb ik op detector2 in 2D, de snijpunten van de curve met de epipolar line bepaald.
    # notitie: deze punten zijn opgeslagen in sign_change_location.

    if len(sign_change_location) == 1:
        subList = []
        epi_curve_intersect3D = project2Dto3D_JT(M2, d_d1, d_p, sign_change_location[
            0])
        intersection = xRayIntersection(S, centerline1[point], S2,
                                        # intersection = xRayIntersection(S, [curveValsX[iPoint], curveValsY[iPoint], curveValsZ[iPoint]], S2,
                                        epi_curve_intersect3D,
                                        domain=[(-300, 300), (-400, 300),
                                                (-800, 200)])  # domain=[(-30, 30), (-40, 30), (-80, -20)]
        intersection = np.around(intersection.astype(np.double), 4)
        subList.append(intersection)
        # print(iPoint, intersection)
        NoIntersectionsPerPoint[point] = 1
        reconstructed_points.append(subList)
        total_reconstructed_points.append(intersection)

    elif len(sign_change_location) > 1:
        # print("len sign change=",len(sign_change_location))
        NoIntersectionsPerPoint[point] = len(sign_change_location)
        subList = []
        for i in range(len(sign_change_location)):
            epi_curve_intersect3D = project2Dto3D_JT(M2, d_d1, d_p, sign_change_location[i])
            intersection = xRayIntersection(S, centerline1[point], S2,
                                            epi_curve_intersect3D,
                                            domain=[(-300, 300), (-400, 300),
                                                    (-800, 200)])  # domain=[(-30, 30), (-40, 30), (-80, -20)]
            intersection = np.around(intersection.astype(np.double), 4)
            subList.append(intersection)
            total_reconstructed_points.append(intersection)
            # print(iPoint,intersection)
        reconstructed_points.append(subList)

allpossible3Dpoints = reconstructed_points

print("Centerline reconstruction complete, it took",time.time()-start,'seconds')

####_______________ PLOTTING ________________######

corner_points = np.array([SS_o1,  # Lower-left corner (x, y, z)
                          P_o1 , # Lower-right corner
                          R_o1 , # Upper-left corner
                          Q_o1])  # Upper-right corner


test = extract_json("Centerline.json")
#OrPointx,OrPointy,OrPointz = readjson("Centerline.json")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(centerline2[0][0],centerline2[0][1],centerline2[0][2], c='red',s=30) #centerline1[0]
plt.sca(ax)
# plt.plot(xvals, yvals, zvals, color='b',linestyle='dotted',alpha=0.2)                    # principle ray
#ax.scatter(S[0], S[1], S[2], color='b')  # source


# plt.plot(xvals2, yvals2, zvals2, color='k',linestyle='dotted',alpha=0.2)
ax.scatter(S[0], S[1], S[2], color='k')
ax.scatter(S2[0], S2[1], S2[2], color='navy')

for i in range(len(test)):
    ax.scatter(test[i][0],test[i][1],test[i][2],c='k')



for x in centerline1:
    break
    ax.scatter(*x, color='b',alpha=0.97,s=50)

for x in centerline2:
    break
    ax.scatter(*x, color='k',s=75)


#x_final = np.array([point_r[0] for point_r in allpossible3Dpoints])
#y_final = np.array([point_r[1] for point_r in allpossible3Dpoints])
#z_final = np.array([point_r[2] for point_r in allpossible3Dpoints])

#plt.scatter(x_final[-1],y_final[-1],zs=z_final[-1], color='teal',s=10,alpha=0.5)
#for kk in range(len(allpossible3Dpoints)):
#    ax.scatter(allpossible3Dpoints[kk][0][0],allpossible3Dpoints[kk][0][1],allpossible3Dpoints[kk][0][2], color='red')

# ax.plot_wireframe(xT, yT, zT, color='green', alpha=0.5, linewidth=0.25)  # epipolar plane
#plt.plot(x_epipolarline, y_epipolarline, z_epipolarline, color='r', linestyle='-', alpha=0.7)  # epipolar line


# plot van detector dmv hoekpunten
x_vals_detector = [[P1[0], SS_1[0], R1[0], Q1[0], P1[0]], [P2[0], SS_2[0], R2[0], Q2[0], P2[0]]]
y_vals_detector = [[P1[1], SS_1[1], R1[1], Q1[1], P1[1]], [P2[1], SS_2[1], R2[1], Q2[1], P2[1]]]
z_vals_detector = [[P1[2], SS_1[2], R1[2], Q1[2], P1[2]], [P2[2], SS_2[2], R2[2], Q2[2], P2[2]]]
#plt.plot(x_vals_detector[0], y_vals_detector[0], z_vals_detector[0], c='k')
plt.plot(x_vals_detector[0], y_vals_detector[0], z_vals_detector[0], c='k')
plt.plot(x_vals_detector[1], y_vals_detector[1], z_vals_detector[1], c='navy')

for x in centerline1:
    ax.scatter(*x, color='b')

for x in centerline2:
    ax.scatter(*x, color='k')


n_eval_detector = 2
#lamdaT = np.linspace(-7500 * d_p, 7500 * d_p, n_eval_detector)
#xT = np.empty((len(lamdaT), len(lamdaT)))
#yT = np.empty((len(lamdaT), len(lamdaT)))
#zT = np.empty((len(lamdaT), len(lamdaT)))
#for i in range(len(lamdaT)):
#    for j in range(len(lamdaT)):
#        xT[i][j] = S2[0] + (lamdaT[i] / np.linalg.norm(r4)) * r4[0] + (lamdaT[j] / np.linalg.norm(r5)) * r5[0]
#        yT[i][j] = S2[1] + (lamdaT[i] / np.linalg.norm(r4)) * r4[1] + (lamdaT[j] / np.linalg.norm(r5)) * r5[1]
#        zT[i][j] = S2[2] + (lamdaT[i] / np.linalg.norm(r4)) * r4[2] + (lamdaT[j] / np.linalg.norm(r5)) * r5[2]
#
#ax.plot_wireframe(xT, yT, zT, color='green', alpha=0.95, linewidth=0.75)  # epipolar plane
#plt.plot([S2[0], epi_curve_intersect3D[0]], [S2[1], epi_curve_intersect3D[1]], [S2[2], epi_curve_intersect3D[2]], color='k', alpha=0.8)



x_reconstructed = np.array([point_r[0] for point_r in total_reconstructed_points])
y_reconstructed = np.array([point_r[1] for point_r in total_reconstructed_points])
z_reconstructed = np.array([point_r[2] for point_r in total_reconstructed_points])
plt.scatter(x_reconstructed,y_reconstructed,zs=z_reconstructed, color='fuchsia',s=5)



ax.scatter(topmid_3d[0], topmid_3d[1], topmid_3d[2], c='yellow', alpha=0.4)
ax.scatter(topmid_3d_2[0], topmid_3d_2[1], topmid_3d_2[2], c='yellow', alpha=0.4)

xLabel = ax.set_xlabel('X-axis', linespacing=1)
yLabel = ax.set_ylabel('Y-axis', linespacing=1)
zLabel = ax.set_zlabel('Z-Axis', linespacing=3.4)
ax.set_aspect('equal')  # , adjustable='box')



fig2 = plt.figure("1st view")
#fig2.canvas.mpl_connect('button_press_event', onclick)
# Connect the button press and release events
#fig2.canvas.mpl_connect('button_press_event', on_button_press)
#fig2.canvas.mpl_connect('button_release_event', on_button_release_1)
ax2 = fig2.add_axes([0, 0, 1, 1])
plt.xlim([-(px/2), (px/2)])
plt.ylim([-(px/2), (px/2)])
phantom_IM7 = mpimg.imread('phantom_IM7.jpg')
extent = [-px/2, px/2, -px/2, px/2]
ax2.imshow(phantom_IM7,extent=extent, aspect='equal',cmap='bone',alpha=0.8)



#x_coords_epi_d1 = [point[0] for point in epipolarpoint_local_2D_func]
#y_coords_epi_d1 = [point[1] for point in epipolarpoint_local_2D_func]
#plt.plot(x_coords_epi_d1, y_coords_epi_d1,c='r')


plt.plot(legVx_2D, legVy_2D, color='k', label='Reparameterized Legendre Curve',alpha=0.5) #todo legVx_2D is in pixels, moet dat
plt.plot(x_coord_fit_proj1, y_coord_fit_proj1, color='crimson', label='Reparameterized Legendre Curve',alpha=0.5)
ax2.legend()


test11=project3Dto2D(P1,SS_1,Q1,centerline1[0])[0]
plt.sca(ax2)
ax2.scatter(test11[0],test11[1],c='yellow')
# Set plot labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

fig3 = plt.figure("2d view")
ax3 = fig3.add_axes([0, 0, 1, 1])
plt.xlim([-(px/2), (px/2)])
plt.ylim([-(px/2), (px/2)])
phantom_IM8 = mpimg.imread('phantom_IM8.jpg')
extent = [-px/2, px/2, -px/2, px/2]
ax3.imshow(phantom_IM8,extent=extent, aspect='equal',cmap='bone',alpha=0.8)

test22=project3Dto2D(P2,SS_2,Q2,centerline2[0])[0]

ax3.scatter(test22[0],test22[1],c='yellow')

plt.plot([sublist[0] for sublist in epipolar_total],[sublist[1] for sublist in epipolar_total])
for i in epipolarpoint_local_2D:
    break
    plt.plot(*i,c='r',alpha=0.5)


plt.plot(legVx2_2D, legVy2_2D, color='k', label='Reparameterized Legendre Curve')
plt.plot(x_coord_fit_proj2, y_coord_fit_proj2, color='crimson', label='Reparameterized Legendre Curve_x_coordfirt',alpha=0.5)

# Set plot labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()


plt.show()
