import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import functionsImport_Jul2023
matplotlib.use('TkAgg')

# input
alpha =     np.radians(90)
beta =      np.radians(90)
alpha2 =    np.radians( 0)
beta2 =     np.radians( 0)

# Angiogram #settings
d_s1 = 816#   900          # 900  # Source to patient was 720
d_sd1 =   1118   #     1200  # Source to detector 1188 eerst
d_d1 = d_sd1 - d_s1  # Patient to detector
d_s2 = 815 #   900
d_sd2 =   1118   #
d_d2 = d_sd2 - d_s2
d_p = 0.29296875 # 0.154#        0.390625  # Pixel spacing
px = 1024  #1024  # Pixels (512)

def rotation(alpha, beta):
    M = np.array([(np.cos(alpha), 0, np.sin(alpha)),
                  (np.sin(-beta) * np.sin(alpha), np.cos(-beta), np.sin(-beta) * -np.cos(alpha)),
                  (-np.cos(-beta) * np.sin(alpha), np.sin(-beta), np.cos(alpha) * np.cos(-beta))])
    return M

M = rotation(alpha, beta)
M2 = rotation(alpha2, beta2)

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


virtualVesselx, virtualVessely, virtualVesselz = functionsImport_Jul2023.readjson('CT-centerlines/centerlines_LAD_2.json') #LAD_2

#position 3D centerline to be within frame
for i in range(len(virtualVesselx)):
    #virtualVesselx[i] -= 40         #-20
    virtualVessely[i] += 35         #+45
    virtualVesselz[i] += 70

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

angioPoints = []
angioPoints2 = []
vp_list = []
vp_list2 = []
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

for point in range(len(virtualVesselx)):
    parameters = projection_1(virtualVesselx[point], virtualVessely[point], virtualVesselz[point])
    x_p = D[0] + (parameters[0] / np.linalg.norm(r2)) * r2[0] + (parameters[1] / np.linalg.norm(r3)) * r3[0]
    y_p = D[1] + (parameters[0] / np.linalg.norm(r2)) * r2[1] + (parameters[1] / np.linalg.norm(r3)) * r3[1]
    z_p = D[2] + (parameters[0] / np.linalg.norm(r2)) * r2[2] + (parameters[1] / np.linalg.norm(r3)) * r3[2]
    v_p = np.array([x_p, y_p, z_p])
    vp_list.append(v_p)
    twoDpoint = functionsImport_Jul2023.project3Dto2D(P1, SS_1, Q1, v_p,px,d_p)
    angioPoints.append(twoDpoint[0])
    # note: second projection starts here:
    parameters2 = projection_2(virtualVesselx[point], virtualVessely[point], virtualVesselz[point])
    x_p2 = D2[0] + (parameters2[0] / np.linalg.norm(r2_2)) * r2_2[0] + (parameters2[1] / np.linalg.norm(r3_2)) * r3_2[0]
    y_p2 = D2[1] + (parameters2[0] / np.linalg.norm(r2_2)) * r2_2[1] + (parameters2[1] / np.linalg.norm(r3_2)) * r3_2[1]
    z_p2 = D2[2] + (parameters2[0] / np.linalg.norm(r2_2)) * r2_2[2] + (parameters2[1] / np.linalg.norm(r3_2)) * r3_2[2]
    v_p2 = np.array([x_p2, y_p2, z_p2])
    vp_list2.append(v_p2)
    twoDpoint_2 = functionsImport_Jul2023.project3Dto2D(P2, SS_2, Q2, v_p2,px,d_p)
    angioPoints2.append(twoDpoint_2[0])

### The part below transfers the angiogram points, into x,y,s matrix, with s a parameter
# that runs from 0 to the length of the curve
dataPoints = angioPoints
dataPoints2 = angioPoints2
testData = np.zeros((len(dataPoints), 3))
testData2 = np.zeros((len(dataPoints2), 3))
testData[0][0] = dataPoints[0][0]
testData[0][1] = dataPoints[0][1]
testData2[0][0] = dataPoints2[0][0]
testData2[0][1] = dataPoints2[0][1]
proj1_2DX = []
proj1_2DY = []
proj2_2DX = []
proj2_2DY = []
ss=[]
ss2=[]

for point in range(1, len(dataPoints)-1): #-1 hoort niet
    x_prev = dataPoints[point - 1][0]
    y_prev = dataPoints[point - 1][1]
    x = dataPoints[point][0]
    y = dataPoints[point][1]
    dx = x - x_prev
    dy = y - y_prev
    dS = np.sqrt((dx) ** 2 + (dy) ** 2)
    testData[point][2] = dS + testData[point - 1][2]
    ss.append( dS + testData[point - 1][2])
    testData[point][0] = x
    testData[point][1] = y
    proj1_2DX.append(x)
    proj1_2DY.append(y)

for point in range(1, len(dataPoints2)-1):
    x_prev2 = dataPoints2[point - 1][0]
    y_prev2 = dataPoints2[point - 1][1]
    x2 = dataPoints2[point][0]
    y2 = dataPoints2[point][1]
    dx2 = x2 - x_prev2
    dy2 = y2 - y_prev2
    dS2 = np.sqrt((dx2) ** 2 + (dy2) ** 2)
    testData2[point][2] = dS2 + testData2[point - 1][2]
    testData2[point][0] = x2
    testData2[point][1] = y2
    ss2.append(dS2 + testData2[point - 1][2])
    proj2_2DX.append(x2)
    proj2_2DY.append(y2)




noPoints = 100#125
step_t = (ss[-1] / (noPoints))
step_s = (ss2[-1] / (noPoints))
t = list(np.arange(0, ss[-1], step_t)) # t is parameter for projection 1
s = list(np.arange(0, ss2[-1], step_s)) # s is parameter for projection 2


best_order_fit = 15 # order of best fit

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


    #ax.scatter(virtualVesselx, virtualVessely, virtualVesselz, color='b')

    ax.scatter(S[0], S[1], S[2], color='k')  # source
    ax.scatter(D[0], D[1], D[2], color='b')  # detectormid
    ax.scatter(S2[0], S2[1], S2[2], color='k')  # source
    ax.scatter(D2[0], D2[1], D2[2], color='b')  # detectormid

    x_vals_detector = [[P1[0], SS_1[0], R1[0], Q1[0], P1[0]], [P2[0], SS_2[0], R2[0], Q2[0], P2[0]]]
    y_vals_detector = [[P1[1], SS_1[1], R1[1], Q1[1], P1[1]], [P2[1], SS_2[1], R2[1], Q2[1], P2[1]]]
    z_vals_detector = [[P1[2], SS_1[2], R1[2], Q1[2], P1[2]], [P2[2], SS_2[2], R2[2], Q2[2], P2[2]]]
    plt.plot(x_vals_detector[0], y_vals_detector[0], z_vals_detector[0], c='k')
    plt.plot(x_vals_detector[1], y_vals_detector[1], z_vals_detector[1], c='k')

    for i in range(0, len(angioPoints2),10):
        ax.scatter(vp_list[i][0], vp_list[i][1], vp_list[i][2], color='r', alpha=0.5)
        ax.scatter(vp_list2[i][0], vp_list2[i][1], vp_list2[i][2], color='g', alpha=0.5)

    # ax.scatter(vp_list[0][0],vp_list[0][1],vp_list[0][2],c='k')
    # Plotting detector midlinepoints in 3D
    ax.scatter(topmid_3d[0], topmid_3d[1], topmid_3d[2], c='yellow', alpha=0.4)
    ax.scatter(botmid_3d[0], botmid_3d[1], botmid_3d[2], c='k', alpha=0.4)
    ax.scatter(rightmid_3d[0], rightmid_3d[1], rightmid_3d[2], c='k', alpha=0.4)
    ax.scatter(leftmid_3d[0], leftmid_3d[1], leftmid_3d[2], c='k', alpha=0.4)
    ax.scatter(topmid_3d_2[0], topmid_3d_2[1], topmid_3d_2[2], c='yellow', alpha=0.4)
    ax.scatter(botmid_3d_2[0], botmid_3d_2[1], botmid_3d_2[2], c='k', alpha=0.4)
    ax.scatter(rightmid_3d_2[0], rightmid_3d_2[1], rightmid_3d_2[2], c='k', alpha=0.4)
    ax.scatter(leftmid_3d_2[0], leftmid_3d_2[1], leftmid_3d_2[2], c='k', alpha=0.4)

    plt.plot(virtualVesselx,virtualVessely,virtualVesselz,c='b')
    ax.set_aspect('equal') #, adjustable='box')

    plt.xlabel('x-axis', fontsize=20)
    plt.ylabel('y-axis', fontsize=20)

    #plt.show()


    fig2 = plt.figure("1st view, red")
    plt.xlim([-px/2, px/2])
    plt.ylim([-px/2, px/2])
    plt.plot(x_coord_fit_proj1, y_coord_fit_proj1, label='Legendre fit_proj', c='r',alpha=0.9)

    for i in range(len(angioPoints)):
        plt.scatter(angioPoints[i][0], angioPoints[i][1], color='b')

    fig3 = plt.figure("2d view, yelow")


    plt.xlim([-px/2, px/2])
    plt.ylim([-px/2, px/2])


    for i in range(len(angioPoints2)):
        plt.scatter(angioPoints2[i][0], angioPoints2[i][1], color='b')

    plt.plot(x_coord_fit_proj2, y_coord_fit_proj2, label='Legendre fit_proj', c='r')

    plt.show()

