import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from functionsImport_Calibrate import *
matplotlib.use('tkAgg')
import time
start = time.time()
from scipy.optimize import lsq_linear
import pydicom as dicom
import matplotlib.image as mpimg
import plotly.graph_objects as go
from PIL import Image
import xray
import pydicom

def translation_matrix(vec):
    m = np.identity(4)
    m[:3, 3] = vec
    return m


def placePointPhillips(point):
    pass


dicom_file_path = 'C:/Users/Jack/PycharmProjects/pythonProject1/Kirsten Maas Phantom/XX_0001'
# Load the DICOM file
dicom_instance = pydicom.dcmread(dicom_file_path)
# Create an instance of the XRayProjection class and pass the DICOM object
series = xray.XRayProjection(dicom_instance)


#note frame 48 and 234
# input
alpha =   np.radians( -106.67101192 )
beta  =   np.radians(  3.16435957)
gamma =   np.radians(-12.61899117)

alpha2 =  np.radians( 57.22800966 )
beta2 =   np.radians( -6.11341731  )
gamma2 =  np.radians(6.99863472)

# Angiogram #settings
d_s1 = 810. # Source to patient
d_sd1 =   1195.01239984 # Source to detector
d_d1 = d_sd1 - d_s1  # Patient to detector
d_s2 = 810.
d_sd2 =  1195.01239984
d_d2 = d_sd2 - d_s2
d_p = 0.74   #  # Pixel spacing (/ estimated magnification factor)
px = 512 - 1    # Pixels #note the - 1

#def Rx(angle): #negative angle due to definition of y-axis
#    return np.array([[1, 0, 0],
#                     [0, np.cos(-angle), -np.sin(-angle)],
#                     [0, np.sin(-angle), np.cos(-angle)]])
#def Ry(angle):
#    return np.array([[np.cos(angle), 0, np.sin(angle)],
#                     [0, 1, 0],
#                     [-np.sin(angle), 0, np.cos(angle)]])
#def Rz(angle):
#    return np.array([[np.cos(angle), -np.sin(angle), 0],
#                     [np.sin(angle), np.cos(angle), 0],
#                     [0, 0, 1]])
#def rotation(alpha, beta):
#    M = np.array([(np.cos(alpha), 0, np.sin(alpha)),
#                  (np.sin(-beta) * np.sin(alpha), np.cos(-beta), np.sin(-beta) * -np.cos(alpha)),
#                  (-np.cos(-beta) * np.sin(alpha), np.sin(-beta), np.cos(alpha) * np.cos(-beta))])
#    return M
#
#
###note not sure if gamma needs to be + or -
#type = "Phillips" #Phillips rotational matrix calibrated
#if 0:
#    if type == "Phillips":
#        M= Rx(beta) @ Ry(alpha) @ Rz(gamma)
#        M2 = Rx(beta2) @ Ry(alpha2) @ Rz(gamma2)
#    else:
#        M = rotation(alpha,beta)
#        M2 = rotation(alpha2,beta2)


v_flip = 1
def mapCenterline(idx, pixelPoint, h_flip = True, vert_flip = False):
    M = series.geo[idx].rotation
    iso_center1 = series.geo[idx].iso_center
    pixel_spacing = series.imager_pixel_spacing[1]
    pixelPoint_use = pixelPoint.copy()
    if h_flip:
        worldPoint = np.array([iso_center1[0] + pixelPoint_use[0] * -pixel_spacing,
                               iso_center1[1] + pixelPoint_use[1] * pixel_spacing,
                               iso_center1[2]])
    elif vert_flip:
        worldPoint = np.array([iso_center1[0] + pixelPoint_use[0] * pixel_spacing,
                               iso_center1[1] + pixelPoint_use[1] * -pixel_spacing,
                               iso_center1[2]])
    elif vert_flip and h_flip:
        worldPoint = np.array([iso_center1[0] + pixelPoint_use[0] * -pixel_spacing,
                               iso_center1[1] + pixelPoint_use[1] * -pixel_spacing,
                               iso_center1[2]])
    else:
        worldPoint = np.array([iso_center1[0] + pixelPoint_use[0] * pixel_spacing,
                               iso_center1[1] + pixelPoint_use[1] * pixel_spacing,
                               iso_center1[2]])
    translation_vector = translation_matrix(worldPoint)

    point_mat = translation_vector @ M
    point_mat_inv = np.linalg.inv(point_mat)
    point_location = point_mat_inv[:3, -1]

    return point_location

#Set-up detector 1: frame 48
M = series.geo[47].rotation

source = translation_matrix([0,0,series.distance_source_to_patient])
source_mat = source @ M
source_mat_inv = np.linalg.inv(source_mat)
Source1 = source_mat_inv[:3, -1]

iso_center1 = series.geo[47].iso_center
detector_translation = translation_matrix(iso_center1)
detector_mat = detector_translation @ M
detector_mat_inv = np.linalg.inv(detector_mat)
Detector1 = detector_mat_inv[:3,-1]

topmid = mapCenterline(47,[0, 512./2],h_flip=True,vert_flip=True)
r2 = (topmid - Detector1) / np.linalg.norm(topmid - Detector1)
midright = mapCenterline(47,[512./2, 0],h_flip=True,vert_flip=True)
r3 = (midright - Detector1) / np.linalg.norm(midright - Detector1)


#Set-up detector 2: frame 234
M2 = series.geo[233].rotation
source_mat2 = source @ M2
source_mat_inv2 = np.linalg.inv(source_mat2)
Source2 = source_mat_inv2[:3, -1]

iso_center2 = series.geo[233].iso_center
detector_translation2 = translation_matrix(iso_center2)
detector_mat2 = detector_translation2 @ M2
detector_mat_inv2 = np.linalg.inv(detector_mat2)
Detector2 = detector_mat_inv2[:3, -1]

topmid2 = mapCenterline(233,[0, 512./2],h_flip=True,vert_flip=True)
r2_2 = (topmid2 - Detector2) / np.linalg.norm(topmid2 - Detector2)
midright2 = mapCenterline(233,[512./2, 0],h_flip=True,vert_flip=True)
r3_2 = (midright2 - Detector2) / np.linalg.norm(midright2 - Detector2)

P1 =   mapCenterline(47,np.array([256,-256]))
Q1 =   mapCenterline(47,np.array([256,256]))
R1 =   mapCenterline(47,np.array([-256,256]))
SS_1 = mapCenterline(47,np.array([-256,-256]))

P2 = mapCenterline(233,np.array([256,-256]))
Q2 = mapCenterline(233,np.array([256,256]))
R2 = mapCenterline(233,np.array([-256,256]))
SS_2 = mapCenterline(233,np.array([-256,-256]))

view1 = np.array([   #frame 48
  [             -22.66,         124   ,         0],
  [            -18.254,      114.56   ,    10.399],
  [            -15.107,      106.38   ,    19.149],
  [            -10.701,      98.824   ,    27.877],
  [            -5.6651,      90.641   ,    37.466],
  [                  0,      81.829   ,    47.922],
  [             4.4062,      75.534   ,     55.59],
  [             8.1829,       68.61   ,    63.462],
  [             10.071,      62.945   ,    69.421],
  [             10.701,      52.244   ,     80.12],
  [             10.071,      43.432   ,    88.937],
  [             10.071,      36.508   ,    95.848],
  [             10.071,      31.473   ,    100.87]
])

view2 = np.array([      #frame 234
  [            3.1473,       96.306,            0],
  [           0.62945,       86.864,       9.7526],
  [           -3.7767,       78.052,       19.586],
  [           -8.8123,       69.869,       29.175],
  [           -13.218,       63.574,       36.844],
  [           -15.736,       56.651,       44.197],
  [           -19.513,       50.985,       50.992],
  [           -22.031,       44.062,       58.345],
  [           -23.919,       38.396,       64.305],
  [           -24.549,       30.214,       72.496],
  [           -24.549,       22.031,       80.663],
  [           -25.807,       15.107,       87.687],
  [           -26.437,       8.8123,           94]
])

proj1_2DX =  view1[:,0]*-1
if v_flip:
    proj1_2DY =  view1[:,1]*-1
else:
    proj1_2DY = view1[:, 1]
ss        =  view1[:,2]

proj2_2DX =  view2[:,0]*-1
if v_flip:
    proj2_2DY =  view2[:,1]*-1
else:
    proj2_2DY = view2[:, 1]
ss2       =  view2[:,2]

noPoints = 100
step_t = (ss[-1] / (noPoints))
step_s = (ss2[-1] / (noPoints))
t = list(np.arange(0, ss[-1], step_t)) # t is parameter for projection 1
s = list(np.arange(0, ss2[-1], step_s)) # s is parameter for projection 2

best_order_fit = 10 # order of best fit
# --- Fit legendre polynomial for X of projections with best order ---
legendre_coefs_x_proj1 = np.polynomial.legendre.legfit(ss, proj1_2DX, best_order_fit) #Least squares fit of Legendre series to data, coefficients
x_coord_fit_proj1 = np.polynomial.legendre.legval(t, legendre_coefs_x_proj1) #Evaluate a Legendre series at points x
legendre_coefs_y_proj1 = np.polynomial.legendre.legfit(ss, proj1_2DY, best_order_fit)
y_coord_fit_proj1 = np.polynomial.legendre.legval(t, legendre_coefs_y_proj1)

#second fit
legendre_coefs_x_proj2 = np.polynomial.legendre.legfit(ss2, proj2_2DX, best_order_fit) #Least squares fit of Legendre series to data, coefficients
x_coord_fit_proj2 = np.polynomial.legendre.legval(s, legendre_coefs_x_proj2) #Evaluate a Legendre series at points x
legendre_coefs_y_proj2 = np.polynomial.legendre.legfit(ss2, proj2_2DY, best_order_fit)
y_coord_fit_proj2 = np.polynomial.legendre.legval(s, legendre_coefs_y_proj2)

legVx =  x_coord_fit_proj1
legVy =  y_coord_fit_proj1
legVx2 = x_coord_fit_proj2
legVy2 = y_coord_fit_proj2

world_points_1 = []
world_points_2 = []
for p in range(len(legVx)):
    temp_point_1 = np.array([legVx[p],legVy[p]])
    world_p = mapCenterline(47, temp_point_1, h_flip=True, vert_flip=True)
    world_points_1.append(world_p)

    temp_point_2 = np.array([legVx2[p], legVy2[p]])
    world_p2 = mapCenterline(233, temp_point_2, h_flip=True, vert_flip=True)
    world_points_2.append(world_p2)


legVx = [value * -d_p for value in legVx]
legVy = [value * d_p for value in legVy]
legVz = d_d1 * np.ones_like(legVx)

legVx_2D = [value / -d_p for value in legVx]
legVy_2D = [value / d_p for value in legVy]

legVx2 = [value * -d_p for value in legVx2]
legVy2 = [value * d_p for value in legVy2]
legVz2 = d_d2 * np.ones_like(legVx2)

legVx2_2D = [value / -d_p for value in legVx2]
legVy2_2D = [value / d_p for value in legVy2]

#transfer variable names
centerline1 = world_points_1

reconstructed_points = []
total_reconstructed_points = []
epipolar_total = []
epipolar_total2 = []

r4 = Source2 - Source1
NoIntersectionsPerPoint = np.zeros(len(world_points_1))
sign_change_location_total = []


test_list = [3,49,79]
test_colors = ['lime','cyan','red']
matches = []

for iPoint in range(len(centerline1)): #len(centerline1)
    r5 = centerline1[iPoint] - Source2
    AA = np.array([(r2_2[0] / np.linalg.norm(r2_2), r3_2[0] / np.linalg.norm(r3_2), -r4[0] / np.linalg.norm(r4),
                    -r5[0] / np.linalg.norm(r5)),
                   (r2_2[1] / np.linalg.norm(r2_2), r3_2[1] / np.linalg.norm(r3_2), -r4[1] / np.linalg.norm(r4),
                    -r5[1] / np.linalg.norm(r5)),
                   (r2_2[2] / np.linalg.norm(r2_2), r3_2[2] / np.linalg.norm(r3_2), -r4[2] / np.linalg.norm(r4),
                    -r5[2] / np.linalg.norm(r5))])
    BB = np.array([Source2[0] - Detector2[0], Source2[1] - Detector2[1], Source2[2] - Detector2[2]])
    n1 = np.cross(r2_2, r3_2)  # normal on detectorplane 1
    n2 = np.cross(r4, r5)  # normal on epipolar plane
    r_epi = np.cross(n1, n2)  # cross product of the  two normals of the planes. This equals to the directional vector of the epipolar line.
    bounds = ([- px / 2 * d_p, - px / 2 * d_p, -np.inf, -np.inf], [px / 2 * d_p, px / 2 * d_p, np.inf, np.inf])
    res = lsq_linear(AA, BB, bounds=bounds, method='bvls', lsmr_tol='auto', verbose=0)
    lam2 = res.x[0]  # dit zijn de lamdas voor een punt op de snijlijn van de 2 vlakken
    lam3 = res.x[1]

    x_epipolar_point = Detector2[0] + (lam2 / np.linalg.norm(r2_2)) * r2_2[0] + (lam3 / np.linalg.norm(r3_2)) * r3_2[0]
    y_epipolar_point = Detector2[1] + (lam2 / np.linalg.norm(r2_2)) * r2_2[1] + (lam3 / np.linalg.norm(r3_2)) * r3_2[1]
    z_epipolar_point = Detector2[2] + (lam2 / np.linalg.norm(r2_2)) * r2_2[2] + (lam3 / np.linalg.norm(r3_2)) * r3_2[2]

    # calculation of lamda range which limits epipolar line to detector
    L1 = calculate_lambda_JT([x_epipolar_point,y_epipolar_point,z_epipolar_point],r_epi,P2,SS_2)
    L2 = calculate_lambda_JT([x_epipolar_point,y_epipolar_point,z_epipolar_point],r_epi,P2,Q2)
    L3 = calculate_lambda_JT([x_epipolar_point,y_epipolar_point,z_epipolar_point],r_epi,Q2,R2)
    L4 = calculate_lambda_JT([x_epipolar_point,y_epipolar_point,z_epipolar_point],r_epi,R2,SS_2)
    lamda_list = [L1,L2,L3,L4]
    lamda_list = sorted(lamda_list)
    min_lambda_epi = lamda_list[1]
    max_lamda_epi = lamda_list[2]
    a = np.linspace( min_lambda_epi, max_lamda_epi , 100)
    x_epipolarline = np.zeros(len(a))
    y_epipolarline = np.zeros(len(a))
    z_epipolarline = np.zeros(len(a))

    epipolarpoint_local_2D = []
    for k in range(len(a)):
        x_value = x_epipolar_point + ( a[k] * r_epi[0] / np.linalg.norm(r_epi) )
        y_value = y_epipolar_point + ( a[k] * r_epi[1] / np.linalg.norm(r_epi) )
        z_value = z_epipolar_point + ( a[k] * r_epi[2] / np.linalg.norm(r_epi) )

        x_epipolarline[k] = x_value
        y_epipolarline[k] = y_value
        z_epipolarline[k] = z_value

        epipolarpoint_local_2D.append(project3Dto2D(P2, SS_2, Q2,[x_value, y_value, z_value],px,d_p)[0])
        epipolar_total.append(project3Dto2D(P2, SS_2, Q2, [x_value, y_value, z_value], px, d_p)[0])

    dirV_epi_2D = epipolarpoint_local_2D[0]-epipolarpoint_local_2D[1]
    dirV_g = np.dot(dirV_epi_2D, np.array(((0, -1), (1, 0))))

    prev_delta_y = 0
    sign_change_location = []

    for point in range(len(legVx2_2D)):                                                                  #note: this loop outputs coordianates of intersection under variable "sign_change_location"
        intersection_d_epi = intersection_two_lines_2D(np.array([legVx2_2D[point], legVy2_2D[point]]), dirV_g, epipolarpoint_local_2D[0], dirV_epi_2D)
        delta_y = legVy2_2D[point]-intersection_d_epi[1]
        sign_delta_y = np.sign(delta_y)
        if point != 0 and sign_delta_y != prev_delta_y:
            sign_change_location.append(np.array( ([legVx2_2D[point], legVy2_2D[point]]) ) )
            sign_change_location_total.append(np.array(([legVx2_2D[point], legVy2_2D[point]])))
        prev_delta_y = sign_delta_y

    if len(sign_change_location) == 1:
        subList = []
        epi_curve_intersect3D = mapCenterline(233, sign_change_location[0])
        intersection = xRayIntersection(Source1, centerline1[iPoint], Source2,    #intersection = xRayIntersection(S, [curveValsX[iPoint], curveValsY[iPoint], curveValsZ[iPoint]], S2,
                                        epi_curve_intersect3D,
                                        domain=[(-600, 600), (-600, 600),
                                                (-800, 600)])  # domain=[(-30, 30), (-40, 30), (-80, -20)]
        intersection = np.around(intersection.astype(np.double), 4)
        subList.append(intersection)
        #print(iPoint, intersection)
        NoIntersectionsPerPoint[iPoint] = 1
        reconstructed_points.append(subList)
        total_reconstructed_points.append(intersection)
        #matches.append(epi_curve_intersect3D)

    elif len(sign_change_location) > 1:
        #print("len sign change=",len(sign_change_location))
        NoIntersectionsPerPoint[iPoint] = len(sign_change_location)
        subList = []
        for i in range(len(sign_change_location)):
            epi_curve_intersect3D = mapCenterline(233,sign_change_location[i])
            intersection = xRayIntersection(Source1, centerline1[iPoint], Source2,
                                            epi_curve_intersect3D,
                                            domain=[(-300, 300), (-400, 300),
                                                    (-800, 200)])  # domain=[(-30, 30), (-40, 30), (-80, -20)]
            intersection = np.around(intersection.astype(np.double), 4)
            subList.append(intersection)
            total_reconstructed_points.append(intersection)
            #print(iPoint,intersection)
        reconstructed_points.append(subList)


####_______________ PLOTTING ________________######
fig = plt.figure("3D view")
ax = fig.add_subplot(111, projection='3d')
plt.sca(ax)

ax.scatter(*Detector1,c='navy')
ax.scatter(*Source1,c='navy')
ax.scatter(*Detector2,c='k')
ax.scatter(*Source2,c='k')

for p in world_points_1:
    ax.scatter(*p,c='b',alpha=0.2)

for p in world_points_2:
    ax.scatter(*p,c='grey',alpha=0.2)

# plot van detector dmv hoekpunten
x_vals_detector = [[P1[0], SS_1[0], R1[0], Q1[0], P1[0]],[P2[0], SS_2[0], R2[0], Q2[0], P2[0]]]
y_vals_detector = [[P1[1], SS_1[1], R1[1], Q1[1], P1[1]],[P2[1], SS_2[1], R2[1], Q2[1], P2[1]]]
z_vals_detector = [[P1[2], SS_1[2], R1[2], Q1[2], P1[2]],[P2[2], SS_2[2], R2[2], Q2[2], P2[2]]]
plt.plot(x_vals_detector[0], y_vals_detector[0], z_vals_detector[0], c='navy')
plt.plot(x_vals_detector[1], y_vals_detector[1], z_vals_detector[1], c='k')

for sublist in reconstructed_points:
    ax.scatter(*sublist[0],c='fuchsia',alpha=0.4)

#ax.scatter(*reconstructed_points[0][0],c=test_colors[0],alpha=0.95)
#ax.scatter(*reconstructed_points[1][0],c=test_colors[1],alpha=0.95)
#ax.scatter(*reconstructed_points[2][0],c=test_colors[2],alpha=0.95)
#
#ax.scatter(*matches[0],c=test_colors[0],alpha=0.95)
#ax.scatter(*matches[1],c=test_colors[1],alpha=0.95)
#ax.scatter(*matches[2],c=test_colors[2],alpha=0.95)
#
#ax.scatter(*world_points_1[test_list[0]],c=test_colors[0],alpha=0.95)
#ax.scatter(*world_points_1[test_list[1]],c=test_colors[1],alpha=0.95)
#ax.scatter(*world_points_1[test_list[2]],c=test_colors[2],alpha=0.95)

curveValsX = []
curveValsY = []
curveValsZ = []
curveValsX2 = []
curveValsY2 = []
curveValsZ2 = []

#for i in range(3):
#    k = test_list[i]
#    dirVector = centerline1[k] - Source1
#    # lamda = np.linspace(0,np.linalg.norm(dirVector),n_eval_curve)
#    lamda = np.linalg.norm(dirVector)
#    value = Source1 + lamda * dirVector / np.linalg.norm(dirVector)
#
#    curveValsX.append(value[0])
#    curveValsY.append(value[1])
#    curveValsZ.append(value[2])
#
#    dirVector2 = matches[i] - Source2
#    lamda2 = np.linalg.norm(dirVector2)
#    value2 = Source2 + lamda2 * dirVector2 / np.linalg.norm(dirVector2)
#    curveValsX2.append(value2[0])
#    curveValsY2.append(value2[1])
#    curveValsZ2.append(value2[2])

#plot test rays
#for kk in range(3):
#    plt.plot([Source1[0], curveValsX[0]], [Source1[1], curveValsY[kk]], [Source1[2], curveValsZ[kk]], color=test_colors[kk], alpha=0.4)
#    plt.plot([Source2[0], curveValsX2[kk]], [Source2[1], curveValsY2[kk]], [Source2[2], curveValsZ2[kk]], color=test_colors[kk], alpha=0.4)


plt.grid(False)
plt.axis('off')

ax.scatter(topmid[0], topmid[1], topmid[2], c='yellow', alpha=0.4)
ax.scatter(topmid2[0], topmid2[1], topmid2[2], c='yellow', alpha=0.4)

xLabel = ax.set_xlabel('X', linespacing=1)
yLabel = ax.set_ylabel('Y', linespacing=1)
zLabel = ax.set_zlabel('Z', linespacing=1)
ax.set_aspect('equal')  # , adjustable='box')

fig2 = plt.figure("1st view")
ax2 = fig2.add_axes([0, 0, 1, 1])
plt.xlim([-(px/2), (px/2)])
plt.ylim([-(px/2), (px/2)])
image1 = mpimg.imread('frame48_padded.png')
image1 = np.fliplr(image1)
if v_flip:
    image1 = np.flipud(image1)
extent = [-px/2, px/2, -px/2, px/2]
ax2.imshow(image1, extent=extent, aspect='equal', cmap='bone', alpha=0.8)


plt.plot(legVx_2D, legVy_2D, color='k', label='Reparameterized Legendre Curve',alpha=0.5) #todo legVx_2D is in pixels, moet dat
plt.plot(x_coord_fit_proj1, y_coord_fit_proj1, color='crimson', label='Reparameterized Legendre Curve',alpha=0.5)
ax2.legend()
plt.sca(ax2)
# Set plot labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

fig3 = plt.figure("2d view")
ax3 = fig3.add_axes([0, 0, 1, 1])
plt.xlim([-(px/2), (px/2)])
plt.ylim([-(px/2), (px/2)])
Image2 = mpimg.imread('frame234_padded.png')
Image2 = np.fliplr(Image2)
if v_flip:
    Image2 = np.flipud(Image2)
extent = [-px / 2, px / 2, -px / 2, px / 2]
ax3.imshow(Image2, extent=extent, aspect='equal', cmap='bone', alpha=0.8)

plt.plot(legVx2_2D, legVy2_2D, color='k', label='Reparameterized Legendre Curve')
plt.plot(x_coord_fit_proj2, y_coord_fit_proj2, color='crimson',alpha=0.5)

plt.plot([sublist[0] for sublist in epipolar_total],[sublist[1] for sublist in epipolar_total],c='r',alpha=0.4)



# Set plot labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
print("script ended")
