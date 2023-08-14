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
R2 = (topmid - Detector1) / np.linalg.norm(topmid - Detector1)
midright = mapCenterline(47,[512./2, 0],h_flip=True,vert_flip=True)
R3 = (midright - Detector1) / np.linalg.norm(midright - Detector1)


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
R2_2 = (topmid - Detector2) / np.linalg.norm(topmid - Detector2)
midright2 = mapCenterline(233,[512./2, 0],h_flip=True,vert_flip=True)
R3_2 = (midright - Detector2) / np.linalg.norm(midright - Detector2)

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

# Make empty lists for legendre polynomial fits
fit_list_x_proj1 = []
x_coord_fit_proj1 = []
fit_list_y_proj1 = []
y_coord_fit_proj1 = []
fit_list_x_proj2 = []
x_coord_fit_proj2 = []
fit_list_y_proj2 = []
y_coord_fit_proj2 = []

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
centerline1 = []
legVx_2D = [value / -d_p for value in legVx]
legVy_2D = [value / d_p for value in legVy]

legVx2 = [value * -d_p for value in legVx2]
legVy2 = [value * d_p for value in legVy2]
legVz2 = d_d2 * np.ones_like(legVx2)

centerline2 = []
legVx2_2D = [value / -d_p for value in legVx2]
legVy2_2D = [value / d_p for value in legVy2] 




####_______________ PLOTTING ________________######
fig = plt.figure("3D view")
ax = fig.add_subplot(111, projection='3d')
plt.sca(ax)

ax.scatter(*Detector1,c='navy')
ax.scatter(*Source1,c='navy')
ax.scatter(*Detector2,c='k')
ax.scatter(*Source2,c='k')

for p in world_points_1:
    ax.scatter(*p,c='lime')

for p in world_points_2:
    ax.scatter(*p,c='lime')

# plot van detector dmv hoekpunten
x_vals_detector = [[P1[0], SS_1[0], R1[0], Q1[0], P1[0]],[P2[0], SS_2[0], R2[0], Q2[0], P2[0]]]
y_vals_detector = [[P1[1], SS_1[1], R1[1], Q1[1], P1[1]],[P2[1], SS_2[1], R2[1], Q2[1], P2[1]]]
z_vals_detector = [[P1[2], SS_1[2], R1[2], Q1[2], P1[2]],[P2[2], SS_2[2], R2[2], Q2[2], P2[2]]]
plt.plot(x_vals_detector[0], y_vals_detector[0], z_vals_detector[0], c='navy')
plt.plot(x_vals_detector[1], y_vals_detector[1], z_vals_detector[1], c='k')



ax.scatter(topmid[0], topmid[1], topmid[2], c='yellow', alpha=0.4)
ax.scatter(topmid2[0], topmid2[1], topmid2[2], c='yellow', alpha=0.4)


xLabel = ax.set_xlabel('X', linespacing=1)
yLabel = ax.set_ylabel('Y', linespacing=1)
zLabel = ax.set_zlabel('Z', linespacing=1)
ax.set_aspect('equal')  # , adjustable='box')
plt.show()


fig2 = plt.figure("1st view")
#fig2.canvas.mpl_connect('button_press_event', onclick)
# Connect the button press and release events
#fig2.canvas.mpl_connect('button_press_event', on_button_press)
#fig2.canvas.mpl_connect('button_release_event', on_button_release_1)
ax2 = fig2.add_axes([0, 0, 1, 1])
plt.xlim([-(px/2), (px/2)])
plt.ylim([-(px/2), (px/2)])
image1 = mpimg.imread('frame48_padded.png')
image1 = np.fliplr(image1)
if v_flip:
    image1 = np.flipud(image1)
extent = [-px/2, px/2, -px/2, px/2]
ax2.imshow(image1, extent=extent, aspect='equal', cmap='bone', alpha=0.8)


#x_coords_epi_d1 = [point[0] for point in epipolarpoint_local_2D_func]
#y_coords_epi_d1 = [point[1] for point in epipolarpoint_local_2D_func]
#plt.plot(x_coords_epi_d1, y_coords_epi_d1,c='r')


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
plt.plot(x_coord_fit_proj2, y_coord_fit_proj2, color='crimson', label='Reparameterized Legendre Curve_x_coordfirt',alpha=0.5)

# Set plot labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()


plt.show()
print("script ended")
