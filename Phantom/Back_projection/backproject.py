import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from functionsImport_Aug2023 import *
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
from scipy.spatial.transform import Rotation as R
np.set_printoptions(precision=4, suppress=True)
vflip = 1

vessel3D = np.array([
    [49.7519, 44.2969, 26.5287],
    [35.61  ,48.5665, 21.9344],
    [27.472 , 50.8256, 19.1902],
    [25.9542,  50.9118 ,18.4495],
    [28.4782,  49.6622 ,18.9602],
    [32.692 , 47.8333 ,20.0358],
    [33.5792,  47.0641, 20.1431],
    [43.7785,  43.327 ,23.0157],
    [57.669 , 38.4121 ,27.0157],
    [59.5066,  37.3413, 27.4457],
    [61.3692,  36.269 ,27.8797],
    [62.375 , 35.4882 ,28.0342],
    [64.2786,  34.4151 ,28.4724],
    [65.3028,  33.6344 ,28.6305],
    [66.3239,  32.8525 ,28.7929],
    [66.4561,  32.3704 ,28.6611],
    [66.5844,  31.8901 ,28.5264],
    [65.8352,  31.7144 ,28.0838],
    [65.9573,  31.2385 ,27.9404],
    [65.2064,  31.0674 ,27.4855],
    [64.454 , 30.8979 ,27.025],
    [64.5655,  30.4265 ,26.8702],
    [63.8095,  30.2595 ,26.4007],
    [63.0514,  30.0931 ,25.9273],
    [62.2911,  29.9269 ,25.4507],
    [61.5285,  29.7607 ,24.9714],
    [60.7637,  29.5944 ,24.4899],
    [59.9967,  29.4277 ,24.0065],
    [59.2276,  29.2607 ,23.5218],
    [58.4566,  29.0932 ,23.0358],
    [57.6838,  28.9253 ,22.549],
    [56.9094,  28.7569 ,22.0614],
    [56.1335,  28.5881 ,21.5731],
    [55.3563,  28.4189 ,21.0841],
    [54.5781,  28.2495 ,20.5945],
    [53.7988,  28.08 ,20.1041],
    [53.0187,  27.9103, 19.6128],
    [52.2377,  27.7408, 19.1205],
    [51.4561,  27.5713, 18.6269],
    [50.6737,  27.4021, 18.132],
    [49.8906,  27.2332, 17.6355],
    [49.1067,  27.0646, 17.1372],
    [47.455 , 27.1884 ,16.3202],
    [46.6691,  27.02 ,15.8176],
    [45.8821,  26.8519, 15.3128],
    [45.0937,  26.6841, 14.8056],
    [44.3038,  26.5166, 14.2962],
    [42.6386,  26.6362, 13.4702],
    [41.8449,  26.4681, 12.956],
    [41.049 , 26.2998 ,12.4397],
    [40.2506,  26.1311, 11.9215],
    [38.5708,  26.2447, 11.0901],
    [37.767 , 26.0739 ,10.569],
    [36.0774,  26.1821, 9.7381],
    [35.2677,  26.0084, 9.2156],
    [34.4549,  25.8332, 8.6934],
    [33.639 , 25.6561 ,8.1721],
    [32.8201,  25.4771, 7.6524],
    [31.9984,  25.2957, 7.1351],
    [31.174 , 25.1119 ,6.6208],
    [30.3473,  24.9255, 6.1104],
    [30.4014,  24.4606, 5.9158],
    [29.5711,  24.269 ,5.4158],
    [29.6184,  23.798 ,5.2372],
    [29.6601,  23.3229, 5.0698],
    [29.6957,  22.8438, 4.9145],
    [29.7247,  22.3606, 4.7722],
    [29.7468,  21.8734, 4.6435],
    [28.9199,  21.6664, 4.1936],
    [28.9381,  21.1732, 4.0896],
    [28.9502,  20.6766, 4.0004],
    [28.9561,  20.1769, 3.926],
    [28.9561,  19.6746, 3.8664],
    [28.9506,  19.1702, 3.821],
    [28.9404,  18.6641, 3.7892],
    [28.926 , 18.1571 ,3.7701],
    [28.9084,  17.6496, 3.7627],
    [29.6378,  16.8544, 4.122],
    [29.611 , 16.3486 ,4.1337],
    [29.5866,  15.8441, 4.1523],
    [29.5664,  15.3413, 4.1763],
    [29.5522,  14.8404, 4.2041],
    [29.5456,  14.3415, 4.2342],
    [29.5483,  13.8445, 4.265],
    [29.5618,  13.3491, 4.2951],
    [29.5873,  12.855 ,4.3234],
    [30.4234,  12.0816, 4.694],
    [30.4934,  11.5869, 4.7165],
    [31.4319,  10.8046, 5.0869],
    [31.547 , 10.3027, 5.1066],
    [32.5694,  9.5007, 5.488],
    [32.7001,  8.9883, 5.5065],
    [32.8182,  8.4733, 5.5205],
    [33.7731,  7.6595, 5.8997],
    [33.7861,  7.1549, 5.8899],
    [33.7269,  6.6646, 5.8588]
])
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

def translation_matrix(vec):
    m = np.identity(4)
    m[:3, 3] = vec
    return m

def projection_1(x, y, z):
    point = np.array([x, y, z])
    dirV = point - Source  # S - point
    A = np.array([(r2[0] / np.linalg.norm(r2), r3[0] / np.linalg.norm(r3), - dirV[0] / np.linalg.norm(dirV)),
                  (r2[1] / np.linalg.norm(r2), r3[1] / np.linalg.norm(r3), - dirV[1] / np.linalg.norm(dirV)),
                  (r2[2] / np.linalg.norm(r2), r3[2] / np.linalg.norm(r3), - dirV[2] / np.linalg.norm(dirV))])
    b = np.array([(Source[0] - D[0]),
                  (Source[1] - D[1]),
                  (Source[2] - D[2])])
    result = np.linalg.solve(A, b)
    return result

dicom_file_path = 'C:/Users/Jack/PycharmProjects/pythonProject1/Kirsten Maas Phantom/XX_0001'
# Load the DICOM file
dicom_instance = pydicom.dcmread(dicom_file_path)
# Create an instance of the XRayProjection class and pass the DICOM object
series = xray.XRayProjection(dicom_instance)



d_p = 0.74
px = 512

for idx in range(1, 230, 20):
    M = series.geo[idx].rotation
    source = translation_matrix([0,0,series.distance_source_to_patient])
    source_mat = source @ M
    source_mat_inv = np.linalg.inv(source_mat)
    Source = source_mat_inv[:3, -1]

    iso_center1 = series.geo[idx].iso_center
    detector_translation = translation_matrix(iso_center1)
    detector_mat = detector_translation @ M
    detector_mat_inv = np.linalg.inv(detector_mat)
    D = detector_mat_inv[:3,-1]

    P1 =   mapCenterline(idx,np.array([256,-256]))
    Q1 =   mapCenterline(idx,np.array([256,256]))
    R1 =   mapCenterline(idx,np.array([-256,256]))
    SS_1 = mapCenterline(idx,np.array([-256,-256]))

    right_mid = mapCenterline(idx,np.array([256, 0]))
    r2 = right_mid - D
    top_mid = mapCenterline(idx,np.array([0, 256]))
    r3 = top_mid - D




    back_projection2D = []
    v_pList = []
    for i in range(len(vessel3D)):
        #print(vessel3D[i,0], vessel3D[i,1], vessel3D[i,2])
        parameters = projection_1(vessel3D[i,0], vessel3D[i,1], vessel3D[i,2])
        dirV = vessel3D[i] - Source
        #print(np.linalg.norm(Source-D)-parameters[2])
        x_p = Source[0] + parameters[2] * dirV[0] / np.linalg.norm(dirV)
        y_p = Source[1] + parameters[2] * dirV[1] / np.linalg.norm(dirV)
        z_p = Source[2] + parameters[2] * dirV[2] / np.linalg.norm(dirV)


        #x_p = D[0] + (parameters[0] / np.linalg.norm(r2)) * r2[0] + (parameters[1] / np.linalg.norm(r3)) * r3[0]
        #y_p = D[1] + (parameters[0] / np.linalg.norm(r2)) * r2[1] + (parameters[1] / np.linalg.norm(r3)) * r3[1]
        #z_p = D[2] + (parameters[0] / np.linalg.norm(r2)) * r2[2] + (parameters[1] / np.linalg.norm(r3)) * r3[2]
        v_p = np.array([x_p, y_p, z_p])
        v_pList.append(v_p)
        twoDpoint = project3Dto2D(P1, SS_1, Q1, v_p, 512, d_p)
        back_projection2D.append(twoDpoint[0])


    fig = plt.figure("3D view")
    ax = fig.add_subplot(111, projection='3d')
    plt.sca(ax)

    for point in vessel3D:
        ax.scatter(*point,c='r',alpha=0.5)

    for point in v_pList:
        ax.scatter(*point,c='blue',alpha=0.7)

    ax.scatter(*right_mid,c='lime')
    ax.scatter(*top_mid, c='yellow')
    ax.scatter(*Source,c='k')
    ax.scatter(*D,c='k')
    x_vals_detector = [[P1[0], SS_1[0], R1[0], Q1[0], P1[0]]]
    y_vals_detector = [[P1[1], SS_1[1], R1[1], Q1[1], P1[1]]]
    z_vals_detector = [[P1[2], SS_1[2], R1[2], Q1[2], P1[2]]]
    plt.plot(x_vals_detector[0], y_vals_detector[0], z_vals_detector[0], c='k')



    xLabel = ax.set_xlabel('X', linespacing=1)
    yLabel = ax.set_ylabel('Y', linespacing=1)
    zLabel = ax.set_zlabel('Z', linespacing=1)
    ax.set_aspect('equal')  # , adjustable='box')

    fig2 = plt.figure("angiogram")
    ax2 = fig2.add_axes([0, 0, 1, 1])
    print('idx = ', idx)
    plt.sca(ax2)
    filename = 'frames/frame'+str(idx)+'.png'
    image1 = mpimg.imread(filename)
    image1 = np.fliplr(image1)
    if vflip:
        image1 = np.flipud(image1)
    extent = [-px/2, px/2, -px/2, px/2]
    ax2.imshow(image1, extent=extent, aspect='equal', cmap='bone', alpha=0.8)
    plt.xlim([-px / 2, px / 2])
    plt.ylim([-px / 2, px / 2])
    for point in back_projection2D:
        ax2.scatter(point[0],point[1],c='r',alpha = 0.5,s=1)



plt.show()
print("script ended")
