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

vessel3D1 = [[np.array([24.5312, 51.1477, 17.825 ])],
 [np.array([24.5222, 50.6809, 17.5962])],
 [np.array([24.489 , 50.2041, 17.4074])],
 [np.array([24.4399, 49.7211, 17.2458])],
 [np.array([24.3814, 49.2351, 17.101 ])],
 [np.array([24.3189, 48.7483, 16.9646])],
 [np.array([24.2565, 48.2625, 16.8302])],
 [np.array([24.1973, 47.779 , 16.6927])],
 [np.array([24.1436, 47.2988, 16.5487])],
 [np.array([24.0969, 46.8223, 16.3956])],
 [np.array([24.0583, 46.3497, 16.2318])],
 [np.array([24.0282, 45.8811, 16.0566])],
 [np.array([24.0066, 45.4163, 15.8696])],
 [np.array([23.9936, 44.9551, 15.6713])],
 [np.array([23.9885, 44.497 , 15.4622])],
 [np.array([23.9909, 44.0417, 15.2433])],
 [np.array([24.0001, 43.5888, 15.0157])],
 [np.array([24.0154, 43.1376, 14.7805])],
 [np.array([24.0361, 42.688 , 14.539 ])],
 [np.array([24.0613, 42.2393, 14.2924])],
 [np.array([24.1023, 42.0348, 14.0759])],
 [np.array([24.1335, 41.5874, 13.8222])],
 [np.array([24.1676, 41.1401, 13.5665])],
 [np.array([24.2039, 40.6926, 13.3097])],
 [np.array([24.2423, 40.2447, 13.0526])],
 [np.array([24.2823, 39.7964, 12.7956])],
 [np.array([24.3239, 39.3475, 12.5391])],
 [np.array([24.367 , 38.898 , 12.2834])],
 [np.array([24.4115, 38.4479, 12.0288])],
 [np.array([24.4576, 37.9972, 11.7751])],
 [np.array([24.5053, 37.5461, 11.5224])],
 [np.array([24.5548, 37.0945, 11.2705])],
 [np.array([24.6064, 36.6426, 11.0192])],
 [np.array([24.6603, 36.1904, 10.7682])],
 [np.array([24.7168, 35.7382, 10.5172])],
 [np.array([24.776 , 35.2859, 10.2659])],
 [np.array([24.8241, 35.0853, 10.0419])],
 [np.array([24.8875, 34.6339,  9.7886])],
 [np.array([24.9544, 34.1827,  9.5341])],
 [np.array([25.025 , 33.7318,  9.2782])],
 [np.array([25.0992, 33.2811,  9.0209])],
 [np.array([25.1774, 32.8306,  8.7622])],
 [np.array([25.2593, 32.3803,  8.502 ])],
 [np.array([25.3451, 31.9301,  8.2404])],
 [np.array([25.4345, 31.4799,  7.9778])],
 [np.array([25.5274, 31.0296,  7.7145])],
 [np.array([25.6236, 30.579 ,  7.451 ])],
 [np.array([25.7228, 30.128 ,  7.1877])],
 [np.array([25.8246, 29.6763,  6.9253])],
 [np.array([25.8848, 29.4817,  6.6844])],
 [np.array([25.9877, 29.0284,  6.4255])],
 [np.array([26.0918, 28.5739,  6.1698])],
 [np.array([26.1965, 28.1177,  5.9186])],
 [np.array([26.3012, 27.6597,  5.6726])],
 [np.array([26.4055, 27.1997,  5.4331])],
 [np.array([26.5086, 26.7375,  5.2011])],
 [np.array([26.6099, 26.2728,  4.9776])],
 [np.array([26.709 , 25.8056,  4.7637])],
 [np.array([26.8052, 25.3356,  4.5605])],
 [np.array([26.898 , 24.8628,  4.3689])],
 [np.array([26.987 , 24.3871,  4.1898])],
 [np.array([27.0716, 23.9085,  4.0241])],
 [np.array([27.1516, 23.4269,  3.8725])],
 [np.array([27.2266, 22.9424,  3.7355])],
 [np.array([27.2964, 22.4551,  3.6138])],
 [np.array([27.3609, 21.9652,  3.5076])],
 [np.array([27.42  , 21.4726,  3.4171])],
 [np.array([27.4738, 20.9778,  3.3424])],
 [np.array([27.5224, 20.4809,  3.2833])],
 [np.array([27.566 , 19.9821,  3.2394])],
 [np.array([27.605 , 19.4817,  3.2103])],
 [np.array([27.6398, 18.9801,  3.1954])],
 [np.array([27.7815, 18.2161,  3.1945])],
 [np.array([27.8121, 17.7128,  3.206 ])],
 [np.array([27.8401, 17.2092,  3.2286])],
 [np.array([27.8662, 16.7054,  3.2611])],
 [np.array([27.8912, 16.2017,  3.3021])],
 [np.array([27.9157, 15.6984,  3.35  ])],
 [np.array([27.9406, 15.1955,  3.4034])],
 [np.array([27.9665, 14.6933,  3.4607])],
 [np.array([27.9943, 14.1917,  3.5204])],
 [np.array([28.0246, 13.6907,  3.5808])],
 [np.array([28.058 , 13.1904,  3.6405])],
 [np.array([28.0951, 12.6905,  3.6981])],
 [np.array([28.1362, 12.1908,  3.7525])],
 [np.array([28.3243, 11.4294,  3.8128])],
 [np.array([28.3759, 10.9292,  3.8582])],
 [np.array([28.4319, 10.4284,  3.8981])],
 [np.array([28.492 ,  9.9266,  3.9322])],
 [np.array([28.556 ,  9.4235,  3.9609])],
 [np.array([28.6232,  8.9189,  3.9848])],
 [np.array([28.6928,  8.4128,  4.0048])],
 [np.array([28.7639,  7.9052,  4.0228])],
 [np.array([28.8352,  7.3965,  4.0406])],
 [np.array([28.9055,  6.8874,  4.0609])],
 [np.array([28.9733,  6.379 ,  4.0869])],
 [np.array([29.037 ,  5.873 ,  4.1221])]]

vessel3D = []
for list in vessel3D1:
    vessel3D.append(list[0])




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
px = 512-1

for idx in range(1, 230, 25):
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
        parameters = projection_1(vessel3D[i][0], vessel3D[i][1], vessel3D[i][2])
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


    #fig = plt.figure("3D view")
    #ax = fig.add_subplot(111, projection='3d')
    #plt.sca(ax)
#
    #for point in vessel3D:
    #    ax.scatter(*point,c='r',alpha=0.5)
#
    #for point in v_pList:
    #    ax.scatter(*point,c='blue',alpha=0.7)
#
    #ax.scatter(*right_mid,c='lime')
    #ax.scatter(*top_mid, c='yellow')
    #ax.scatter(*Source,c='k')
    #ax.scatter(*D,c='k')
    #x_vals_detector = [[P1[0], SS_1[0], R1[0], Q1[0], P1[0]]]
    #y_vals_detector = [[P1[1], SS_1[1], R1[1], Q1[1], P1[1]]]
    #z_vals_detector = [[P1[2], SS_1[2], R1[2], Q1[2], P1[2]]]
    #plt.plot(x_vals_detector[0], y_vals_detector[0], z_vals_detector[0], c='k')
#
#
#
    #xLabel = ax.set_xlabel('X', linespacing=1)
    #yLabel = ax.set_ylabel('Y', linespacing=1)
    #zLabel = ax.set_zlabel('Z', linespacing=1)
    #ax.set_aspect('equal')  # , adjustable='box')

    fig2 = plt.figure("angiogram_frame_"+str(idx))
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
