import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import math
import time
start = time.time()
matplotlib.use('TkAgg')
from scipy.optimize import lsq_linear
from functionsImport_Jul2023 import *
import virtualAngiogram_Jul2023

def reconstruct3D(Point,d_s,d_sd,d_p,px):
    epipolarpoint_local_2D_func = []
    betaF =   virtualAngiogram_Jul2023.beta
    alphaF =  virtualAngiogram_Jul2023.alpha
    beta2F =  virtualAngiogram_Jul2023.beta2
    alpha2F = virtualAngiogram_Jul2023.alpha2
    d_sF = d_s
    d_sdF = d_sd
    d_dF = d_sdF - d_sF  # Patient to detector
    d_pF = d_p  # 0.390625  # Pixel spacing
    pxF = px  # 512  # Pixels
    MF = virtualAngiogram_Jul2023.rotation(alphaF, betaF)
    M2F = virtualAngiogram_Jul2023.rotation(alpha2F, beta2F)
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

beta = virtualAngiogram_Jul2023.beta
alpha = virtualAngiogram_Jul2023.alpha
beta2 = virtualAngiogram_Jul2023.beta2
alpha2 = virtualAngiogram_Jul2023.alpha2
# Angiogram #settings

d_s1 = virtualAngiogram_Jul2023.d_s1 #788.2679#   900          # 900  # Source to patient was 720
d_sd1 =  virtualAngiogram_Jul2023.d_sd1 #1108   #     1200  # Source to detector 1188 eerst
d_d1 =  d_sd1 - d_s1  # Patient to detector
d_s2 = virtualAngiogram_Jul2023.d_s2 #809.8909#   900
d_sd2 = virtualAngiogram_Jul2023.d_sd2 #1148   #
d_d2 =   d_sd2 - d_s2
d_p = virtualAngiogram_Jul2023.d_p   #0.154#        0.390625  # Pixel spacing
px =  virtualAngiogram_Jul2023.px   #1024  # Pixels (512)

M = virtualAngiogram_Jul2023.rotation(alpha, beta)
M2 = virtualAngiogram_Jul2023.rotation(alpha2, beta2)

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

# _____________Detector______________#
r2 = np.array([0, 1, 0])  # basic x and y directional vectors to define the detector.
r2 = np.matmul(M, r2)
r3 = np.array([1, 0, 0])
r3 = np.matmul(M, r3)

r2_2 = np.array([0, 1, 0])
r2_2 = np.matmul(M2, r2_2)
r3_2 = np.array([1, 0, 0])
r3_2 = np.matmul(M2, r3_2)

legVx_2D =  virtualAngiogram_Jul2023.x_coord_fit_proj1
legVy_2D =  virtualAngiogram_Jul2023.y_coord_fit_proj1
legVx2_2D = virtualAngiogram_Jul2023.x_coord_fit_proj2
legVy2_2D = virtualAngiogram_Jul2023.y_coord_fit_proj2

legVx =  virtualAngiogram_Jul2023.x_coord_fit_proj1
legVy =  virtualAngiogram_Jul2023.y_coord_fit_proj1
legVx2 = virtualAngiogram_Jul2023.x_coord_fit_proj2
legVy2 = virtualAngiogram_Jul2023.y_coord_fit_proj2

legVx = [value * -d_p for value in legVx]
legVy = [value * d_p for value in legVy]
legVz = d_d1 * np.ones_like(legVx)
centerline1 = []
legVx2 = [value * -d_p for value in legVx2]
legVy2 = [value * d_p for value in legVy2]
legVz2 = d_d2 * np.ones_like(legVx2)
centerline2 = []



for i in range(len(legVx)):
    point = np.array([legVx[i], legVy[i], legVz[i]])
    pointRot = np.matmul(M, point)
    centerline1.append(pointRot)

for i in range(len(legVx2)):
    point2 = np.array([legVx2[i], legVy2[i], legVz2[i]])
    pointRot2 = np.matmul(M2, point2)
    centerline2.append(pointRot2)



# ____________ Epipolar planes and lines __________ #
reconstructed_points = []
total_reconstructed_points = []

r4 = S2 - S
NoIntersectionsPerPoint = np.zeros(len(centerline1))
sign_change_location_total = []
for iPoint in range(len(centerline1)):
    r5 = centerline1[iPoint] - S2
    AA = np.array([(r2_2[0] / np.linalg.norm(r2_2), r3_2[0] / np.linalg.norm(r3_2), -r4[0] / np.linalg.norm(r4),
                    -r5[0] / np.linalg.norm(r5)),
                   (r2_2[1] / np.linalg.norm(r2_2), r3_2[1] / np.linalg.norm(r3_2), -r4[1] / np.linalg.norm(r4),
                    -r5[1] / np.linalg.norm(r5)),
                   (r2_2[2] / np.linalg.norm(r2_2), r3_2[2] / np.linalg.norm(r3_2), -r4[2] / np.linalg.norm(r4),
                    -r5[2] / np.linalg.norm(r5))])
    BB = np.array([S2[0] - D2[0], S2[1] - D2[1], S2[2] - D2[2]])
    n1 = np.cross(r2_2, r3_2)  # normal on detectorplane 1
    n2 = np.cross(r4, r5)  # normal on epipolar plane
    r_epi = np.cross(n1, n2)  # cross product of the  two normals of the planes. This equals to the directional vector of the epipolar line.
    bounds = ([- px / 2 * d_p, - px / 2 * d_p, -np.inf, -np.inf], [px / 2 * d_p, px / 2 * d_p, np.inf, np.inf])
    res = lsq_linear(AA, BB, bounds=bounds, method='bvls', lsmr_tol='auto', verbose=0)
    lam2 = res.x[0]  # dit zijn de lamdas voor een punt op de snijlijn van de 2 vlakken
    lam3 = res.x[1]

    x_epipolar_point = D2[0] + (lam2 / np.linalg.norm(r2_2)) * r2_2[0] + (lam3 / np.linalg.norm(r3_2)) * r3_2[0]
    y_epipolar_point = D2[1] + (lam2 / np.linalg.norm(r2_2)) * r2_2[1] + (lam3 / np.linalg.norm(r3_2)) * r3_2[1]
    z_epipolar_point = D2[2] + (lam2 / np.linalg.norm(r2_2)) * r2_2[2] + (lam3 / np.linalg.norm(r3_2)) * r3_2[2]

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
    #todo remove loop below, gets overwritten every point and useless anyway
    for k in range(len(a)):
        x_value = x_epipolar_point + ( a[k] * r_epi[0] / np.linalg.norm(r_epi) )
        y_value = y_epipolar_point + ( a[k] * r_epi[1] / np.linalg.norm(r_epi) )
        z_value = z_epipolar_point + ( a[k] * r_epi[2] / np.linalg.norm(r_epi) )

        x_epipolarline[k] = x_value
        y_epipolarline[k] = y_value
        z_epipolarline[k] = z_value

        epipolarpoint_local_2D.append(project3Dto2D(P2, SS_2, Q2,[x_value, y_value, z_value],px,d_p)[0])


    dirV_epi_2D = epipolarpoint_local_2D[0]-epipolarpoint_local_2D[1]                               #note first transfer epipolar line in vector representation to 2D
    dirV_g = np.dot(dirV_epi_2D, np.array(((0, -1), (1, 0))))                                   #note (g is perpendicular line on epi) we will define directionalVector perpendicular to dirVector from epiline

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
    #print(len(sign_change_location))


#notitie: vanaf hier heb ik op detector2 in 2D, de snijpunten van de curve met de epipolar line bepaald.
#notitie: deze punten zijn opgeslagen in sign_change_location.

    if len(sign_change_location) == 1:
        subList = []
        epi_curve_intersect3D = project2Dto3D_JT(M2, d_d2, d_p, sign_change_location[0]) #todo gaan er nu nog ff vanuit dat die maar 1x intersect.
        intersection = xRayIntersection(S, centerline1[iPoint], S2,    #intersection = xRayIntersection(S, [curveValsX[iPoint], curveValsY[iPoint], curveValsZ[iPoint]], S2,
                                        epi_curve_intersect3D,
                                        domain=[(-300, 300), (-400, 300),
                                                (-800, 200)])  # domain=[(-30, 30), (-40, 30), (-80, -20)]
        intersection = np.around(intersection.astype(np.double), 4)
        subList.append(intersection)
        #print(iPoint, intersection)
        NoIntersectionsPerPoint[iPoint] = 1
        reconstructed_points.append(subList)
        total_reconstructed_points.append(intersection)

    elif len(sign_change_location) > 1:
        #print("len sign change=",len(sign_change_location))
        NoIntersectionsPerPoint[iPoint] = len(sign_change_location)
        subList = []
        for i in range(len(sign_change_location)):
            epi_curve_intersect3D = project2Dto3D_JT(M2, d_d2, d_p, sign_change_location[i])
            intersection = xRayIntersection(S, centerline1[iPoint], S2,
                                            epi_curve_intersect3D,
                                            domain=[(-300, 300), (-400, 300),
                                                    (-800, 200)])  # domain=[(-30, 30), (-40, 30), (-80, -20)]
            intersection = np.around(intersection.astype(np.double), 4)
            subList.append(intersection)
            total_reconstructed_points.append(intersection)
            #print(iPoint,intersection)
        reconstructed_points.append(subList)


print("All 3D points created now searching for shortest path, it took",time.time()-start,'seconds')

### Dijkstra's algorithm for shortest path, graph

allpossible3Dpoints = reconstructed_points
codedList=NoIntersectionsPerPoint
codedList[-1]= 1
# Change code 0 to 1 for graph
value_zero_index = np.where(codedList == 0.)
value_zero_index = value_zero_index[0]
for index in value_zero_index:
    codedList[index] = 1

graph = Graph()

## Define edges and weights
#for i in range(0, len(allpossible3Dpoints) - 1):
#    for j in range(0, len(allpossible3Dpoints[i])):
#        for k in range(0, len(allpossible3Dpoints[i+1])):
#            graph.add_edge(f"{i}{j}", f"{i+1}{k}", calc_distance_point_point_3D(allpossible3Dpoints[i][j], allpossible3Dpoints[i+1][k]))


# Define edges and weights
for i in range(0, len(codedList) - 1):
    #print('i is', i)
    a=int(codedList[i])
    #print('a is', a)
    for j in range(0, a):
        #print('j is', j)
        b=int(codedList[i+1])
        #print('b is', b)
        for k in range(0, b):
            #print("allpossible3Dpoints[i][j] =",allpossible3Dpoints[i][j],"and allpossible3Dpoints[i]", allpossible3Dpoints[i])
            graph.add_edge(str(i) + str(j), str(i+1) + str(k), calc_distance_point_point_3D(allpossible3Dpoints[i][j], allpossible3Dpoints[i+1][k]))
            #print('distance', calc_distance_point_point_3D(allpossible3Dpoints[i][j], allpossible3Dpoints[i+1][k]))



#print(graph.edges)
#print(graph.weights)

# For graph, define last node
lastNodeNumber = '0'

lastNode = str(len(codedList)-2) + lastNodeNumber
shortestPath = dijsktra(graph, '00', lastNode)
#print('The shortest path is', shortestPath)
#print(len(shortestPath))
optionIndex = [] # save last character as string as 'correct' coordinate option
for i in range(0, len(shortestPath)):
    index = int(shortestPath[i][-1])
    optionIndex.append(index)
#print('option index list is', optionIndex)

finalList3D = []
for i in range(0, len(shortestPath)):
    index_path_coord = optionIndex[i]
    finalList3D.append(allpossible3Dpoints[i][index_path_coord])

print("Centerline reconstruction complete, it took",time.time()-start,'seconds')

#note what we are doing here is taking the 3D reconstructed line, doing a projection back on both detectors so we can assess the result and
#note click the diameters on both projections per point
backProjectionView1 = []
backProjectionView2 = []
from virtualAngiogram_Jul2023 import projection_1, projection_2
"""
for i in range(0,len(allpossible3Dpoints)):
    parameters = projection_1(allpossible3Dpoints[i][0], allpossible3Dpoints[i][1], allpossible3Dpoints[i][2])
    x_p = D[0] + (parameters[0] / np.linalg.norm(r2)) * r2[0] + (parameters[1] / np.linalg.norm(r3)) * r3[0]
    y_p = D[1] + (parameters[0] / np.linalg.norm(r2)) * r2[1] + (parameters[1] / np.linalg.norm(r3)) * r3[1]
    z_p = D[2] + (parameters[0] / np.linalg.norm(r2)) * r2[2] + (parameters[1] / np.linalg.norm(r3)) * r3[2]
    v_p = np.array([x_p, y_p, z_p])
    twoDpoint = project3Dto2D(P1, SS_1, Q1, v_p)
    backProjectionView1.append(twoDpoint[0])

    parameters2 = projection_2(allpossible3Dpoints[i][0], allpossible3Dpoints[i][1], allpossible3Dpoints[i][2])
    x_p2 = D2[0] + (parameters2[0] / np.linalg.norm(r2_2)) * r2_2[0] + (parameters2[1] / np.linalg.norm(r3_2)) * r3_2[0]
    y_p2 = D2[1] + (parameters2[0] / np.linalg.norm(r2_2)) * r2_2[1] + (parameters2[1] / np.linalg.norm(r3_2)) * r3_2[1]
    z_p2 = D2[2] + (parameters2[0] / np.linalg.norm(r2_2)) * r2_2[2] + (parameters2[1] / np.linalg.norm(r3_2)) * r3_2[2]
    v_p2 = np.array([x_p2, y_p2, z_p2])
    twoDpoint_2 = project3Dto2D(P2, SS_2, Q2, v_p2)
    backProjectionView2.append(twoDpoint_2[0])
"""
####_______________ PLOTTING ________________######
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.sca(ax)
# Set the view to look straight up the Z-axis
ax.view_init(elev=-90, azim=95)
# plt.plot(xvals, yvals, zvals, color='b',linestyle='dotted',alpha=0.2)                    # principle ray
ax.scatter(S[0], S[1], S[2], color='b')  # source

#for sublist in allpossible3Dpoints:
#    for a in sublist:
#        ax.scatter(*a,c='fuchsia',alpha=0.4)

for kk in finalList3D:
    ax.scatter(*kk,c='cyan',alpha=0.8)

plt.plot(virtualAngiogram_Jul2023.virtualVesselx,virtualAngiogram_Jul2023.virtualVessely,virtualAngiogram_Jul2023.virtualVesselz,c='navy')

# plt.plot(xvals2, yvals2, zvals2, color='k',linestyle='dotted',alpha=0.2)
ax.scatter(S2[0], S2[1], S2[2], color='k')


#ax.scatter(*project2Dto3D_JT(M2, d_d2, d_p, sign_change_location[0]),c='g',s=50)
#ax.scatter(*project2Dto3D_JT(M2, d_d2, d_p, sign_change_location[1]),c='g')

for x in centerline1:
    ax.scatter(*x, color='b')

for x in centerline2:
    ax.scatter(*x, color='k')

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
plt.plot(x_vals_detector[0], y_vals_detector[0], z_vals_detector[0], c='b')
plt.plot(x_vals_detector[1], y_vals_detector[1], z_vals_detector[1], c='k')
#plt.plot(x_epipolarline,y_epipolarline,z_epipolarline,c='r')

curveValsX = []
curveValsX2 = []
curveValsY = []
curveValsY2 = []
curveValsZ = []
curveValsZ2 = []


for i in range(len(centerline1)):
    plt.plot([S[0],centerline1[i][0]], [ S[1],centerline1[i][1]],[ S[2],centerline1[i][2]], color='b',alpha=0.2)
    plt.plot([S2[0],  centerline2[i][0]], [ S2[1], centerline2[i][1]], [ S2[2], centerline2[i][2]], color='k',alpha=0.2)





##n_eval_detector = 2
##lamdaT = np.linspace(-4000 * d_p, 4000 * d_p, n_eval_detector)
##xT = np.empty((len(lamdaT), len(lamdaT)))
##yT = np.empty((len(lamdaT), len(lamdaT)))
##zT = np.empty((len(lamdaT), len(lamdaT)))
##for i in range(len(lamdaT)):
##    for j in range(len(lamdaT)):
##        xT[i][j] = S2[0] + (lamdaT[i] / np.linalg.norm(r4)) * r4[0] + (lamdaT[j] / np.linalg.norm(r5)) * r5[0]
##        yT[i][j] = S2[1] + (lamdaT[i] / np.linalg.norm(r4)) * r4[1] + (lamdaT[j] / np.linalg.norm(r5)) * r5[1]
##        zT[i][j] = S2[2] + (lamdaT[i] / np.linalg.norm(r4)) * r4[2] + (lamdaT[j] / np.linalg.norm(r5)) * r5[2]
##
##ax.plot_wireframe(xT, yT, zT, color='green', alpha=0.95, linewidth=0.75)  # epipolar plane
#ax.scatter(centerline1[0][0],centerline1[0][1],centerline1[0][2],c='r')
#plt.plot([S2[0], epi_curve_intersect3D[0]], [S2[1], epi_curve_intersect3D[1]], [S2[2], epi_curve_intersect3D[2]], color='k', alpha=0.8)

from virtualAngiogram_Jul2023 import topmid_3d, topmid_3d_2
#ax.scatter(topmid_3d[0], topmid_3d[1], topmid_3d[2], c='yellow', alpha=0.4)
#ax.scatter(topmid_3d_2[0], topmid_3d_2[1], topmid_3d_2[2], c='yellow', alpha=0.4)

#xLabel = ax.set_xlabel('X-axis', linespacing=3.2)
#yLabel = ax.set_ylabel('Y-axis', linespacing=3.1)
#zLabel = ax.set_zlabel('Z-Axis', linespacing=3.4)
ax.set_aspect('equal')  # , adjustable='box')
# Hide grid lines
ax.grid(False)

# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_axis_off()
print("3D plot done")


fig2 = plt.figure("1st view")
plt.xlim([-(px/2), (px/2)])
plt.ylim([-(px/2), (px/2)])
plt.plot(virtualAngiogram_Jul2023.x_coord_fit_proj1, virtualAngiogram_Jul2023.y_coord_fit_proj1, color='crimson', label='Reparameterized Legendre Curve',alpha=0.5)


# Set plot labels and legend 
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()


fig3 = plt.figure("2d view")
plt.plot(legVx2_2D, legVy2_2D, color='k', label='Reparameterized Legendre Curve')


plt.xlim([-(px/2), (px/2)])
plt.ylim([-(px/2), (px/2)])

#for i in epipolarpoint_local_2D:
#    plt.scatter(*i,c='r',alpha=0.5)
plt.scatter(*sign_change_location[0],c='g')
plt.scatter(*sign_change_location[1],c='g')
plt.plot([epipolarpoint_local_2D[0][0],epipolarpoint_local_2D[-1][0]],[epipolarpoint_local_2D[0][1],epipolarpoint_local_2D[-1][1]],c='r')

ax.set_axis_off()
plt.show()

print('Execution time was ', time.time()-start, 'seconds.')
