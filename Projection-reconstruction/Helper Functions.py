import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.optimize import curve_fit, minimize, root
import sympy as sym
from sympy.solvers.solveset import linsolve

def readjson(nameJson):
    """
    Read Json file and open centerline coordinates from this.
    """
    with open(nameJson, 'r') as json_data:
        data = json.load(json_data)
    x_points = []
    y_points = []
    z_points = []
    for i in data["points"]:
        x_points.append(i[0])
        y_points.append(i[1])
        z_points.append(i[2])

    return x_points, y_points, z_points


def map_range(value, from_range, to_range):
    """
    Map a value from one range to another.
    """
    # Get the length of the ranges
    from_range_length = from_range[1] - from_range[0]
    to_range_length = to_range[1] - to_range[0]
    # Calculate the ratio between the ranges
    ratio = to_range_length / from_range_length
    # Calculate the equivalent point on the new range
    new_value = (value - from_range[0]) * ratio + to_range[0]
    return new_value

def project3Dto2D(P,S,Q,point):
    """
    :param P: corner P
    :param S: corner S
    :param Q: corner Q
    :param point: Point to be mapped from 3D coordinates to 2D local
    :return:  ang_mapped_coord:mapped to angiogram so -256,256 pixels ___ mapped_coord: in mm with bottomwleft 0,0
    """
    PS = S-P
    PQ = Q-P
    PA = point-P
    D1 = np.linalg.norm(np.cross(PA, PS)) / np.linalg.norm(PS) # distance between point and PS,
    # or xaxis = the yaxis in 2D!
    #D2 = np.linalg.norm(np.cross(PA, PQ)) / np.linalg.norm(PQ) # distance between point and PQ,
    # or yaxis = the xaxis in 2D!
    D2 = np.linalg.norm(np.cross(PA, PQ)) / np.linalg.norm(PQ)
    D22= 158-D2
    mapped_coord = np.array((D22, D1)) #in [mm] ; this function assumse bottom left is (0,0)

    scaled_x = map_range(mapped_coord[0],[-78.696, 78], [-256,256]) *-1 #scaled_x = map_range(mapped_coord[0],[0, 115], [-512,512]) *-1  #(flipping of X coordinates.) ### scaled_x = map_range(mapped_coord[0],[0, 158], [-512,512]) *-1
    scaled_y = map_range(mapped_coord[1], [-78, 78], [-256, 256]) # #scaled_y = map_range(mapped_coord[1], [0, 115], [-512, 512]) #scaled_y = map_range(mapped_coord[1], [0, 200], [-512, 512])
    ang_mapped_coord=np.array((scaled_x, scaled_y)) #scaled to -256,256 and flipped x axis

    return ang_mapped_coord, mapped_coord

def project2Dto3D_JT(rot_mat,d_d,d_p,point2D):
    """
    point2D should be in [mm]
    """
    point2D_copy = point2D.copy()
    # transfer the 2D reparametrised coordiantes to 3D by adding z_value at the detector plane , then rotate with matmul(M)
    #point2D_copy[0] = point2D_copy[0]
    #point2D_copy[1] = point2D_copy[1]
    point = np.array([point2D_copy[0], point2D_copy[1], d_d])
    pointRot = np.matmul(rot_mat, point)

    return pointRot

def xRayIntersection(p1,p2,p3,p4,domain=None):
    # Calculate the directional vectors of the lines
    v1 = p2 - p1
    v2 = p4 - p3


    cross_product = np.cross(v1, v2)

    matrix = np.column_stack((v1, -v2, cross_product))
    rhs = p3 - p1


    # Solve the system of equations
    solution = np.linalg.solve(matrix, rhs)
    intersection = p1 + solution[0] * v1

    if domain is not None:
        for i in range(3):
            if intersection[i] < domain[i][0] or intersection[i] > domain[i][1]:
                print("Intersection is outside the domain.")
                print(f"Coordinate {i}: {intersection[i]}, Domain Range: {domain[i][0]} - {domain[i][1]}")
                intersection = None

    #add computation of m1m2 and middlepoint between them and return that as intersection

    """
        Find shortest distance between two skew lines
        Input: position and direction vectors of the two lines
        Output: The two coordinates of the points closest to each other
        """

    sv1 = p1
    rv1 = v1
    sv2 = p3
    rv2  = v2

    t = sym.Symbol('t')
    w = sym.Symbol('w')

    PPx = np.add(sv1[0], (np.multiply(t, rv1[0])))
    #print('PPx is', PPx)
    PPy = np.add(sv1[1], (np.multiply(t, rv1[1])))
    PPz = np.add(sv1[2], (np.multiply(t, rv1[2])))

    PP = np.array([PPx, PPy, PPz])

    QQx = np.add(sv2[0], (np.multiply(w, rv2[0])))
    QQy = np.add(sv2[1], (np.multiply(w, rv2[1])))
    QQz = np.add(sv2[2], (np.multiply(w, rv2[2])))

    QQ = np.array([QQx, QQy, QQz])

    PPQQ = np.subtract(QQ, PP)

    # Condition is that PPQQ.rv1 = 0 and PPQQ.rv2 = 0, then perpendicular to both lines

    PPQQ_u1 = np.dot(PPQQ, rv1)  # this is equation 1
    # print('PPQQ_u1 is', PPQQ_u1)
    PPQQ_u2 = np.dot(PPQQ, rv2)  # this is equation 2
    # print('PPQQ_u2 is', PPQQ_u2)
    t_and_w_1 = sym.solve(PPQQ_u1)
    # print('t_and_w_1 is', t_and_w_1)
    t_and_w_2 = sym.solve(PPQQ_u2)
    # print('t_and_w_2 is', t_and_w_2)

    lin_solve = (linsolve([PPQQ_u1, PPQQ_u2], (t, w)))
    (t, w) = next(iter(lin_solve))

    px = np.add(sv1[0], (np.multiply(t, rv1[0])))
    py = np.add(sv1[1], (np.multiply(t, rv1[1])))
    pz = np.add(sv1[2], (np.multiply(t, rv1[2])))

    PP = np.array([px, py, pz])

    qx = np.add(sv2[0], (np.multiply(w, rv2[0])))
    qy = np.add(sv2[1], (np.multiply(w, rv2[1])))
    qz = np.add(sv2[2], (np.multiply(w, rv2[2])))

    QQ = np.array([qx, qy, qz])

    #print(PP,QQ)
    def middle_between_two_points(point1, point2):
        """Find the middle coordinate between two points"""
        middle_x = (point1[0] + point2[0]) / 2
        middle_y = (point1[1] + point2[1]) / 2
        middle_z = (point1[2] + point2[2]) / 2

        middle = np.array([middle_x, middle_y, middle_z])

        return middle

    intersection = middle_between_two_points(PP,QQ)
    #print("interesection = ",intersection)

    return intersection

def fit_legendre_polynomials(s, x, y, degree):
    """
    Fits Legendre polynomials to the x and y data using the given degree and returns the polynomial coefficients.
    """
    x_coeffs, _ = curve_fit(legendre_polynomial, s, x, p0=[1] * degree)
    y_coeffs, _ = curve_fit(legendre_polynomial, s, y, p0=[1] * degree)
    return x_coeffs, y_coeffs

def legendre_polynomial(s, *coeffs):
    """
    Computes the value of the Legendre polynomial with coefficients 'coeffs' for the given value of s.
    """
    result = 0
    for i, c in enumerate(coeffs):
        result += c * np.polynomial.legendre.legval(s, [0] * i + [1])
    return result

def reparameterize_curve(s, x_coeffs, y_coeffs):
    """
    Reparameterizes the curve using the fitted polynomials and returns the s-parametrized x and y data.
    """
    x = legendre_polynomial(s, *x_coeffs)
    y = legendre_polynomial(s, *y_coeffs)
    return x, y

def intersection_two_lines_2D(p1, v1, p2, v2):
    """
    :param p1: support Vec 1, position on curve
    :param v1: dir Vec 1
    :param p2: support Vec 2, epipolar point
    :param v2: dir Vec 2, epipolar direction
    :return: intersecstional point
    """
    # Convert the vectors to NumPy arrays for easier computation
    p1 = np.array(p1)
    v1 = np.array(v1)
    p2 = np.array(p2)
    v2 = np.array(v2)

    # Create a matrix A with the direction vectors as rows
    A = np.array([(v1[0],-v2[0]),(v1[1],-v2[1])])

    # Create a vector b with the difference of the two starting points
    b = p2 - p1

    # Solve the linear system Ax = b
    x = np.linalg.solve(A, b)

    # Calculate the intersection point by substituting the solution into one of the line equations
    intersection_point = p1 + x[0] * v1

    return intersection_point

def plot_legendre_curve(x, y, x_coeffs, y_coeffs, s_reparam):
    """
    Plot the original data points and the reparameterized Legendre curve.
    """
    # Plot original data points
    plt.scatter(x, y, color='red', label='Original Data Points')

    # Reparameterize the curve using the fitted coefficients
    reparam_x, reparam_y = reparameterize_curve(s_reparam, x_coeffs, y_coeffs)

    # Plot the reparameterized Legendre curve
    plt.plot(reparam_x, reparam_y, color='blue', label='Reparameterized Legendre Curve')

    # Set plot labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    # Show the plot
    plt.show()

def calculate_lambda_JT(epi_s,epi_dir,point1,point2):
    """
    :param epi_s: epipolar support vector
    :param epi_dir: epipolar directional vector
    :param point1: detector corner point
    :param point2: detector corner point
    :return: lamda for epipolar line on the intersection between epi line and detector edge spanned between p1 and p2
    """
    lineVec = (point1 - point2)

    A = np.array([[lineVec[0], -epi_dir[0]/np.linalg.norm(epi_dir)],
                  [lineVec[1], -epi_dir[1]/np.linalg.norm(epi_dir)]])
    b = np.array([(epi_s[0]-point1[0]),
                  (epi_s[1]-point1[1])])
    sol = np.linalg.solve(A,b)

    lamda_epi = sol[1]
    return lamda_epi



def calc_distance_point_point_3D(p1, p2):
    x_part = np.subtract(p2[0], p1[0])
    x_part = x_part ** 2
    y_part = np.subtract(p2[1], p1[1])
    y_part = y_part ** 2
    z_part = np.subtract(p2[2], p1[2])
    z_part = z_part ** 2
    x_and_y = np.add(x_part, y_part)
    total = np.add(x_and_y, z_part)
    d = np.sqrt(total)

    return d

from collections import defaultdict

# Dijkstra's algorithm
# From Ben Alex Keen (benalexkeen.com)
# IMPLEMENTING DIJKSTRA'S SHORTEST PATH ALGORITHM WITH PYTHON

class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}

    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are not bi-directional
        self.edges[from_node].append(to_node)
        self.weights[(from_node, to_node)] = weight


def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
        #print('current node is', current_node)

    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path


import math

def calculate_distance(point1, point2):
    print("point1=",point1,"point2=",point2)
    distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
    return distance


def select_shortest_path(sublists):
    final_points = []

    for i, sublist in enumerate(sublists):
        if len(sublist) == 1:
            # If the sublist has only one point, add it directly to the final list
            final_points.append(sublist[0])
        else:
            current_distance = calculate_distance(sublist[0], sublist[1])
            selected_point = sublist[0]  # Assume the first point as the initially selected point

            if current_distance <= 0:
                # If the distance is non-positive, skip this sublist
                continue

            if i < len(sublists) - 1:
                # If there is a next sublist
                next_distance1 = calculate_distance(sublist[0], sublists[i + 1][0])
                next_distance2 = calculate_distance(sublist[0], sublists[i + 1][1])

                if next_distance1 < next_distance2:
                    selected_point = sublist[
                        1]  # Choose the second point if it has a shorter distance to the next sublist
                else:
                    selected_point = sublist[0]  # Choose the first point otherwise

            final_points.append(selected_point)

    return final_points
