import numpy as np
from scipy.spatial.transform import Rotation as R
from collections.abc import Iterable
import math
#from transformations import *

import pdb
def translation_matrix(vec):

    m = np.identity(4)

    m[:3, 3] = vec

    return m
class Volume:
    def __init__(self, dicom_volume):
        pixel_measures_sequence = dicom_volume[0x5200, 0x9229][0][0x0028, 0x9110][0]
        pixel_spacing = pixel_measures_sequence.PixelSpacing # size of a single pixel in the iso center (within patient) in mm
        slice_thickness = pixel_measures_sequence.SliceThickness
        
        self.volume = dicom_volume
        self.nr_of_voxels = np.array([dicom_volume.Columns, dicom_volume.Rows, dicom_volume.NumberOfFrames])
        self.voxel_size = np.array([pixel_spacing[1], pixel_spacing[0], slice_thickness])
        self.pixel_array = dicom_volume.pixel_array
        
    def extent(self):
        return (self.nr_of_voxels-1) * self.voxel_size

    def set_table_pos(self, dicom_rot):
        # from PhilipsIGT package
        xvc_arm_latitudinal_position = dicom_rot[0x2003, 0x2083].value
        xvc_arm_longitudinal_position = dicom_rot[0x2003, 0x2082].value
        x_vision_table_lateral_pos = dicom_rot[0x2003, 0x2031].value
        x_vision_table_longitude_pos = dicom_rot[0x2003, 0x2030].value
        x_vision_table_height = dicom_rot[0x2003, 0x202F].value
        
        self.table_pos = np.array([
            x_vision_table_lateral_pos - xvc_arm_latitudinal_position,
            x_vision_table_longitude_pos - xvc_arm_longitudinal_position,
            x_vision_table_height])

class XRayProjection:
    # based on PhilipsIGT package
    def __init__(self, dicom, x_ray_type='rotation', dicom_volume=None):
        self.dicom = dicom
        self.pixel_array = dicom.pixel_array
        self.x_ray_type = x_ray_type

        self.number_of_frames = 1
        if hasattr(dicom, 'NumberOfFrames'):
            self.number_of_frames = int(dicom.NumberOfFrames)

        self.nr_of_pixels = np.array([dicom.Columns, dicom.Rows, self.number_of_frames])
        self.max_pixel_value = 2 ** (dicom.HighBit+1) - 1

        if (0x2003, 0x20A2) in dicom:
            self.sequences = dicom[0x2003, 0x20A2]
        elif (0x2003, 0x102E) in dicom:
            self.sequences = dicom[0x2003, 0x102E]
        else:
            self.sequences = np.arange(0, self.number_of_frames)

        self.distance_source_to_detector = float(dicom.DistanceSourceToDetector)
        self.distance_source_to_patient = float(dicom.DistanceSourceToPatient)
        self.distance_detector_to_patient = self.distance_source_to_detector - self.distance_source_to_patient

        # difference between pixel centers on detector in mm
        if "ImagerPixelSpacing" in dicom:
            #DICOM is [1]=columns=x, [0]=rows=y
            # size of a single pixel on the detector in mm
            self.imager_pixel_spacing = np.array([dicom.ImagerPixelSpacing[1], dicom.ImagerPixelSpacing[0]])
        else:
            #XVisionPixelWidth, XVisionPixelHeight
            self.imager_pixel_spacing = np.array([dicom[0x2003, 0x201E].value, dicom[0x2003, 0x201F].value])
        
        self.geo = []
        if self.x_ray_type == 'roadmap':
            for frame_nb, element in enumerate(self.sequences):
                self.geo.append(self.RoadmapRunGeometry(element, frame_nb, dicom))
        elif self.x_ray_type == 'rotation':
            self.calset = dicom[0x2003, 0x204C]
            self.version = str(self.calset[60]) + '.' + str(self.calset[61])
            self.geo_first_image = int.from_bytes(self.calset[62:66], byteorder='little', signed=True)
            self.geo_nr_of_images = int.from_bytes(self.calset[66:70], byteorder='little', signed=True)

            while len(self.geo) < self.geo_nr_of_images:
                self.geo.append(self.RotationalRunGeometry(self.calset, len(self.geo), self.sequences[len(self.geo)]))

        # self.camera_matrices = [self.compute_camera_matrix(idx, znear, zfar) for idx in range(len(self.pixel_array))]

    def normalize_images(self):
        self.pixel_array = (self.pixel_array - np.min(self.pixel_array))/(np.max(self.pixel_array) - np.min(self.pixel_array))

    def angle_to_360(self, angle):
        if angle < 0:
            angle = np.float(360 + angle)
        else:
            angle = np.float(angle)
        return angle

    def angles(self, idx):
        if self.x_ray_type == 'roadmap':
            theta = np.float(self.geo[idx].prop_angle)
            phi = np.float(self.geo[idx].roll_angle)
            larm = np.float(self.geo[idx].larm_angle)
        elif self.x_ray_type == 'rotation':
            theta = np.float(self.geo[idx].cardan_angle_Y) #self.primary_angle
            phi = np.float(self.geo[idx].cardan_angle_X) #self.secondary_angle#
            larm = np.float(self.geo[idx].cardan_angle_Z)
        return [theta, phi, larm]
    
    def table_offset(self, idx, initial_tab_pos=[0,0,0]):
        print(self.geo[idx].table_pos - initial_tab_pos)
        return self.geo[idx].table_pos - initial_tab_pos

    def detector_size(self):
        # nr of pixels of image for width and height
        dimensions = np.array([self.nr_of_pixels[0], self.nr_of_pixels[1], 1])

        # imager pixel spacing = PHYSICAL distance between center of each image pixel in mm
        spacing = np.append(self.imager_pixel_spacing, 1.0)

        size = ((dimensions-1) * spacing) # -1 because #center of pixels = width - 1
        # size = dimensions * spacing

        return size
    
    def detector_center(self):
        det_size = self.detector_size()
        return 0.5 * det_size

    def detector_plane(self, idx, rotation=True):
        image_geo = self.geo[idx]

        O = np.array([0, 0, -image_geo.iso_center[2], 1])
        x_center, y_center, _ = self.detector_center()

        xy_00 = [-x_center, -y_center, -image_geo.iso_center[2], 1]
        xy_01 = [-x_center, y_center, -image_geo.iso_center[2], 1]
        xy_10 = [x_center, -y_center, -image_geo.iso_center[2], 1]
        xy_11 = [x_center, y_center, -image_geo.iso_center[2], 1]
        center = O

        rotation_matrix = image_geo.rotation
        center_tr = translation_matrix(center[:3])
        center_rotated = rotation_matrix.dot(center_tr)

        coordinates = [xy_00, xy_01, xy_11, xy_10]

        if rotation:
            rotated = list(coordinates)
            for i, coordinate in enumerate(coordinates):
                point_tr = translation_matrix(np.array([coordinate[0], coordinate[1], 0]))
                # cam = m1.dot(m2.dot(m3))
                pt = center_rotated.dot(point_tr)
                rotated[i] = pt[:3, -1]
            coordinates = rotated

        return np.array(coordinates).T

    def estimate_volume_size(self, idx):
        dDetector = np.array([self.imager_pixel_spacing[0], self.imager_pixel_spacing[1], 1])
        nDetector = self.nr_of_pixels

        self.dVoxel = dDetector * self.distance_source_to_patient / self.distance_source_to_detector
        self.nVoxel = np.ceil(np.array([nDetector[0]+np.abs(0)/dDetector[0], nDetector[0]+np.abs(0/dDetector[0]), nDetector[1]]))
        self.sVoxel = self.nVoxel / self.dVoxel

        return [self.dVoxel, self.nVoxel, self.sVoxel]

    def detector_matrix(self, idx, initial_tab_pos=np.array([0,0,0])):
        image_geo = self.geo[idx]
        iso_center = image_geo.iso_center
        print(idx)
        # translate to detector
        # m1 = translation_matrix(np.array([0,0,-self.distance_detector_to_patient]))
        # m1 = translation_matrix(np.array([iso_center[0], iso_center[1], iso_center[2]]))
        m1 = translation_matrix(np.array([iso_center[0], iso_center[1], iso_center[2]]))
        #print(m1)
        # rotate according to c-arm
        m2 = image_geo.rotation
        # table translation
        table_offset = self.table_offset(idx, initial_tab_pos)
        m3 = translation_matrix(table_offset)
        #print(m2)
        world_to_cam = m1.dot(m2.dot(m3))
        cam_to_world = np.linalg.inv(world_to_cam)

        return world_to_cam, cam_to_world
    
    def source_matrix(self, idx, initial_tab_pos=np.array([0,0,0]), calibrated=True):
        # source position and rotation w.r.t. WORLD coordinates
        image_geo = self.geo[idx]
        iso_center = image_geo.iso_center
        #print(idx)
        # translate to source
        m1 = translation_matrix(np.array([0, 0, self.distance_source_to_patient]))
        #print(m1)
        if calibrated:
            m1 = translation_matrix(np.array([iso_center[0], iso_center[1], (self.distance_source_to_detector + iso_center[2])]))

        # rotate according to c-arm
        m2 = image_geo.rotation
        #print(m2)
        # table translation
        table_offset = self.table_offset(idx, initial_tab_pos)
        # table_offset = np.array([0,0,0])
        m3 = translation_matrix(table_offset)
        world_to_cam = m1.dot(m2.dot(m3)) #this works before (without table?)
        # world_to_cam =  m1.dot(m3.dot(m2))
        cam_to_world = np.linalg.inv(world_to_cam)

        return world_to_cam, cam_to_world
    
    class RoadmapRunGeometry:
        def __init__(self, element, frame_nb, dicom):
            # obtain dicom attributes
            if isinstance(element, Iterable) and (0x2003, 0x20A3) in element:
                self.image_number = element[0x2003, 0x20A3].value #XVLiveImageNumber
                self.prop_angle = np.array(element[0x2003, 0x20A4].value) #XVPropAngle RAO|LAO
                self.roll_angle = np.array(element[0x2003, 0x20E1].value) #XVRollAngle CRAU|CRA
                self.larm_angle = np.array(element[0x2003, 0x20A5].value) #XVLarmAngle

                #NO CALIBRATION
                self.iso_center = np.array(element[0x2003, 0x20A7].value) #XVIsoCenter 
                self.distance_source_to_detector = np.array(element[0x2003, 0x20A6].value) #XVFocusPosition distance of source focal spot to detector in mm
                self.distance_iso_to_detector = np.array(element[0x2003, 0x20A7].value)
                self.distance_source_to_iso = self.distance_source_to_detector - self.distance_iso_to_detector
            # only table is stored per frame.
            else:
                # get the info from the global DICOM, not per frame.
                self.image_number = frame_nb
                self.prop_angle = float(dicom.PositionerPrimaryAngle)
                self.roll_angle = float(dicom.PositionerSecondaryAngle)
                self.larm_angle = 0.

                self.distance_source_to_detector = float(dicom.DistanceSourceToDetector)
                self.distance_source_to_patient = float(dicom.DistanceSourceToPatient)
                self.distance_detector_to_patient = self.distance_source_to_detector - self.distance_source_to_patient
                self.iso_center = np.array([0,0, -(self.distance_source_to_detector - self.distance_source_to_patient)]) # no calibration

            # table pos per frame
            self.table_pos = TablePosFromDicom(dicom, element)

            self.rotation = np.identity(4)
            self.rotation[:3, :3] = (
                R.from_rotvec(np.deg2rad(self.larm_angle) * np.array([0, 0, 1])) *
                R.from_rotvec(np.deg2rad(self.prop_angle) * np.array([0, 1, 0])) *
                R.from_rotvec(np.deg2rad(self.roll_angle) * np.array([1, 0, 0]))).inv().as_matrix()
            # self.rotation[:3, :3] = (
            #     R.from_rotvec(np.deg2rad(self.roll_angle) * np.array([0, 1, 0])) *
            #     R.from_rotvec(np.deg2rad(self.prop_angle) * np.array([1, 0, 0])) *
            #     R.from_rotvec(np.deg2rad(self.larm_angle) * np.array([0, 0, 1]))).as_matrix()

            self.xv_image_rotated, self.xv_flip_vertical, self.xv_flip_horizontal = False, False, False
            self.xv_detector_angle = 0.
            if isinstance(element, Iterable) and (0x2003, 0x20AE) in element:
                self.xv_image_rotated = element[0x2003, 0x20AE].value.lower() == 'true'
                self.xv_flip_vertical = element[0x2003, 0x20AA].value.lower() == 'true'
                self.xv_flip_horizontal = element[0x2003, 0x20A9].value.lower() == 'true'
                self.xv_detector_angle = element[0x2003, 0x20A8].value
        
    class RotationalRunGeometry:
        def __init__(self, raw, n, element):
            idx = 70 + 60 + 2 + n * (60 + 2 + 52)  # 70 bytes for calset, 60 bytes geo header, 2 bytes version, 52 per geo entry
            #imageNumber, rotation, angulation, larm, xisocenter, yisocenter, zisocenter:
            self.element = element

            # these are the calibrated angles and isocenters!
            self.image_number = int.from_bytes(raw[idx:idx+4], byteorder='little', signed=True)
            self.cardan_angle_Y = np.frombuffer(raw[idx+4:idx+12], np.float64)
            self.cardan_angle_X = np.frombuffer(raw[idx+12:idx+20], np.float64)
            self.cardan_angle_Z = np.frombuffer(raw[idx+20:idx+28], np.float64)

            self.rotation = np.identity(4)
            self.rotation[:3, :3] = (
                R.from_rotvec(np.deg2rad(self.cardan_angle_Y) * np.array([0, 1, 0])) *
                R.from_rotvec(np.deg2rad(self.cardan_angle_X) * np.array([1, 0, 0])) *
                R.from_rotvec(np.deg2rad(self.cardan_angle_Z) * np.array([0, 0, 1]))).as_matrix()
            
            self.iso_center = np.array([
                np.frombuffer(raw[idx+28:idx+36], np.float64),
                np.frombuffer(raw[idx+36:idx+44], np.float64),
                np.frombuffer(raw[idx+44:idx+52], np.float64)]).flatten()
            
            self.distance_source_to_detector = np.array(element[0x2003, 0x20A6].value) #XVFocusPosition distance of source focal spot to detector in mm
            self.distance_iso_to_detector = np.array(element[0x2003, 0x20A7].value)
            self.distance_source_to_iso = self.distance_source_to_detector - self.distance_iso_to_detector

            self.table_pos = np.array([0,0,0])

            self.xv_flip_horizontal = self.element[0x2003, 0x20A9].value.lower() == 'true'

def TablePosFromDicom(dicom, element):
    # phantom data
    if isinstance(element, Iterable) and (0x2003, 0x2083) in element:
        XVCArmLatitudinalPosition = element[0x2003, 0x2083].value
        XVCArmLongitudinalPosition = element[0x2003, 0x2082].value
        XVisionTableLateralPos = dicom[0x2003, 0x2031].value
        XVisionTableLongitudePos = dicom[0x2003, 0x2030].value
        XVisionTableHeight = dicom[0x2003, 0x202F].value
        return np.array([
            XVisionTableLateralPos - XVCArmLatitudinalPosition,
            XVisionTableLongitudePos - XVCArmLongitudinalPosition,
            XVisionTableHeight])
    # PIPPSAI data
    elif isinstance(element, Iterable) and (0x2003, 0x1265) in dicom:
        AlluraBeamTransversal = dicom[0x2003, 0x1265].value
        AlluraBeamLongitudinal = dicom[0x2003, 0x1207].value
        TableTopLateralPosition = element.TableTopLateralPosition  #dicom[0x300A, 0x012A].value
        TableTopLongitudinalPosition = element.TableTopLongitudinalPosition  #dicom[0x300A, 0x0129].value
        TableTopVerticalPosition = element.TableTopVerticalPosition  #dicom[0x300A, 0x0128].value
        return np.array([ #ORIGINAL
            TableTopLateralPosition - AlluraBeamTransversal,
            TableTopLongitudinalPosition - AlluraBeamLongitudinal,
            TableTopVerticalPosition])
    else:
        return np.array([0.0, 0.0, 0.0])  #no table pos found in the DICOM meta data
