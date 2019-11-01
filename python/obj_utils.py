import os

import numpy as np

class ObjectLabel:
    """Object Label Class
    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                      'Misc' or 'DontCare'

    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                      truncated refers to the object leaving image boundaries

    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                      0 = fully visible, 1 = partly occluded
                      2 = largely occluded, 3 = unknown

    1    alpha        Observation angle of object, ranging [-pi..pi]

    4    bbox         2D bounding box of object in the image (0-based index):
                      contains left, top, right, bottom pixel coordinates

    3    dimensions   3D object dimensions: height, width, length (in meters)

    3    location     3D object location x,y,z in camera coordinates (in meters)

    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]

    1    score        Only for results: Float, indicating confidence in
                      detection, needed for p/r curves, higher is better.
    """

    def __init__(self):
        self.type = ""  # Type of object
        self.truncation = 0.
        self.occlusion = 0.
        self.alpha = 0.
        self.x1 = 0.
        self.y1 = 0.
        self.x2 = 0.
        self.y2 = 0.
        self.h = 0.
        self.w = 0.
        self.l = 0.
        self.t = (0., 0., 0.)
        self.ry = 0.
        self.score = 0.

    def __eq__(self, other):
        """Compares the given object to the current ObjectLabel instance.

        :param other: object to compare to this instance against
        :return: True, if other and current instance is the same
        """
        if not isinstance(other, ObjectLabel):
            return False

        if self.__dict__ != other.__dict__:
            return False
        else:
            return True


def project_to_image(point_cloud, p, dist=None):
    """ Projects a 3D point cloud to 2D points for plotting

    :param point_cloud: 3D point cloud (3, N) in camera frame
    :param p: Camera matrix (3, 3)
    :param dist: Camera distortion 5-element array

    :return: pts_2d: the image coordinates of the 3D points in the shape (2, N)
    """

    # Roughly, x = p*X + distortion
    # See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    # or cv2.projectPoints
    if dist is not None:
        dist = np.asarray(dist).reshape(-1)
        pts_2d = point_cloud[0:2, :]/point_cloud[2, :]
        # print('pts_2d[0, :] = ', pts_2d[0, :])
        # print('pts_2d[1, :] = ', pts_2d[1, :])
        r = pts_2d[0, :]*pts_2d[0, :] + pts_2d[1, :]*pts_2d[1, :]
        # print('r =', r)
        pts_2d[0, :] = pts_2d[0, :]*(1 + dist[0]*r + dist[1]*r*r + dist[4]*r*r*r) + 2*dist[2]*pts_2d[0, :]*pts_2d[1, :] + dist[3]*(
                    r + 2*pts_2d[0, :]*pts_2d[0, :])
        pts_2d[1, :] = pts_2d[1, :]*(1 + dist[0]*r + dist[1]*r*r + dist[4]*r*r*r) + 2*dist[3]*pts_2d[0, :]*pts_2d[1, :] + dist[2]*(
                    r + 2*pts_2d[1, :]*pts_2d[1, :])

        pts_2d[0, :] = p[0, 0]*pts_2d[0, :] + p[0, 1]*pts_2d[1, :] + p[0, 2]
        pts_2d[1, :] = p[1, 0]*pts_2d[0, :] + p[1, 1]*pts_2d[1, :] + p[1, 2]
    else:
        pts_2d = np.dot(p, point_cloud)

        pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
        pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]

        pts_2d = np.delete(pts_2d, 2, 0)
    # I saw a super weird result for image 001100008803
    # Believe it or not, the result is actually correct!
    # Here is the weired pointcloud!
    # corners3d =  [[1.88295422 1.88295422 1.30648903 1.30648903 1.88295422 1.88295422 1.30648903 1.30648903]
    #               [1.75651726 1.75651726 1.75651726 1.75651726 0.50058626 0.50058626 0.50058626 0.50058626]
    #               [1.8464271  1.10691622 1.10691622 1.8464271  1.8464271  1.10691622 1.10691622 1.8464271 ]]

    # if point_cloud[1,0]>1.756 and point_cloud[1,0]<1.757:
    #     print('project_to_image ----> point_cloud = ', point_cloud)
    #     pts_2d_fake = np.dot(p, point_cloud)
    #
    #     pts_2d_fake[0, :] = pts_2d_fake[0, :] / pts_2d_fake[2, :]
    #     pts_2d_fake[1, :] = pts_2d_fake[1, :] / pts_2d_fake[2, :]
    #
    #     pts_2d_fake = np.delete(pts_2d_fake, 2, 0)
    #     print('pts_2d =', pts_2d)
    #     print('pts_2d_fake =', pts_2d_fake)
    return pts_2d


def box_3d_to_object_label(box_3d, obj_type='Pedestrian'):
    """Turns a box_3d into an ObjectLabel

    Args:
        box_3d: 3D box in the format [x, y, z, l, w, h, ry]
        obj_type: Optional, the object type

    Returns:
        ObjectLabel with the location, size, and rotation filled out
    """

    obj_label = ObjectLabel()

    obj_label.type = obj_type

    obj_label.t = box_3d.take((0, 1, 2))
    obj_label.l = box_3d[3]
    obj_label.w = box_3d[4]
    obj_label.h = box_3d[5]
    obj_label.ry = box_3d[6]

    return obj_label


def compute_box_corners_3d(object_label):
    """Computes the 3D bounding box corner positions from an ObjectLabel

    :param object_label: ObjectLabel to compute corners from
    :return: a numpy array of 3D corners if the box is in front of the camera,
             an empty array otherwise
    """

    # Compute rotational matrix
    rot = np.array([[+np.cos(object_label.ry), 0, +np.sin(object_label.ry)],
                    [0, 1, 0],
                    [-np.sin(object_label.ry), 0, +np.cos(object_label.ry)]])

    l = object_label.l
    w = object_label.w
    h = object_label.h

    # 3D BB corners
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))

    corners_3d[0, :] = corners_3d[0, :] + object_label.t[0]
    corners_3d[1, :] = corners_3d[1, :] + object_label.t[1]
    corners_3d[2, :] = corners_3d[2, :] + object_label.t[2]

    return corners_3d


def project_to_image_space(box_3d, calib_p2,
                           truncate=False, image_size=None,
                           discard_before_truncation=True, distortion=None):
    """ Projects a box_3d into image space

    Args:
        box_3d: single box_3d to project. [x, y, z, l, w, h, ry]
        calib_p2: stereo calibration p2 matrix
        truncate: if True, 2D projections are truncated to be inside the image
        image_size: [w, h] must be provided if truncate is True,
            used for truncation
        discard_before_truncation: If True, discard boxes that are larger than
            80% of the image in width OR height BEFORE truncation. If False,
            discard boxes that are larger than 80% of the width AND
            height AFTER truncation.

    Returns:
        Projected box in image space [x1, y1, x2, y2]
            Returns None if box is not inside the image
    """

    obj_label = box_3d_to_object_label(box_3d) # from [x, y, z, l, w, h, ry] => [t, l, w, h, ry]
    corners_3d = compute_box_corners_3d(obj_label)

    projected = project_to_image(corners_3d, calib_p2, dist=distortion)

    x1 = np.amin(projected[0])
    y1 = np.amin(projected[1])
    x2 = np.amax(projected[0])
    y2 = np.amax(projected[1])

    img_box = np.array([x1, y1, x2, y2])

    if truncate:
        if not image_size:
            raise ValueError('Image size must be provided')

        image_w = image_size[0]
        image_h = image_size[1]

        # Discard invalid boxes (outside image space)
        if img_box[0] > image_w or \
                img_box[1] > image_h or \
                img_box[2] < 0 or \
                img_box[3] < 0:
            return None

        # Discard boxes that are larger than 80% of the image width OR height
        if discard_before_truncation:
            img_box_w = img_box[2] - img_box[0]
            img_box_h = img_box[3] - img_box[1]
            if img_box_w > (image_w * 0.8) or img_box_h > (image_h * 0.8):
                return None

        # Truncate remaining boxes into image space
        if img_box[0] < 0:
            img_box[0] = 0
        if img_box[1] < 0:
            img_box[1] = 0
        if img_box[2] > image_w:
            img_box[2] = image_w
        if img_box[3] > image_h:
            img_box[3] = image_h

        # Discard boxes that are covering the the whole image after truncation
        if not discard_before_truncation:
            img_box_w = img_box[2] - img_box[0]
            img_box_h = img_box[3] - img_box[1]
            if img_box_w > (image_w * 0.8) and img_box_h > (image_h * 0.8):
                return None

    return img_box

