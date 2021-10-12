import numpy as np
import cv2

# 3D 模型
model_points = np.array([
    (0.0, 0.0, 0.0),  # 鼻頭
    (0.0, -330.0, -65.0),  # 下巴
    (-225.0, 170.0, -135.0),  # 左眼中心
    (225.0, 170.0, -135.0),  # 右眼中心
    (-150.0, -150.0, -125.0),  # 嘴巴左邊中心
    (150.0, -150.0, -125.0)  # 嘴巴右邊中心
])


class HeadPoseEstimator:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height

    @staticmethod
    def _get_2d_points(rotation_vector, translation_vector, camera_matrix, dist_coeffs, val):
        point_3d = []
        rear_size = val[0]
        rear_depth = val[1]
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = val[2]
        front_depth = val[3]
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # 將3D座標投影到2D平面上
        (point_2d, _) = cv2.projectPoints(point_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        point_2d = np.int32(point_2d.reshape(-1, 2))
        return point_2d

    def _head_pose_points(self, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
        rear_size = 1
        rear_depth = 0
        front_size = self.frame_width
        front_depth = front_size * 2
        val = [rear_size, rear_depth, front_size, front_depth]
        point_2d = self._get_2d_points(rotation_vector, translation_vector, camera_matrix, dist_coeffs, val)
        p1 = point_2d[2]
        p2 = (point_2d[5] + point_2d[8]) // 2
        return tuple(p1), tuple(p2)

    def head_pose_estimate(self, shape):
        face_3d_points = np.array([
            shape[33],  # 鼻頭
            shape[8],  # 下巴
            shape[36],  # 左眼中心
            shape[45],  # 右眼中心
            shape[48],  # 嘴巴左邊中心
            shape[54]  # 嘴巴右邊中心
        ], dtype="double")

        # 粗估攝影機相關參數
        focal_length = self.frame_width
        center = (self.frame_width / 2, self.frame_height / 2)
        camera_matrix = np.array([[
            focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double")

        # 假設攝影機都是已對焦
        dist_coeffs = np.zeros((4, 1))

        # 計算旋轉與轉換矩陣
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points,
            face_3d_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE)

        # 將一個"與臉部垂直"的3D座標投影到2D平面上
        (nose_end_point2D, _) = cv2.projectPoints(np.array([0.0, 0.0, 1000.0]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)

        # 取得投影到2D平面的點 (後面用來計算臉部垂直方向角度)
        vertical_p1 = (int(face_3d_points[0][0]), int(face_3d_points[0][1]))
        vertical_p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        # 取得水平方向角度用的座標
        (horizontal_p1, horizontal_p2) = self._head_pose_points(rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        return vertical_p1, vertical_p2, horizontal_p1, horizontal_p2
