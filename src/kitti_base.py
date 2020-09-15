import open3d as o3d
import json
import os
import numpy as np
from PIL import Image
import glob
import cv2
import shutil
import yaml
import colorsys

def calib_velo2cam(fn_v2c):
    """
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    """
    for line in open(fn_v2c, "r"):
        (key, val) = line.split(':', 1)
        if key == 'R':
            R = np.fromstring(val, sep=' ')
            R = R.reshape(3, 3)
        if key == 'T':
            T = np.fromstring(val, sep=' ')
            T = T.reshape(3, 1)
    return R, T

def calib_cam2cam(fn_c2c, mode = '02'):
    """
    If your image is 'rectified image' :get only Projection(P : 3x4) matrix is enough
    but if your image is 'distorted image'(not rectified image) :
        you need undistortion step using distortion coefficients(5 : D)
    In this code, only P matrix info is used for rectified image
    """
    # with open(fn_c2c, "r") as f: c2c_file = f.readlines()
    for line in open(fn_c2c, "r"):
        (key, val) = line.split(':', 1)
        if key == ('P_rect_' + mode):
            P = np.fromstring(val, sep=' ')
            P = P.reshape(3, 4)
            P = P[:3, :3]  # erase 4th column ([0,0,0])
    return P

#source: http://www.open3d.org/docs/release/tutorial/Basic/kdtree.html
def filter_ego_pts(lidar):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar)
    # # pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    # # print("Find closest 40 pts, paint green.")
    [k, idx, _] = pcd_tree.search_knn_vector_3d(np.array([0,0,0]), 40)
    # # np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
    # # print("Visualize the point cloud.")
    # # o3d.visualization.draw_geometries([pcd])
    # source_pts = np.asarray(pcd.points)
    # mask = np.arange(source_pts.shape[0], dtype=int)
    # mask = np.delete(mask, np.array(idx))
    # pcd_result = o3d.geometry.PointCloud()
    # pcd_result.points = o3d.utility.Vector3dVector(source_pts[mask,:])
    # return pcd_result
    return np.array(idx)

class PointCloud_Vis():
    def __init__(self,cfg, new_config = False, width = 800, height = 800):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=width, height=height, left=0, top=0)
        self.vis.get_render_option().load_from_json('config/render_option.json')
        self.cfg = cfg

        self.new_config = new_config
        # Modify json file or there will be errors when we change window size
        print('Load config file [%s]'%(self.cfg))
        data = json.load(open(self.cfg,'r'))
        data['intrinsic']['width'] = width
        data['intrinsic']['height'] = height
        data['intrinsic']['intrinsic_matrix'][6] = (width-1)/2
        data['intrinsic']['intrinsic_matrix'][7] = (height-1)/2
        json.dump(data, open(self.cfg,'w'),indent=4)
        self.param = o3d.io.read_pinhole_camera_parameters(self.cfg)

        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        self.vis.register_key_callback(32, lambda vis: exit())
    
    def __del__(self):
        self.vis.destroy_window()

    def update(self,pcd):
        self.pcd.points = pcd.points
        self.pcd.colors = pcd.colors
        self.vis.remove_geometry(self.pcd)
        self.vis.add_geometry(self.pcd)
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.param)
        # self.vis.update_geometry() # bug fix for open3d 0.9.0.0
        self.vis.poll_events()
        self.vis.update_renderer()

        if self.new_config:
            print('Move the frame to the place you want')
            print('---Press [Q] to save---')
            self.vis.run()
            data = json.load(open(self.cfg,'r'))

            self.param = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters(self.cfg,self.param)
            self.new_config = False

            # add our own parameters
            cfg = json.load(open(self.cfg,'r'))
            cfg['h_fov'] = data['h_fov']
            cfg['v_fov'] = data['v_fov']
            cfg['x_range'] = data['x_range']
            cfg['y_range'] = data['y_range']
            cfg['z_range'] = data['z_range']
            cfg['d_range'] = data['d_range']
            json.dump(cfg, open(self.cfg,'w'),indent=4)

            print('Saved. Please restart using [%s]' % self.cfg)
            exit()

    def capture_screen(self,fn, depth = False):
        if depth:
            self.vis.capture_depth_image(fn, False)
        else:
            self.vis.capture_screen_image(fn, False)

class Semantic_KITTI_Utils():
    def __init__(self, root, n_scans_stitched):
        self.root = root
        self.n_scans_stitched = n_scans_stitched
        self.init()

    def set_part(self, part='00', saved_poses_path=''):
        length = {
            '00': 4540,'01':1100,'02':4660,'03':800,'04':270,'05':2760,
            '06':1100,'07':1100,'08':4070,'09':1590,'10':1200
        }
        assert part in length.keys(), 'Only %s are supported' %(length.keys())
        # self.sequence_root = os.path.join(self.root, 'sequences/%s/'%(part))
        self.frame_root = os.path.join(self.root, 'data_odometry_color/dataset/sequences/', part)
        self.vel_root = os.path.join(self.root, 'data_odometry_velodyne/dataset/sequences/', part)
        self.label_root = os.path.join(self.root, 'data_odometry_labels/dataset/sequences/', part)
        # self.overlay_root = os.path.join(self.root, "SLAM_"+'10'+'/dataset/sequences/', part) #str(self.n_scans_stitched)

        assert os.path.exists(self.frame_root), 'Broken dataset %s' % (self.frame_root)
        assert os.path.exists(self.vel_root), 'Broken dataset %s' % (self.vel_root)
        assert os.path.exists(self.label_root), 'Broken dataset %s' % (self.label_root)

        self.index = 0
        self.max_index = length[part]

        saved_poses_file = os.path.join(saved_poses_path, "global_poses_"+part+".txt")
        if os.path.exists(saved_poses_file):
            saved_poses = np.loadtxt(saved_poses_file)
            saved_poses = saved_poses.reshape((saved_poses.shape[0], 4, 4))
            self.global_poses = [saved_poses[i] for i in range(np.shape(saved_poses)[0])]

        return self.max_index
    
    def get_max_index(self):
        return self.max_index

    def init(self):
        self.R, self.T = calib_velo2cam('config/calib_velo_to_cam.txt')
        self.P = calib_cam2cam('config/calib_cam_to_cam.txt' ,mode="02")
        self.RT = np.concatenate((self.R, self.T), axis=1)

        self.sem_cfg = yaml.load(open('config/semantic-kitti.yaml','r'), Loader=yaml.SafeLoader)
        self.class_names = self.sem_cfg['labels']
        self.learning_map = self.sem_cfg['learning_map']
        self.learning_map_inv = self.sem_cfg['learning_map_inv']
        self.learning_ignore = self.sem_cfg['learning_ignore']
        self.sem_color_map = self.sem_cfg['color_map']

        self.kitti_color_map = [[0,0,0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                        [153, 153, 153], [250, 170, 30], [220, 220, 0],[107, 142, 35], [152, 251, 152], [0, 130, 180],
                        [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230],[119, 11, 32]]

        self.points = o3d.geometry.PointCloud()
        self.mask_pts_to_remove = np.array([])

    def load_points(self, ind):
        # load point cloud
        lidar = np.fromfile(os.path.join(self.vel_root, 'velodyne/%06d.bin' %(ind)), dtype=np.float32).reshape((-1, 4))[:, :3]
        # find ind of pts below ground
        mask_below_ground = (lidar[:,2] < -2) # found empirically
        ind_below_ground = np.where(mask_below_ground)[0]
        # find ind of points due to ego vehicle
        ind_ego_vehicle = filter_ego_pts(lidar)
        self.ind_pts_to_remove = np.append(ind_below_ground, ind_ego_vehicle)
        lidar = np.delete(lidar, self.ind_pts_to_remove, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar)
        return pcd

    def get_sem_label(self, label, pc):
        label  = np.delete(label, self.ind_pts_to_remove)

        if label.shape[0] == pc.shape[0]:
            self.sem_label = label & 0xFFFF  # semantic label in lower half
            self.inst_label = label >> 16  # instance id in upper half
            assert((self.sem_label + (self.inst_label << 16) == label).all()) # sanity check
        else:
            print("Points shape: ", pc.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")

    def load(self,index = None):
        """  Load the frame, point cloud and semantic labels from file """
        label = np.array([], dtype=np.uint32)
        self.index = index
        if self.index == self.max_index:
            print('End of sequence')
            return False
 
        if self.index is not 0:
            # incorporate delta of transform bw index-1 and index
            self.points.transform(np.dot(np.linalg.inv(self.global_poses[self.index]), self.global_poses[self.index-1]))
            # remove pc at ind-1 
            pcd2 = self.load_points(self.index -1)
            pt_to_remove_list = list(range(0,np.array(pcd2.points).shape[0]))
            self.points = o3d.geometry.select_down_sample(self.points, pt_to_remove_list, invert=True)
            # remove sem label at ind-1
            self.sem_label = np.delete(self.sem_label, pt_to_remove_list)

            if (self.index + self.n_scans_stitched) < self.max_index: 
                # append pc at self.index + self.n_scans_stitched
                pcd1 = self.load_points(self.index - 1 + self.n_scans_stitched)
                # incorporate delta of transform bw index and index and index - 1 + n_scans_stitched
                delta_tf = np.dot(np.linalg.inv(self.global_poses[self.index]), self.global_poses[self.index - 1 + self.n_scans_stitched])
                pcd1.transform(delta_tf)
                self.points = self.points + pcd1
                # append sem label at self.index + self.n_scans_stitched
                label = np.fromfile(os.path.join(self.label_root, 'labels/%06d.label' %(self.index - 1 + self.n_scans_stitched)), dtype=np.uint32).reshape((-1))
                self.get_sem_label(label, np.array(pcd1.points))
        else:
            for i in range(self.index, self.index + self.n_scans_stitched):
                pcd1 = self.load_points(i)
                # incorporate delta of transform bw index and index and index - i
                delta_tf = np.dot(np.linalg.inv(self.global_poses[self.index]), self.global_poses[i])
                pcd1.transform(delta_tf)
                # append pcd1 to points
                self.points = self.points + pcd1
                # append sem label to labels
                label = np.fromfile(os.path.join(self.label_root, 'labels/%06d.label' %(i)), dtype=np.uint32).reshape((-1))
                self.get_sem_label(label, np.array(pcd1.points))

        self.frame = cv2.imread(os.path.join(self.frame_root, 'image_2/%06d.png' % (self.index)))
        assert self.frame is not None, 'Broken dataset %s' % (self.frame_root)
        
        # self.overlay_frame = cv2.imread(os.path.join(self.overlay_root, 'slam_depth_overlay_%06d.png' %(self.index)))

        return True
    
    def set_filter(self, h_fov, v_fov, x_range = None, y_range = None, z_range = None, d_range = None):
        # rough velodyne azimuth range corresponding to camera horizontal fov
        self.h_fov = h_fov if h_fov is not None else (-180, 180)
        self.v_fov = v_fov if v_fov is not None else (-25, 2)
        self.x_range = x_range if x_range is not None else (-10000, 10000)
        self.y_range = y_range if y_range is not None else (-10000, 10000)
        self.z_range = z_range if z_range is not None else (-10000, 10000)
        self.d_range = d_range if d_range is not None else (-10000, 10000)

        self.min_bound = [self.x_range[0], self.y_range[0], self.z_range[0]]
        self.max_bound = [self.x_range[1], self.y_range[1], self.z_range[1]]

    def hv_in_range(self, m, n, fov, fov_type='h'):
        """ extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit 
            horizontal limit = azimuth angle limit
            vertical limit = elevation angle limit
        """
        if fov_type == 'h':
            return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), \
                                    np.arctan2(n, m) < (-fov[0] * np.pi / 180))
        elif fov_type == 'v':
            return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), \
                                    np.arctan2(n, m) > (fov[0] * np.pi / 180))
        else:
            raise NameError("fov type must be set between 'h' and 'v' ")

    def box_in_range(self,x,y,z,d, x_range, y_range, z_range, d_range):
        """ extract filtered in-range velodyne coordinates based on x,y,z limit """
        return np.logical_and.reduce((
                x > x_range[0], x < x_range[1],
                y > y_range[0], y < y_range[1],
                z > z_range[0], z < z_range[1],
                d > d_range[0], d < d_range[1]))

    def points_basic_filter(self, points):
        """
            filter points based on h,v FOV and x,y,z distance range.
            x,y,z direction is based on velodyne coordinates
            1. azimuth & elevation angle limit check
            2. x,y,z distance limit
            return a bool array
        """
        assert points.shape[1] == 4, points.shape # [N,3]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) # this is much faster than d = np.sqrt(np.power(points,2).sum(1))

        # extract in-range fov points
        h_points = self.hv_in_range(x, y, self.h_fov, fov_type='h')
        v_points = self.hv_in_range(d, z, self.v_fov, fov_type='v')
        combined = np.logical_and(h_points, v_points)

        # extract in-range x,y,z points
        in_range = self.box_in_range(x,y,z,d, self.x_range, self.y_range, self.z_range, self.d_range)
        combined = np.logical_and(combined, in_range)

        return combined

    def extract_points(self,voxel_size = 0.01):
        # filter in range points based on fov, x,y,z range setting
        combined = self.points_basic_filter(self.points)
        points = self.points[combined]
        label = self.sem_label[combined]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])

        # approximate_class must be set to true
        # see this issue for more info https://github.com/intel-isl/o3d/issues/1085
        # pcd, trace = pcd.voxel_down_sample_and_trace(voxel_size,self.min_bound,self.max_bound,approximate_class=True)
        pcd, trace = o3d.voxel_down_sample_and_trace(pcd, voxel_size,self.min_bound,self.max_bound,approximate_class=True)
        to_index_org = np.max(trace, 1)

        pts = points[to_index_org]
        sem_label = label[to_index_org]
        self.pts = pts
        colors = np.array([self.sem_color_map[x] for x in sem_label])
        pcd.colors = o3d.utility.Vector3dVector(colors/255.0)

        return pcd,sem_label

    def get_in_view_pts(self, pcd, sem_label):
        """ 
            Convert o3d.geometry.PointCloud object to [4, N] array
                        [x_1 , x_2 , .. ]
            xyz_v   =   [y_1 , y_2 , .. ]
                        [z_1 , z_2 , .. ]
                        [ 1  ,  1  , .. ]
        """
        # The [N,3] downsampled array
        pts_3d = np.asarray(pcd.points)

        # finter out the points not in view
        h_points = self.hv_in_range(pts_3d[:,0], pts_3d[:,1], [-50,50], fov_type='h')
        pts_3d = pts_3d[h_points]
        sem_label = sem_label[h_points]

        return pts_3d, sem_label

    def project_3d_to_2d(self, pts_3d):
        assert pts_3d.shape[1] == 3, pts_3d.shape
        pts_3d = pts_3d.copy()
        
        # Create a [N,1] array
        one_mat = np.ones((pts_3d.shape[0], 1),dtype=np.float32)

        # Concat and change shape from [N,3] to [N,4] to [4,N]
        xyz_v = np.concatenate((pts_3d, one_mat), axis=1).T

        # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
        for i in range(xyz_v.shape[1]):
            xyz_v[:3, i] = np.matmul(self.RT, xyz_v[:, i])

        xyz_c = xyz_v[:3]

        # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
        for i in range(xyz_c.shape[1]):
            xyz_c[:, i] = np.matmul(self.P, xyz_c[:, i])

        # normalize image(pixel) coordinates(x,y)
        xy_i = xyz_c / xyz_c[2]

        # get pixels location
        pts_2d = xy_i[:2].T

        x, y, z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        dist_normalize = (dist - dist.min()) / (dist.max() - dist.min())
        color = [[int(x*255) for x in colorsys.hsv_to_rgb(hue,1,1)] for hue in dist_normalize]
        return pts_2d, color

    def draw_2d_points(self, pts_2d, color):
        """ draw 2d points in camera image """
        assert pts_2d.shape[1] == 2, pts_2d.shape

        image = self.frame.copy()
        pts_2d = pts_2d.astype(np.int32).tolist()

        for (x,y),c in zip(pts_2d,color):
            cv2.circle(image, (x, y), 2, [c[2],c[1],c[0]], -1)
            
        return image

    def draw_2d_sem_points(self, pts_2d, sem_label):
        """ draw 2d points in camera image """
        assert pts_2d.shape[1] == 2, pts_2d.shape
        assert pts_2d.shape[0] == sem_label.shape[0], str(pts_2d.shape) + ' '+  str(sem_label.shape)

        image = self.frame.copy()
        pts_2d = pts_2d.astype(np.int32).tolist()
        colors = [self.sem_color_map[x] for x in sem_label.tolist()]

        for (x,y),c in zip(pts_2d,colors):
            cv2.circle(image, (x, y), 2, [c[2],c[1],c[0]], -1)
        return image

    def draw_2d_sem_points_with_learning_mapping(self, pts_2d, sem_label_learn):
        """ draw 2d points in camera image """
        assert pts_2d.shape[1] == 2, pts_2d.shape
        assert pts_2d.shape[0] == sem_label_learn.shape[0], str(pts_2d.shape) + ' '+  str(sem_label_learn.shape)
        
        image = self.frame.copy()
        pts_2d = pts_2d.astype(np.int32).tolist()
        colors = [self.kitti_color_map[x] for x in sem_label_learn.tolist()]

        for (x,y),c in zip(pts_2d,colors):
            cv2.circle(image, (x, y), 2, [c[2],c[1],c[0]], -1)
        return image

    def learning_mapping(self,sem_label):
        # Note: Here the 19 classs are different from the original KITTI 19 classes
        num_classes = 20
        class_names = [
            'unlabelled',     # 0
            'car',            # 1
            'bicycle',        # 2
            'motorcycle',     # 3
            'truck',          # 4
            'other-vehicle',  # 5
            'person',         # 6
            'bicyclist',      # 7
            'motorcyclist',   # 8
            'road',           # 9
            'parking',        # 10
            'sidewalk',       # 11
            'other-ground',   # 12
            'building',       # 13
            'fence',          # 14
            'vegetation',     # 15
            'trunk',          # 16
            'terrain',        # 17
            'pole',           # 18
            'traffic-sign'    # 19
        ]
        sem_label_learn = [self.learning_map[x] for x in sem_label]
        sem_label_learn = np.array(sem_label_learn, dtype=np.uint8)
        return sem_label_learn

    def inv_learning_mapping(self,sem_label_learn):
        sem_label = [self.learning_map_inv[x] for x in sem_label_learn]
        sem_label = np.array(sem_label, dtype=np.uint16)
        return sem_label