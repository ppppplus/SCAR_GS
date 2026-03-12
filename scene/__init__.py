#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, load_ply2dict
from scene.gaussian_model import GaussianModel
from scene.cargs_modelsc import CARGaussianModelSC
# from scene.deform_model import DeformModel
from scene.degs_model import DeGSModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.other_utils import match_gaussians, cal_cluster_centers
import torch
import numpy as np
import random, json
from scene.dataset_readers import fetchObjPly
from typing import List
from utils.graphics_utils import ObjPointCloud

# from encoder.feat_comp.feature_compression import FeatureCompressor
# from encoder.detic_encoder.detic_extractor import get_prompt_embeddings
# from gaussian_renderer.render import render_gs, render_by_dict, render_gs_with_screw
# from encoder.feat_comp.feature_compression import decode_features

class Scene:
    gaussians: GaussianModel
    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], ckpt=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        # if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        # else:
        #     raise ValueError("No scene info file found!")
        if os.path.exists(os.path.join(args.dataset_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.dataset_path, args.semantic_features_type, args.semantic_features_dim, args.images, args.eval)
        elif os.path.exists(os.path.join(args.dataset_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.dataset_path,  args.semantic_features_type, args.semantic_features_dim, args.white_background, args.eval) 
        else:
            assert False, "Could not recognize scene type!"
    
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Cameras extent: ", self.cameras_extent)
        print("Loading Cameras")

        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if (not self.loaded_iter) and (not ckpt):
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            with open(scene_info.obj_ids_path, 'rb') as src_file, open(os.path.join(self.model_path, "obj_ids.npy") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            
            json_cams = []
            camlist = []
            # if scene_info.test_cameras:
            #     camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # for resolution_scale in resolution_scales:
        #     print("Loading Training Cameras")
        #     self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # print("Loading Test Cameras")
            # self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        elif ckpt is False:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.obj_ids, self.cameras_extent, scene_info.semantic_feature_dim) 

    

    def save(self, iteration, is_best=False):
        if is_best:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_best")
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            with open(os.path.join(point_cloud_path, "iter.txt"), 'w') as f:
                f.write(f"iteration: {iteration}")
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    # def save_2gs(self, iteration, num_slots, vis_cano=False, vis_center=False):
    #     point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    #     self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_0.ply"))
    #     self.gaussians1.save_ply(os.path.join(point_cloud_path, "point_cloud_1.ply"))
    #     cano_gs = GaussianModel(self.gaussians.max_sh_degree)
    #     large_motion_state = match_gaussians(os.path.join(point_cloud_path, "point_cloud.ply"), cano_gs, num_slots, vis_cano)
    #     cal_cluster_centers(os.path.join(point_cloud_path, "point_cloud.ply"), num_slots, vis_center)
        
    #     return large_motion_state
    
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class CARGSScene:
    gaussians: CARGaussianModelSC
    def __init__(self, args: ModelParams, gaussians: CARGaussianModelSC, dataset_path, base_model_path, screw_init_path, load_iteration=None, state=0, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.base_model_path = base_model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.base_model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        # if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        # else:
        #     raise ValueError("No scene info file found!")
        if os.path.exists(os.path.join(dataset_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](dataset_path, args.semantic_features_type, args.semantic_features_dim, args.images, args.eval)
        elif os.path.exists(os.path.join(dataset_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](dataset_path,  args.semantic_features_type, args.semantic_features_dim, args.white_background, args.eval) 
        else:
            assert False, "Could not recognize scene type!"
    
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Cameras extent: ", self.cameras_extent)
        print("Loading Cameras")

        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            with open(scene_info.obj_ids_path, 'rb') as src_file, open(os.path.join(self.model_path, "obj_ids.npy") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            
            json_cams = []
            camlist = []
            # if scene_info.test_cameras:
            #     camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # for resolution_scale in resolution_scales:
        #     print("Loading Training Cameras")
        #     self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # print("Loading Test Cameras")
            # self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            # load existing model
            self.gaussians.load_ply(
                os.path.join(self.base_model_path,
                            "point_cloud",
                            "iteration_" + str(self.loaded_iter),
                            "point_cloud.ply")
            )
            # screw_init_path = os.path.join(self.base_model_path, "catch", "ours_" + str(self.loaded_iter), "screw_init.json")
            with open(screw_init_path, "r") as f:
                screw_init = json.load(f)
            self.gaussians.init_mobility_from_change_ids(screw_init)

        else:
            raise RuntimeError(
                "CARScene initialization failed: loaded_iter is required.\n"
            )

    def save(self, iteration, is_best=False):
        if is_best:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_best")
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            with open(os.path.join(point_cloud_path, "iter.txt"), 'w') as f:
                f.write(f"iteration: {iteration}")
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

class CARGSScene_backup:
    gaussians: CARGaussianModelSC
    def __init__(self, args: ModelParams, gaussians: CARGaussianModelSC, base_model_path, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.base_model_path = base_model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.base_model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        # if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        # else:
        #     raise ValueError("No scene info file found!")
        if os.path.exists(os.path.join(args.dataset_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.dataset_path, args.semantic_features_type, args.semantic_features_dim, args.images, args.eval)
        elif os.path.exists(os.path.join(args.dataset_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.dataset_path,  args.semantic_features_type, args.semantic_features_dim, args.white_background, args.eval) 
        else:
            assert False, "Could not recognize scene type!"
    
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Cameras extent: ", self.cameras_extent)
        print("Loading Cameras")

        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            with open(scene_info.obj_ids_path, 'rb') as src_file, open(os.path.join(self.model_path, "obj_ids.npy") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            
            json_cams = []
            camlist = []
            # if scene_info.test_cameras:
            #     camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # for resolution_scale in resolution_scales:
        #     print("Loading Training Cameras")
        #     self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # print("Loading Test Cameras")
            # self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            # load existing model
            self.gaussians.load_ply(
                os.path.join(self.base_model_path,
                            "point_cloud",
                            "iteration_" + str(self.loaded_iter),
                            "point_cloud.ply")
            )
            screw_init_path = os.path.join(self.base_model_path, "catch", "ours_" + str(self.loaded_iter), "screw_init.json")
            with open(screw_init_path, "r") as f:
                screw_init = json.load(f)
            # initialize articulation parameters
            # self.gaussians.init_articulation(screw_init)
            self.gaussians.init_articulation(screw_init)

        else:
            raise RuntimeError(
                "CARScene initialization failed: loaded_iter is required.\n"
            )
        self.end_ply = fetchObjPly(args.dataset_path)

    def save(self, iteration, is_best=False):
        if is_best:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_best")
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            with open(os.path.join(point_cloud_path, "iter.txt"), 'w') as f:
                f.write(f"iteration: {iteration}")
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getEndPcd(self):
        return self.end_ply

    def getEndPcd_byobjids(
        self,
        target_obj_ids: List[int],
        ) -> ObjPointCloud:
        """
        Select points whose obj_id is in target_obj_ids.

        Args:
            pcd (ObjPointCloud): input point cloud
            target_obj_ids (List[int]): object ids to select

        Returns:
            ObjPointCloud: filtered point cloud
        """
        if len(target_obj_ids) == 0:
            raise ValueError("target_obj_ids is empty")

        target_obj_ids = set(target_obj_ids)

        mask = np.isin(self.end_ply.obj_ids, list(target_obj_ids))

        if mask.sum() == 0:
            raise ValueError(
                f"No points found for obj_ids={target_obj_ids}"
            )

        return ObjPointCloud(
            points=self.end_ply.points[mask],
            colors=self.end_ply.colors[mask],
            normals=self.end_ply.normals[mask],
            obj_ids=self.end_ply.obj_ids[mask],
        )
    
class DeGSScene:
    def __init__(self, args: ModelParams, gaussians: DeGSModel, base_model_path, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.end_dataset_path = args.dataset_path
        self.model_path = args.model_path
        self.base_model_path = base_model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.end_ply = None
        self.args = args

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.base_model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        # if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        # else:
        #     raise ValueError("No scene info file found!")
        if os.path.exists(os.path.join(args.dataset_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.dataset_path, args.semantic_features_type, args.semantic_features_dim, args.images, args.eval)
        elif os.path.exists(os.path.join(args.dataset_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.dataset_path,  args.semantic_features_type, args.semantic_features_dim, args.white_background, args.eval) 
        else:
            assert False, "Could not recognize scene type!"
    
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Cameras extent: ", self.cameras_extent)
        print("Loading Cameras")

        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            with open(scene_info.obj_ids_path, 'rb') as src_file, open(os.path.join(self.model_path, "obj_ids.npy") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            
            json_cams = []
            camlist = []
            # if scene_info.test_cameras:
            #     camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        if self.loaded_iter:
            # load existing model
            self.gaussians.load_ply(
                os.path.join(self.base_model_path,
                            "point_cloud",
                            "iteration_" + str(self.loaded_iter),
                            "point_cloud.ply")
            )
            screw_init_path = os.path.join(self.base_model_path, "catch", "ours_" + str(self.loaded_iter), "screw_init.json")
            with open(screw_init_path, "r") as f:
                screw_init = json.load(f)
            # initialize articulation parameters
            # self.gaussians.init_articulation(screw_init)
            self.gaussians.init_deform(screw_init)

        else:
            raise RuntimeError(
                "DeGSScene initialization failed: loaded_iter is required.\n"
            )

        self.end_ply = fetchObjPly(self.end_dataset_path)

    def save(self, iteration, is_best=False):
        if is_best:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_best")
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            with open(os.path.join(point_cloud_path, "iter.txt"), 'w') as f:
                f.write(f"iteration: {iteration}")
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getEndPcd(self):
        return self.end_ply

    def getEndPcd_byobjids(
        self,
        target_obj_ids: List[int],
        ) -> ObjPointCloud:
        """
        Select points whose obj_id is in target_obj_ids.

        Args:
            pcd (ObjPointCloud): input point cloud
            target_obj_ids (List[int]): object ids to select

        Returns:
            ObjPointCloud: filtered point cloud
        """
        if len(target_obj_ids) == 0:
            raise ValueError("target_obj_ids is empty")

        target_obj_ids = set(target_obj_ids)

        mask = np.isin(self.end_ply.obj_ids, list(target_obj_ids))

        if mask.sum() == 0:
            raise ValueError(
                f"No points found for obj_ids={target_obj_ids}"
            )

        return ObjPointCloud(
            points=self.end_ply.points[mask],
            colors=self.end_ply.colors[mask],
            normals=self.end_ply.normals[mask],
            obj_ids=self.end_ply.obj_ids[mask],
        )