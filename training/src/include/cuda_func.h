#ifndef _CUDA_FUNC_H
#define _CUDA_FUNC_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "tqdm.h"
#include "cuda_utils.h"
// #include "interpolation.h"


void gen_voxels(
    const std::vector<float3> vertices,
    const std::vector<int3> faces, 
    const std::vector<float3> colors,
    const std::vector<float3> centers,
    const std::vector<float3> sizes,
    const std::vector<int> ConFaceIdx,
    const std::vector<int> ConFaceNum,
    const float voxel_size,
    const int num_voxel,
    std::vector<float4> &voxels);


void get_VisImg(
    const std::vector<float3> vertices,
    const std::vector<int3> faces, 
    const std::vector<float3> centers,
    const std::vector<int> IndexMap,
    const std::vector<int> BConFaceIdx,
    const std::vector<int2> BConFaceNum,
    const float* Ks, const float* C2Ws,
    const int num_camera,
    const int3 tile_shape,
    const float3 scene_min_corner,
    const float tile_size,
    const int max_tracingTile,
    const int num_block,
    const int height, const int width,
    std::vector<int> &VisImg);



void get_trainData(
    const int trainTileIdx,
    const std::string img_path,
    const std::vector<float3> vertices,
    const std::vector<int3> faces, 
    const std::vector<float3> colors,
    const std::vector<float3> centers,
    const std::vector<int> IndexMap,
    const std::vector<int> BConFaceIdx,
    const std::vector<int2> BConFaceNum,
    const std::vector<int> imgIdxs,
    const float* Ks, const float* C2Ws,
    const int num_camera,
    const int3 tile_shape,
    const float3 scene_min_corner,
    const float tile_size,
    const int max_tracingTile,
    const int num_block,
    const int height, const int width,
    const bool debug,
    std::vector<float3> &data);


void get_trainData_v2(
    const int trainTileIdx,
    const std::string img_path,
    const std::string diffuse_path,
    const std::vector<float3> vertices,
    const std::vector<int3> faces, 
    const std::vector<float3> colors,
    const std::vector<float3> centers,
    const std::vector<int> IndexMap,
    const std::vector<int> BConFaceIdx,
    const std::vector<int2> BConFaceNum,
    const std::vector<int> imgIdxs,
    const float* Ks, const float* C2Ws,
    const int num_camera,
    const int3 tile_shape,
    const float3 scene_min_corner,
    const float tile_size,
    const int max_tracingTile,
    const int num_block,
    const int height, const int width,
    const bool debug,
    std::vector<float> &data);


void get_trainData_v3(
    const int trainTileIdx,
    const std::string img_path,
    const std::string dep_path,
    const std::vector<float3> vertices,
    const std::vector<int3> faces, 
    const std::vector<float3> centers,
    const std::vector<int> IndexMap,
    const std::vector<int> BConFaceIdx,
    const std::vector<int2> BConFaceNum,
    const std::vector<int> imgIdxs,
    const float* Ks, const float* C2Ws,
    const int num_camera,
    const int3 tile_shape,
    const float3 scene_min_corner,
    const float tile_size,
    const int max_tracingTile,
    const int num_block,
    const int height, const int width,
    const int patch_size, 
    const bool debug,
    std::vector<float> &data);


void get_trainData_v4(
    const int trainTileIdx,
    const std::string img_path,
    const std::string diffuse_path,
    const std::vector<float3> centers,
    const std::vector<int> IndexMap,
    const std::vector<int> imgIdxs,
    const std::vector<int> SparseToGroup,
    const int num_render_tiles, 
    float* voxels, 
    short* nodes, 
    const float* Ks, const float* C2Ws,
    const int num_camera,
    const int3 tile_shape,
    const float3 scene_min_corner,
    const float tile_size,
    const int num_voxel,
    const float voxel_size,
    const float sample_step,
    const int max_tracingTile,
    const int height, const int width,
    const int patch_size, 
    const bool debug,
    std::vector<float> &data);

void render_TileScene(
    const std::vector<float4> voxels,
    const std::vector<float3> centers,
    const float3 scene_min_corner,
    const int3 tile_shape,
    const float tile_size,
    const int max_tracingTile,
    const std::vector<int> IndexMap,
    const int num_voxel,
    const float voxel_size,
    const float* K,
    const float* C2W,
    const int height, 
    const int width,
    const int Nsamples,
    std::vector<float3> &frame);
    
#endif 