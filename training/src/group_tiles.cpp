#include "cuda_func.h"
#include "plyIO.h"
#define PY_SSIZE_T_CLEAN

#include <cnpy.h>
#include <Python.h>
#include <time.h>
#include <algorithm>

int mean_tiles_per_group, max_tiles_per_group, min_tiles_per_group;

int majorityElement(std::vector<int>& nums) {
    std::unordered_map<int, int> counter;
    for (int num : nums) {
        if (++counter[num] > nums.size() / 2) {
            return num;
        }
    }
    return 0;
}

int get_index(int x, int y, int z, int3 shape)
{
    return x * shape.y * shape.z + y * shape.z + z;
}
void get_loc(int &x, int &y, int &z, int index, int3 shape)
{
    x = index / (shape.y * shape.z);
    int temp = index % (shape.y * shape.z);
    y = temp / shape.z;
    z = temp % shape.z;
}


void BFS_search(std::queue<int> Q,
                int ref_label,
                int3 tile_shape, 
                std::vector<int> IndexMap, 
                std::vector<int> tile_labels,
                std::vector<bool> &visited,
                std::vector<int> &group_tiles)
{
    while (!Q.empty())
    {
        if (group_tiles.size() >= mean_tiles_per_group) break;
        int ref_denseIdx = Q.front();
        Q.pop();
        int tileIdx = IndexMap[ref_denseIdx];
        if (tileIdx == -1) continue;
        if (visited[tileIdx]) continue;
        int label = tile_labels[tileIdx];
        if (label != ref_label) continue;

        group_tiles.emplace_back(tileIdx);
        visited[tileIdx] = true;

        int x,y,z;
        get_loc(x,y,z,ref_denseIdx,tile_shape);

        int min_x = max(x-1, 0);
        int max_x = min(x+1, tile_shape.x-1);
        int min_y = max(y-1, 0);
        int max_y = min(y+1, tile_shape.y-1);
        int min_z = max(z-1, 0);
        int max_z = min(z+1, tile_shape.z-1);

        for (int _x = min_x; _x <= max_x; _x++)
        {
            int denseIdx = get_index(_x,y,z,tile_shape);
            Q.push(denseIdx);
        }
        for (int _y = min_y; _y <= max_y ; _y++)
        {
            int denseIdx = get_index(x,_y,z,tile_shape);
            Q.push(denseIdx);  
        }
        for(int _z = min_z; _z <= max_z; _z++)
        {
            int denseIdx = get_index(x,y,_z,tile_shape);
            Q.push(denseIdx); 
        }

        // for (int _x = min_x; _x <= max_x ; _x++)
        // {
        //     for (int _y = min_y; _y <= max_y ; _y++)
        //     {
        //         for(int _z = min_z; _z <= max_z; _z++)
        //         {
        //             int denseIdx = get_index(_x,_y,_z,tile_shape);
        //             Q.push(denseIdx);
        //         }
        //     }
        // }
    }
}

void DFS_search(int ref_denseIdx, int ref_label,
                int3 tile_shape, 
                std::vector<int> IndexMap, 
                std::vector<int> tile_labels,
                std::vector<bool> &visited,
                std::vector<int> &group_tiles)
{
    // std::cout << ref_denseIdx << " " << IndexMap.size() << std::endl;
    if (group_tiles.size() == mean_tiles_per_group) return;
    int tileIdx = IndexMap[ref_denseIdx];
    if (tileIdx == -1) return;
    if (visited[tileIdx]) return;

    int label = tile_labels[tileIdx];
    if (label != ref_label) return;

    group_tiles.emplace_back(tileIdx);
    visited[tileIdx] = true;

    int x,y,z;
    get_loc(x,y,z,ref_denseIdx,tile_shape);

    int min_x = max(x-1, 0);
    int max_x = min(x+1, tile_shape.x-1);
    int min_y = max(y-1, 0);
    int max_y = min(y+1, tile_shape.y-1);
    int min_z = max(z-1, 0);
    int max_z = min(z+1, tile_shape.z-1);

    for (int _x = min_x; _x <= max_x; _x++)
    {
        int denseIdx = get_index(_x,y,z,tile_shape);
        DFS_search(denseIdx, ref_label, tile_shape, IndexMap, tile_labels, visited, group_tiles);
    }
    for (int _y = min_y; _y <= max_y ; _y++)
    {
        int denseIdx = get_index(x,_y,z,tile_shape);
        DFS_search(denseIdx, ref_label, tile_shape, IndexMap, tile_labels, visited, group_tiles);
    }
    for(int _z = min_z; _z <= max_z; _z++)
    {
        int denseIdx = get_index(x,y,_z,tile_shape);
        DFS_search(denseIdx, ref_label, tile_shape, IndexMap, tile_labels, visited, group_tiles);
    }
    // for (int _x = min_x; _x <= max_x ; _x++)
    // {
    //     for (int _y = min_y; _y <= max_y ; _y++)
    //     {
    //         for(int _z = min_z; _z <= max_z; _z++)
    //         {
    //             int denseIdx = get_index(_x,_y,_z,tile_shape);
    //             DFS_search(denseIdx, ref_label, tile_shape, IndexMap, tile_labels, visited, group_tiles);
    //         }
    //     }
    // }
}


void merge_neighbor_group(
    std::vector<int> ref_group_tileIdxs,
    int ref_groupIdx,
    int3 tile_shape,
    std::vector<int> IndexMap,
    std::vector<int> SparseToDense,
    std::vector<int> &tile2group,
    std::vector<std::vector<int>> &groups)
{

    std::vector<int> num_tiles_of_neighbor;
    std::vector<int> neighbor_groupIdx;

    for (int i=0; i<ref_group_tileIdxs.size(); i++)
    {
        int ref_tileIdx = ref_group_tileIdxs[i];
        int ref_denseIdx = SparseToDense[ref_tileIdx];
        int x,y,z;
        get_loc(x,y,z,ref_denseIdx,tile_shape);

        int min_x = max(x-1, 0);
        int max_x = min(x+1, tile_shape.x-1);
        int min_y = max(y-1, 0);
        int max_y = min(y+1, tile_shape.y-1);
        int min_z = max(z-1, 0);
        int max_z = min(z+1, tile_shape.z-1);

        for (int _x = min_x; _x <= max_x ; _x++)
        {
            int denseIdx = get_index(_x,y,z,tile_shape);
            int tileIdx = IndexMap[denseIdx];
            if (tileIdx == -1) continue;
            int groupIdx = tile2group[tileIdx];
            if (groupIdx == ref_groupIdx) continue;
            if (groupIdx == -1) continue;
            if (groups[groupIdx].size() != 0)
            {
                num_tiles_of_neighbor.emplace_back((int)groups[groupIdx].size());
                neighbor_groupIdx.emplace_back(groupIdx);
            }
        }
        for (int _y = min_y; _y <= max_y ; _y++)
        {
            int denseIdx = get_index(x,_y,z,tile_shape);
            int tileIdx = IndexMap[denseIdx];
            if (tileIdx == -1) continue;
            int groupIdx = tile2group[tileIdx];
            if (groupIdx == ref_groupIdx) continue;
            if (groupIdx == -1) continue;
            if (groups[groupIdx].size() != 0)
            {
                num_tiles_of_neighbor.emplace_back((int)groups[groupIdx].size());
                neighbor_groupIdx.emplace_back(groupIdx);
            }
        }
        for(int _z = min_z; _z <= max_z; _z++)
        {
            int denseIdx = get_index(x,y,_z,tile_shape);
            int tileIdx = IndexMap[denseIdx];
            if (tileIdx == -1) continue;
            int groupIdx = tile2group[tileIdx];
            if (groupIdx == ref_groupIdx) continue;
            if (groupIdx == -1) continue;
            if (groups[groupIdx].size() != 0)
            {
                num_tiles_of_neighbor.emplace_back((int)groups[groupIdx].size());
                neighbor_groupIdx.emplace_back(groupIdx);
            }
        }
   

    }

    int min_num_tiles = max_tiles_per_group + 10000000;
    int min_groupIdx = -1;
    for (int i=0; i<num_tiles_of_neighbor.size(); i++)
    {
        if (num_tiles_of_neighbor[i] < min_num_tiles)
        {
            min_num_tiles = num_tiles_of_neighbor[i];
            min_groupIdx = neighbor_groupIdx[i];
        }
    }

    if (min_groupIdx != -1)
    {
        groups[min_groupIdx].insert(groups[min_groupIdx].end(), groups[ref_groupIdx].begin(), groups[ref_groupIdx].end());
        groups[ref_groupIdx].clear();
        for (int i=0; i<groups[min_groupIdx].size(); i++)
        {
            int tileIdx = groups[min_groupIdx][i];
            tile2group[tileIdx] = min_groupIdx;
        }
    }
}

std::vector<int> group_tiles(
    std::string model_path, // mesh 
    std::string label_path, // one face one label 
    std::vector<float3> centers,
    std::vector<int> IndexMap,
    std::vector<int> SparseToDense,
    std::vector<int> ignores,
    float3 scene_min_corner,
    float tile_size,
    int3 tile_shape, 
    std::vector<int> &tiles_list,
    std::vector<int> &group_start)
{
    std::cout << "读取模型" << std::endl;
    std::vector<float3> vertices;
    std::vector<int3> faces;
    std::vector<float2> uv_array;
    read_plyFile(model_path, vertices, faces, uv_array);

    cnpy::NpyArray label_npy = cnpy::npy_load(label_path);
    int* face_label = label_npy.data<int>();

    std::cout << "完成数据读取" << std::endl;

    int num_face = (int)faces.size();

    std::vector<int> tile_labels;
    std::vector<std::vector<int>> temp_grouped_tiles;
    for (int i=0; i<centers.size(); i++)
    {
        std::vector<int> temp;
        temp_grouped_tiles.emplace_back(temp);
        tile_labels.emplace_back(-1);
    }
    
    tqdm bar1;
    for (int i=0; i<num_face; i++)
    {
        bar1.progress(i, num_face);
        int3 vidx = faces[i];
        float3 A = vertices[vidx.x];
        float3 B = vertices[vidx.y];
        float3 C = vertices[vidx.z];

        int label = face_label[i];

        float3 min_c = fminf(fminf(A, B), C);
        float3 max_c = fmaxf(fmaxf(A, B), C);

        min_c = min_c - scene_min_corner;
        max_c = max_c - scene_min_corner;


        int3 min_idx = make_int3(min_c / tile_size);
        int3 max_idx = make_int3(max_c / tile_size);

        min_idx.x = clamp(min_idx.x, 0, tile_shape.x-1);
        min_idx.y = clamp(min_idx.y, 0, tile_shape.y-1);
        min_idx.z = clamp(min_idx.z, 0, tile_shape.z-1);

        max_idx.x = clamp(max_idx.x, 0, tile_shape.x-1);
        max_idx.y = clamp(max_idx.y, 0, tile_shape.y-1);
        max_idx.z = clamp(max_idx.z, 0, tile_shape.z-1);

        for (int x=min_idx.x; x<=max_idx.x; x++)
        {
            for (int y=min_idx.y; y<=max_idx.y; y++)
            {
                for (int z=min_idx.z; z<=max_idx.z; z++)
                {
                    int dense_idx = get_index(x, y, z, tile_shape);
                    int tileIdx = IndexMap[dense_idx];
                    if (tileIdx == -1) continue;

                    float3 tile_center = centers[tileIdx];
                    bool intersect = AABB_triangle_intersection(tile_center, make_float3(tile_size,tile_size,tile_size), A, B, C);

                    if (intersect == false) continue;

                    temp_grouped_tiles[tileIdx].emplace_back(label);
                }
            }
        }
    }
    bar1.finish();


    for (int i=0; i<temp_grouped_tiles.size(); i++)
    {
        if (temp_grouped_tiles[i].size() == 0) continue;
        int lable = majorityElement(temp_grouped_tiles[i]);
        tile_labels[i] = lable;
    }
 
    std::vector<bool> visited;
    for (int i=0; i<centers.size(); i++)
    {
        visited.emplace_back(false);
    }
    for (int i=0; i<ignores.size(); i++)
    {
        visited[ignores[i]] = true;
    }


    std::vector<std::vector<int>> groups;
    std::vector<int> tile2group((int)centers.size());
    for (int i=0; i<centers.size(); i++)
    {
        tile2group[i] = -1;
    }
    std::vector<int> group_labels;
    tqdm bar2;
    for (int i=0; i<centers.size(); i++)
    {
        bar1.progress(i, centers.size());

        if (visited[i]) continue;

        std::vector<int> group_tiles;
        int denseIdx = SparseToDense[i];
        int label = tile_labels[i];
        // std::queue<int> Q;
        // Q.push(denseIdx);
        DFS_search(denseIdx, label, tile_shape, IndexMap, tile_labels, visited, group_tiles);

        if (group_tiles.size() == 0){
            std::cout << "tile " << i << " error" << std::endl;
            continue;
        }

        for (int j=0; j<group_tiles.size(); j++)
        {
            tile2group[group_tiles[j]] = (int)groups.size();
        }
        groups.emplace_back(group_tiles);
        group_labels.emplace_back(label);
    }
    bar2.finish();

    // std::cout << "DEBUG " << groups[0].size() << std::endl;

    for (int i=0; i<groups.size(); i++)
    {
        if (groups[i].size() >= min_tiles_per_group) continue;
        int ori_num = groups[i].size();
        merge_neighbor_group(groups[i], i, tile_shape,
                             IndexMap, SparseToDense, tile2group, groups);
        if (groups[i].size() != 0)
        {
            std::cout << "Error Merge " << i << " " << groups[i].size() << std::endl;
        }else{
            std::cout << "Merge group " << i << " from " << ori_num << " to " << 0 << std::endl;
        }
        // std::cout << "DEBUG " << groups[0].size() << std::endl;
    }


    for (int i=0; i<groups.size(); i++)
    {
        if (groups[i].size() == 0) continue;

        if (groups[i].size() < min_tiles_per_group)
        {
            std::cout << i << " exists small groups " << groups[i].size() << std::endl;
        }
        
        group_start.emplace_back((int)tiles_list.size());
        tiles_list.insert(tiles_list.end(), groups[i].begin(), groups[i].end());
    }

    return tile_labels;
}

PyObject* Pygroup_tiles(PyObject* self, PyObject* args)
{
    char *s1,*s2;
    char *byte_centers, *byte_IndexMap, *byte_SparseToDense, *byte_ignores;
    float3 scene_min_corner;
    int size_centers, size_IndexMap, size_SparseToDense, size_ignores;
    float tile_size;
    int3 tile_shape;

    if (!PyArg_ParseTuple(args, "sss#s#s#s#ffffiiiiii", 
        &s1, &s2, &byte_centers, &size_centers,
        &byte_IndexMap, &size_IndexMap,
        &byte_SparseToDense, &size_SparseToDense,
        &byte_ignores, &size_ignores,
        &scene_min_corner.x, &scene_min_corner.y, &scene_min_corner.z,
        &tile_size,
        &tile_shape.x, &tile_shape.y, &tile_shape.z, 
        &max_tiles_per_group, &min_tiles_per_group, &mean_tiles_per_group))
    {
        return NULL;
    }

    std::string model_path(s1);
    std::string label_path(s2);
    std::vector<float3> centers((float3*)byte_centers, (float3*)byte_centers + size_centers/12);
    std::vector<int> IndexMap((int*)byte_IndexMap, (int*)byte_IndexMap + size_IndexMap/4);
    std::vector<int> SparseToDense((int*)byte_SparseToDense, (int*)byte_SparseToDense + size_SparseToDense/4);
    std::vector<int> ignores((int*)byte_ignores, (int*)byte_ignores + size_ignores/4);


    std::vector<int> tiles_list;
    std::vector<int> group_start;
    std::vector<int> tile_labels = group_tiles(model_path, label_path, centers, IndexMap, SparseToDense, ignores,
                                               scene_min_corner, tile_size, tile_shape,
                                               tiles_list, group_start);

    PyObject* Result = Py_BuildValue("y#y#y#",
                        (char*)&tiles_list[0], tiles_list.size()*sizeof(int),
                        (char*)&group_start[0], group_start.size()*sizeof(int),
                        (char*)&tile_labels[0], tile_labels.size()*sizeof(int));
    return Result;

}

static PyMethodDef gf_methods[] = {
    {"group_tiles", Pygroup_tiles, METH_VARARGS},
    {NULL,NULL}
};

static struct PyModuleDef callModuleDef = {
    PyModuleDef_HEAD_INIT,
    "group_tiles",
    "",
    -1,
    gf_methods
};

extern "C"

PyMODINIT_FUNC PyInit_group_tiles(void){
    PyModule_Create(&callModuleDef);
}
