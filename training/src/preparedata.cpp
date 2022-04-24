#include "cuda_func.h"
#include "plyIO.h"
#define PY_SSIZE_T_CLEAN

#include <cnpy.h>
#include <Python.h>
#include <time.h>
#include <algorithm>

bool debug = false;

cv::Vec3b getColorSubpix(const cv::Mat& img, cv::Point2f pt)
{
    cv::Mat patch;
    cv::getRectSubPix(img, cv::Size(1,1), pt, patch);
    return patch.at<cv::Vec3b>(0,0);
}


std::vector<float3> sample_point_uniform(int num_side, float3 tile_center, float tile_size)
{
    std::vector<float3> samples;
    float step = tile_size / num_side;

    float3 min_corner = tile_center - tile_size / 2.0f + step / 2.0f;

    for (int x=0; x<num_side; x++)
    {
        for (int y=0; y<num_side; y++)
        {
            for (int z=0; z<num_side; z++)
            {
                float3 p = min_corner + make_float3(x*step, y*step, z*step);
                samples.emplace_back(p);
            }
        }
    }
    return samples;
}

void gen_init_data(
    std::vector<float> data,
    std::vector<int> imgIdxs,
    const std::vector<int> BConFaceIdx,
    const std::vector<int2> BConFaceNum,
    float* Ks,
    float* C2Ws,
    float3 tile_center,
    float tile_size,
    std::vector<float> &initData)
{

   assert( data.size() % 11 == 0 );

   for (int i=0; i<data.size(); i+=11)
   {
       int idx = 11 * i;
       float3 rays_o = make_float3(data[idx], data[idx+1], data[idx+2]);
       float3 rays_d = make_float3(data[idx+3], data[idx+4], data[idx+5]);

   }
}


void gen_init_data(
    std::string img_path,
    std::string dep_path,
    std::vector<int> imgIdxs, 
    float* Ks,
    float* C2Ws,
    int sample_num,
    float3 tile_center,
    float tile_size,
    int height, int width,
    std::vector<float> &initData)
{
    std::vector<float3> samples = sample_point_uniform(sample_num, tile_center, tile_size);

    float3* sum_colors = new float3[(int)samples.size()];
    int* count = new int[(int)samples.size()];
    for (int i=0; i<samples.size(); i++)
    {
        sum_colors[i] = make_float3(0.0f, 0.0f, 0.0f);
        count[i] = 0;
    }

    float* m = new float[12];

    tqdm bar;
    for (int i=0; i<imgIdxs.size(); i++)
    {
        bar.progress(i, imgIdxs.size());
        cv::Mat img = cv::imread(img_path + "/" + std::to_string(imgIdxs[i]) + ".png");
        cnpy::NpyArray dep = cnpy::npy_load(dep_path + "/" + std::to_string(imgIdxs[i]) + ".npy");
        float* depth_data = dep.data<float>();

        float* K = Ks + imgIdxs[i] * 9;
        float* C2W = C2Ws + imgIdxs[i] * 12;

        compute_project_matrix(K, C2W, m);

        for (int j=0; j<samples.size(); j++)
        {
            float3 p = samples[j];
            float x = m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3];
            float y = m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7];
            float z = m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11];

            if (z <= 0) continue;

            float px = x / z;
            float py = y / z;

            if (px < 0 || px >= width || py < 0 || py >= height) continue;

            float d = depth_data[(int)py * width + (int)px];
            if (abs(z - d) > 0.01f) continue;
            
            cv::Point2f pixel(px, py);
            cv::Vec3b color = getColorSubpix(img, pixel);
            sum_colors[j] = sum_colors[j] + make_float3((float)color[2],(float)color[1],(float)color[0])/255.0f;
            count[j] += 1;
        }
    }
    bar.finish();

    for (int i=0; i<samples.size(); i++)
    {
        float3 p = samples[i];

        float alpha = 0.0f;
        float3 c = make_float3(0.0f, 0.0f, 0.0f);

        if (count[i] != 0)
        {
            c = sum_colors[i] / count[i];
            alpha = 0.9f;
        }

        initData.emplace_back(p.x);
        initData.emplace_back(p.y);
        initData.emplace_back(p.z);
        initData.emplace_back(c.x);
        initData.emplace_back(c.y);
        initData.emplace_back(c.z);
        initData.emplace_back(alpha);

    }

    delete [] sum_colors;
    delete [] count;
    delete [] m;
}

void gen_init_data(
    std::vector<float3> vertices,
    std::vector<int3> faces,
    std::vector<float3> colors,
    const std::vector<int> BConFaceIdx,
    const std::vector<int2> BConFaceNum,
    int num_block,
    int trainTileIdx,
    int sample_num,
    float3 tile_center,
    float tile_size,
    float voxel_size,
    std::vector<float> &initData)
{
    std::vector<float3> samples = sample_point_uniform(sample_num, tile_center, tile_size);

    float3 min_corner = tile_center - tile_size / 2.0f;
    float block_size = tile_size / num_block;

    int base_idx = trainTileIdx * num_block * num_block * num_block;

    for (int i=0; i<samples.size(); i++)
    {
        float3 p = samples[i];
        int3 bidx = make_int3((p - min_corner) / block_size);
        int block_idx = base_idx + bidx.z + bidx.y * num_block + bidx.x * num_block * num_block;
        int2 block_info = BConFaceNum[block_idx];
        int start = block_info.x;
        int num_face = block_info.y;

        // std::cout << p.x << " " << p.y << " " << p.z << std::endl;
        // std::cout << start << " " << num_face << std::endl;

        float alpha = 0.0f;
        float3 c = make_float3(0.0f, 0.0f, 0.0f);

        float min_sum = 100000000.0f;
        int face_idx = -1;

        for (int j=start; j<start+num_face; j++)
        {
            int3 vidx = faces[BConFaceIdx[j]];
            float3 A = vertices[vidx.x];
            float3 B = vertices[vidx.y];
            float3 C = vertices[vidx.z];
            float3 bweight = barycentric_weight(A,B,C,p);
            float sum = bweight.x + bweight.y + bweight.z;
            // std::cout << i << " " << j << " " << sum << std::endl;
            if (sum < min_sum)
            {
                face_idx = BConFaceIdx[j];
                min_sum = sum;
            }
            // if (sum - 1 <= voxel_size)
            // {
            //     alpha = 0.9f;
            //     float3 CA = colors[vidx.x];
            //     float3 CB = colors[vidx.y];
            //     float3 CC = colors[vidx.z];
            //     c = bweight.x * CA + bweight.y * CB + bweight.z * CC;
            //     break;
            // }
        }

        if (min_sum - 1 <= voxel_size)
        {
            int3 vidx = faces[face_idx];
            float3 A = vertices[vidx.x];
            float3 B = vertices[vidx.y];
            float3 C = vertices[vidx.z];
            float3 bweight = barycentric_weight(A,B,C,p);
            alpha = 0.9f;
            float3 CA = colors[vidx.x];
            float3 CB = colors[vidx.y];
            float3 CC = colors[vidx.z];
            c = bweight.x * CA + bweight.y * CB + bweight.z * CC;
        }

        c = c / 255.0f;

        initData.emplace_back(p.x);
        initData.emplace_back(p.y);
        initData.emplace_back(p.z);
        initData.emplace_back(c.x);
        initData.emplace_back(c.y);
        initData.emplace_back(c.z);
        initData.emplace_back(alpha);
    }
}

void init_nodes_flag(
    std::vector<float3> vertices,
    std::vector<int3> faces,
    int num_voxel,
    float voxel_size, 
    float tile_size, 
    float3 tile_center, 
    float dilate_size, 
    short* nodes_flag)
{ 

    // printf("init nodes by geometry\n");

    float3 min_corner = tile_center - tile_size / 2.0f;
    float3 max_corner = tile_center + tile_size / 2.0f;

    for (int i=0; i<faces.size(); i++)
    {
        int3 vidx = faces[i];
        float3 A = vertices[vidx.x];
        float3 B = vertices[vidx.y];
        float3 C = vertices[vidx.z];
        float3 min_c = fminf(fminf(A, B), C);
        float3 max_c = fmaxf(fmaxf(A, B), C);

        // -----------------------
        float3 tri_center = (min_c + max_c) / 2.0f;
        float3 tri_size = (max_c - min_c) + 2.0f * dilate_size; // 涨dilate 
        float3 half_trisize = tri_size / 2.0f;
        min_c = tri_center - half_trisize;
        max_c = tri_center + half_trisize;
        // -------------------------
        if (max_c.x <= min_corner.x || max_c.y <= min_corner.y || max_c.z <= min_corner.z ||
            min_c.x >= max_corner.x || min_c.y >= max_corner.y || min_c.z >= max_corner.z ) continue;

        min_c = min_c - min_corner;
        max_c = max_c - min_corner;

        int3 min_idx = make_int3(min_c / voxel_size);
        int3 max_idx = make_int3(max_c / voxel_size);

        min_idx = clamp(min_idx, 0, num_voxel-1);
        max_idx = clamp(max_idx, 0, num_voxel-1);

        for (int x=min_idx.x; x<=max_idx.x; x++)
        {
            for (int y=min_idx.y; y<=max_idx.y; y++)
            {
                for (int z=min_idx.z; z<=max_idx.z; z++)
                {
                    int voxel_idx = x * num_voxel * num_voxel + y * num_voxel + z;
                    // std::cout << voxel_idx << std::endl;
                    nodes_flag[voxel_idx] = 1;
                }
            }
        }

    }   
    // printf("Finished init nodes flag\n");
}

void preparedata(
    std::string model_path,
    std::string texture_path,
    std::string img_path,
    const std::vector<float3> centers,
    const std::vector<int> IndexMap,
    const std::vector<int> BConFaceIdx,
    const std::vector<int2> BConFaceNum,
    const int3 tile_shape,
    const float3 scene_min_corner,
    int trainTileIdx,
    int max_tracingtile,
    float* Ks,
    float* C2Ws,
    int num_camera,
    float tile_size,
    int num_block,
    int height, int width,
    std::vector<int> imgIdxs, // 当前tile包含的图片
    std::vector<float3> &data) 
{

    std::vector<float3> vertices;
    std::vector<int3> faces;
    std::vector<float2> uv_array;
    read_plyFile(model_path, vertices, faces, uv_array);

    std::vector<float3> colors;
    if (uv_array.size() != 0 && texture_path != "")
    {
        cv::Mat texture = cv::imread(texture_path);
        cv::Size s = texture.size();
        printf("Texture height %d width %d\n", s.height, s.width);
        for (int i=0; i<uv_array.size(); i++)
        {
            float2 uv = uv_array[i];
            float x = uv.x * s.width;
            float y = s.height - uv.y * s.height;
            cv::Point2f p(x, y);
            cv::Vec3b color = getColorSubpix(texture, p);
            colors.emplace_back(make_float3(color[2],color[1],color[0])/255.0f);
        }
    }else{
        for (int i=0; i<vertices.size(); i++)
        {
            colors.emplace_back(make_float3(0,0,0));
        }
    }
    uv_array.clear();
    get_trainData(trainTileIdx, img_path, vertices, faces, colors, centers, IndexMap, BConFaceIdx,
                  BConFaceNum, imgIdxs, Ks, C2Ws, num_camera, tile_shape, 
                  scene_min_corner, tile_size, max_tracingtile, num_block, height, width, debug, data);


}

void preparedata_v2(
    std::string model_path,
    std::string texture_path,
    std::string img_path,
    std::string diffuse_path,
    const std::vector<float3> centers,
    const std::vector<int> IndexMap,
    const std::vector<int> BConFaceIdx,
    const std::vector<int2> BConFaceNum,
    const int3 tile_shape,
    const float3 scene_min_corner,
    int trainTileIdx,
    int max_tracingtile,
    float* Ks,
    float* C2Ws,
    int num_camera,
    float tile_size,
    int num_block,
    int height, int width,
    std::vector<int> imgIdxs, 
    std::vector<float> &data) 
{


    std::vector<float3> vertices;
    std::vector<int3> faces;
    std::vector<float2> uv_array;
    read_plyFile(model_path, vertices, faces, uv_array);

    std::vector<float3> colors;
    if (uv_array.size() != 0 && texture_path != "")
    {
        cv::Mat texture = cv::imread(texture_path);
        cv::Size s = texture.size();
        printf("Texture height %d width %d\n", s.height, s.width);
        for (int i=0; i<uv_array.size(); i++)
        {
            float2 uv = uv_array[i];
            float x = uv.x * s.width;
            float y = s.height - uv.y * s.height;
            cv::Point2f p(x, y);
            cv::Vec3b color = getColorSubpix(texture, p);
            colors.emplace_back(make_float3(color[2],color[1],color[0])/255.0f);
        }
    }else{
        for (int i=0; i<vertices.size(); i++)
        {
            colors.emplace_back(make_float3(0,0,0));
        }
    }
    uv_array.clear();

    get_trainData_v2(trainTileIdx, img_path, diffuse_path, vertices, faces, colors, centers, IndexMap, BConFaceIdx,
                  BConFaceNum, imgIdxs, Ks, C2Ws, num_camera, tile_shape, 
                  scene_min_corner, tile_size, max_tracingtile, num_block, height, width, debug, data);
}


void load_colors(std::string path, std::vector<float3> &colors)
{
    cnpy::NpyArray arr = cnpy::npy_load(path);
    float* data = arr.data<float>();
    colors.insert(colors.end(), (float3*)data, (float3*)data + arr.shape[0]);
}


template<typename T> 
static std::vector<int> argsort(const std::vector<T>& array)
{
	const int array_len(array.size());
	std::vector<int> array_index(array_len, 0);
	for (int i = 0; i < array_len; ++i)
		array_index[i] = i;

	std::sort(array_index.begin(), array_index.end(),
		[&array](int pos1, int pos2) {return (array[pos1] > array[pos2]);});

	return array_index;
}

std::vector<int> filter_visImg_v3(std::vector<int> imgIdx, float3 tile_center, float* C2Ws, int need)
{
    if (imgIdx.size() <= need) return imgIdx;

    std::vector<float> dis_list;
    for (int i=0; i<imgIdx.size(); i++)
    {
        int idx = imgIdx[i];
        float* C2W = C2Ws + idx * 12;
        float3 camera_center = make_float3(C2W[3],C2W[7],C2W[11]);
        float3 vec = tile_center - camera_center;
        float dis = norm(vec);
        dis_list.emplace_back(dis);
    }

    std::vector<int> output;
    std::vector<int> array_index = argsort(dis_list);
    for (int i=array_index.size()-1; i>=0; i--)
    {
        if (output.size() >= need) break;
        output.emplace_back(imgIdx[array_index[i]]);
    }
    return output;
}

std::vector<int> filter_visImg_v2(std::vector<int> imgIdx, float3 tile_center, float* C2Ws, int need)
{
    if (imgIdx.size() <= need) return imgIdx;


    float3 z_axis = make_float3(0,0,1);
    std::vector<float> sim_list;
    for (int i=0; i<imgIdx.size(); i++)
    {
        int idx = imgIdx[i];
        float* C2W = C2Ws + idx * 12;
        float3 camera_center = make_float3(C2W[3],C2W[7],C2W[11]);
        float3 vec = tile_center - camera_center;
        vec = vec / norm(vec);
        float sim = dot(z_axis, vec);
        sim_list.emplace_back(sim);
    }

    std::vector<int> output;
    // std::cout << "1" << std::endl;
    std::vector<int> array_index = argsort(sim_list);
    for (int i=0; i<imgIdx.size(); i++)
    {
        // std::cout << i << " " << array_index[i] << std::endl;
        output.emplace_back(imgIdx[array_index[i]]);
    }

    std::vector<int> angle_order;
    std::vector<int> add_idx = {0};
    // std::cout << "2" << std::endl;

    int step = 1, count = 1;
    int len = imgIdx.size() - 1;

    while(true)
    {
        for (int i=1; i<=step; i+=2)
        {
            add_idx.emplace_back((int)(1.0f * i / step * len));
            count++;
            if (count > len) break;
        }
        if (count > len) break;
        step *= 2;
    }

    // std::cout << "3" << std::endl;

    for (int i=0; i<add_idx.size(); i++)
    {
        if (angle_order.size() >= need) break;
        angle_order.emplace_back(output[add_idx[i]]);
    }
    // std::cout << "4" << std::endl;
    return angle_order;
}

std::vector<int> filter_visImg(
    std::vector<int> imgIdx, float3 tile_center, float* C2Ws, float dis_thesh, int need)
{
    if (imgIdx.size() <= need) return imgIdx;

    std::vector<int> output;

    for (int i=0; i<imgIdx.size(); i++)
    {
        int idx = imgIdx[i];
        float* C2W = C2Ws + idx * 12;
        float3 camera_center = make_float3(C2W[3],C2W[7],C2W[11]);
        float3 vec = tile_center - camera_center;
        float dis = norm(vec);
        if (dis > dis_thesh) continue;
        output.emplace_back(idx);
    }

    if (output.size() > need)
    {
        random_shuffle(output.begin(), output.end());
        std::vector<int> new_out;
        new_out.insert(new_out.end(), output.begin(), output.begin()+need);
        return new_out;
    }else{
        return output;
    }

}

std::vector<int> filter_visImg(
    std::vector<int> imgIdx, int need)
{
    if (imgIdx.size() <= need) return imgIdx;
    else{
        random_shuffle(imgIdx.begin(), imgIdx.end());
        std::vector<int> new_out;
        new_out.insert(new_out.end(), imgIdx.begin(), imgIdx.begin()+need);
        return new_out;
    }
}



void preparedata_patch(
    std::string model_path,
    std::string dep_path,
    std::string img_path,
    const std::vector<int> predefine_imgIdxs,
    const std::vector<int> tileIdxs,
    const std::vector<int> VisImg,
    const std::vector<int> ignore,
    const std::vector<float3> centers,
    const std::vector<int> IndexMap,
    const std::vector<int> BConFaceIdx,
    const std::vector<int2> BConFaceNum,
    const int3 tile_shape,
    const float3 scene_min_corner,
    int max_tracingtile,
    float* Ks,
    float* C2Ws,
    int num_camera,
    float tile_size,
    int num_block,
    int height, int width,
    int patch_size,
    int num_voxel,
    float voxel_size, 
    float dilate_size,
    float dis_thesh,
    int need,
    std::vector<int> &tiles_flag, 
    std::vector<short> &nodes_flag,
    std::vector<float> &data) 
{
    std::cout << "Load model" << std::endl;
    std::vector<float3> vertices;
    std::vector<int3> faces;
    std::vector<float2> uv_array;
    read_plyFile(model_path, vertices, faces, uv_array);
    uv_array.clear();

    std::cout << "\n====== Generate data =======\n";

    for (int i=0; i<tileIdxs.size(); i++)
    {
        int trainTileIdx = tileIdxs[i];
        std::cout << i << "/" << tileIdxs.size() << " Prepare data of tile " << trainTileIdx << std::endl;
        std::vector<int> raw_imgIdxs;
        const int* tile_visImg = VisImg.data() + trainTileIdx * num_camera;
        for (int j=0; j<num_camera; j++)
        {
            if (tile_visImg[j] == 1 && ignore[j] == 0)
            {
                raw_imgIdxs.emplace_back(j);
            } 
        }

        std::cout << "Num image: " << raw_imgIdxs.size() << std::endl;

        std::vector<int> imgIdxs = filter_visImg_v3(raw_imgIdxs, centers[trainTileIdx], C2Ws, need);

        std::cout << "Num image after selection: " << imgIdxs.size() << std::endl;

        for (int j=0; j<predefine_imgIdxs.size(); j++)
        {
            if (ignore[predefine_imgIdxs[j]] == 0)
            {
                imgIdxs.emplace_back(predefine_imgIdxs[j]);
            }
        }

        std::cout << "Append predefined images: " << imgIdxs.size() << std::endl;

        long before = data.size();

        if (imgIdxs.size() > 0)
        {
            get_trainData_v3(trainTileIdx, img_path, dep_path, vertices, faces, centers, IndexMap, BConFaceIdx,
                            BConFaceNum, imgIdxs, Ks, C2Ws, num_camera, tile_shape, 
                            scene_min_corner, tile_size, max_tracingtile, num_block, height, width, patch_size, debug, data);
        }

        long after = data.size();

        int num = (after - before) / (patch_size * patch_size * 14);
        for (int j=0; j<num; j++)
        {
            tiles_flag.emplace_back(i);
        }
        
        init_nodes_flag(vertices, faces, num_voxel, voxel_size, tile_size, centers[trainTileIdx], 
                        dilate_size, nodes_flag.data()+i*num_voxel*num_voxel*num_voxel);
    }
    
}

void preparedata_patch_sec(
    std::string diffuse_path,
    std::string img_path,
    const std::vector<int> predefine_imgIdxs,
    const std::vector<int> tileIdxs,
    const std::vector<int> VisImg,
    const std::vector<int> ignore,
    const std::vector<float3> centers,
    const std::vector<int> IndexMap,
    const std::vector<int> SparseToGroup,
    float* voxels, 
    short* nodes, 
    const float sample_step,
    const int3 tile_shape,
    const float3 scene_min_corner,
    int max_tracingtile,
    float* Ks,
    float* C2Ws,
    int num_camera,
    float tile_size,
    int height, int width,
    int patch_size,
    int num_voxel,
    float voxel_size, 
    float dis_thesh,
    int need,
    std::vector<int> &tiles_flag, 
    std::vector<float> &data)
{
    int num_render_tiles = (int)tileIdxs.size();

    std::vector<int> raw_imgIdxs;
    for (int i=0; i<tileIdxs.size(); i++)
    {
        int trainTileIdx = tileIdxs[i];
        std::cout << i << "/" << tileIdxs.size() << " 生成 tile " << trainTileIdx << " 数据" << std::endl;
        const int* tile_visImg = VisImg.data() + trainTileIdx * num_camera;
        for (int j=0; j<num_camera; j++)
        {
            if (tile_visImg[j] == 1 && ignore[j] == 0)
            {
                bool exist = false;
                for (int k=0; k<raw_imgIdxs.size(); k++)
                {
                    if(raw_imgIdxs[k] == j)
                    {
                        exist = true;
                        break;
                    }
                }
                if (!exist) raw_imgIdxs.emplace_back(j);
            } 
        }

    }

    std::vector<int> imgIdxs = filter_visImg(raw_imgIdxs, need);

    imgIdxs.insert(imgIdxs.end(), predefine_imgIdxs.begin(), predefine_imgIdxs.end());

    // imgIdxs.clean();
    // imgIdxs.emplace_back(400, 401, 402, 403, 1225, 1226, 1227, 1250, 2122, 2123, 2136, 2137, 2138, 2139, 2154,
    // 2155, 2156, 21257, 2182, 2183, 2184, 2195, 2196, 2197, 2198, );

    if (imgIdxs.size() > 0)
    {
        get_trainData_v4(0, img_path, diffuse_path, centers, IndexMap, imgIdxs, 
                        SparseToGroup, num_render_tiles, voxels, nodes, 
                        Ks, C2Ws, num_camera, tile_shape, 
                        scene_min_corner, tile_size, num_voxel, voxel_size, sample_step,
                        max_tracingtile, height, width, patch_size, debug, data);
    }

}





PyObject* Pypreparedata(PyObject* self, PyObject* args)
{
    char *s1, *s2,*s3;
    char *byte_imgIdxs, *byte_centers, *byte_IndexMap, *byte_BConFaceIdx, *byte_BConFaceNum;
    float tile_size;
    int max_tracingtile;
    char *byte_Ks, *byte_C2Ws;
    int height, width, num_camera, trainTileIdx, num_block;
    int size_K, size_C2W, size_imgIdx, size_centers, size_IndexMap, size_BConFaceIdx, size_BConFaceNum;
    int3 tile_shape;
    float3 scene_min_corner;
    if (!PyArg_ParseTuple(args, "sssfiiiiis#s#s#s#s#s#s#iiifffp",
       &s1, &s2, &s3, &tile_size,
       &num_block, &height, &width, &trainTileIdx, &max_tracingtile,
       &byte_Ks, &size_K, &byte_C2Ws, &size_C2W,
       &byte_imgIdxs, &size_imgIdx,
       &byte_centers, &size_centers,
       &byte_IndexMap, &size_IndexMap,
       &byte_BConFaceIdx, &size_BConFaceIdx,
       &byte_BConFaceNum, &size_BConFaceNum,
       &tile_shape.x, &tile_shape.y, &tile_shape.z,
       &scene_min_corner.x, &scene_min_corner.y, &scene_min_corner.z, &debug))
    {
        return NULL;
    }
    std::string model_path(s1);
    std::string texture_path(s2);
    std::string img_path(s3);
    float* Ks = (float*)byte_Ks;
    float* C2Ws = (float*)byte_C2Ws;
    num_camera = size_K / 36;


    std::vector<int> imgIdxs((int*)byte_imgIdxs, (int*)byte_imgIdxs+size_imgIdx/4);
    std::vector<float3> centers((float3*)byte_centers, (float3*)byte_centers+size_centers/12);
    std::vector<int> IndexMap((int*)byte_IndexMap, (int*)byte_IndexMap+size_IndexMap/4);
    std::vector<int> BConFaceIdx((int*)byte_BConFaceIdx, (int*)byte_BConFaceIdx+size_BConFaceIdx/4);
    std::vector<int2> BConFaceNum((int2*)byte_BConFaceNum, (int2*)byte_BConFaceNum+size_BConFaceNum/8);

    std::vector<float3> data;
    preparedata( model_path, texture_path, img_path, centers, IndexMap,
                 BConFaceIdx, BConFaceNum, tile_shape, scene_min_corner, trainTileIdx,
                 max_tracingtile, Ks, C2Ws, num_camera, tile_size, num_block, height, width,
                 imgIdxs, data);
    PyObject* Result = Py_BuildValue("y#",
                        (char*)&data[0], data.size()*sizeof(float3));
    return Result;
}


PyObject* Pypreparedata_v2(PyObject* self, PyObject* args)
{
    char *s1, *s2,*s3, *s4;
    char *byte_imgIdxs, *byte_centers, *byte_IndexMap, *byte_BConFaceIdx, *byte_BConFaceNum;
    float tile_size;
    int max_tracingtile;
    char *byte_Ks, *byte_C2Ws;
    int height, width, num_camera, trainTileIdx, num_block;
    int size_K, size_C2W, size_imgIdx, size_centers, size_IndexMap, size_BConFaceIdx, size_BConFaceNum;
    int3 tile_shape;
    float3 scene_min_corner;
    if (!PyArg_ParseTuple(args, "ssssfiiiiis#s#s#s#s#s#s#iiifffp",
       &s1, &s2, &s3, &s4,&tile_size,
       &num_block, &height, &width, &trainTileIdx, &max_tracingtile,
       &byte_Ks, &size_K, &byte_C2Ws, &size_C2W,
       &byte_imgIdxs, &size_imgIdx,
       &byte_centers, &size_centers,
       &byte_IndexMap, &size_IndexMap,
       &byte_BConFaceIdx, &size_BConFaceIdx,
       &byte_BConFaceNum, &size_BConFaceNum,
       &tile_shape.x, &tile_shape.y, &tile_shape.z,
       &scene_min_corner.x, &scene_min_corner.y, &scene_min_corner.z, &debug))
    {
        return NULL;
    }
    std::string model_path(s1);
    std::string texture_path(s2);
    std::string img_path(s3);
    std::string diffuse_path(s4);
    float* Ks = (float*)byte_Ks;
    float* C2Ws = (float*)byte_C2Ws;
    num_camera = size_K / 36;

    std::vector<int> imgIdxs((int*)byte_imgIdxs, (int*)byte_imgIdxs+size_imgIdx/4);
    std::vector<float3> centers((float3*)byte_centers, (float3*)byte_centers+size_centers/12);
    std::vector<int> IndexMap((int*)byte_IndexMap, (int*)byte_IndexMap+size_IndexMap/4);
    std::vector<int> BConFaceIdx((int*)byte_BConFaceIdx, (int*)byte_BConFaceIdx+size_BConFaceIdx/4);
    std::vector<int2> BConFaceNum((int2*)byte_BConFaceNum, (int2*)byte_BConFaceNum+size_BConFaceNum/8);

    std::vector<float> data;
    preparedata_v2( model_path, texture_path, img_path, diffuse_path, centers, IndexMap,
                 BConFaceIdx, BConFaceNum, tile_shape, scene_min_corner, trainTileIdx,
                 max_tracingtile, Ks, C2Ws, num_camera, tile_size, num_block, height, width,
                 imgIdxs, data);
               
    PyObject* Result = Py_BuildValue("y#",
                        (char*)&data[0], data.size()*sizeof(float));
    return Result;
}

PyObject* Pypreparedata_patch(PyObject* self, PyObject* args)
{
    char *s1,*s3, *s5;
    char *byte_centers, *byte_IndexMap, *byte_BConFaceIdx, *byte_BConFaceNum;
    int max_tracingtile;
    char *byte_predefine;
    char *byte_Ks, *byte_C2Ws;
    char *byte_tileIdxs, *byte_VisImg, *byte_ignore; 
    int size_tileIdxs, size_VisImg, size_ignore;
    int height, width, num_camera, num_block, patch_size;
    int size_K, size_C2W, size_imgIdx, size_centers, size_IndexMap, size_BConFaceIdx, size_predefine;
    long size_BConFaceNum;
    int num_voxel;
    float voxel_size, dilate_size;
    float dis_thresh;
    int need;
    bool sec_itr;
    // bool init_voxel;
    int3 tile_shape;
    float3 scene_min_corner;
    if (!PyArg_ParseTuple(args, "ssss#s#s#s#s#s#s#s#s#s#iiiiiiiifffifffip",
       &s1, &s3, &s5,
       &byte_tileIdxs, &size_tileIdxs, 
       &byte_VisImg, &size_VisImg, &byte_ignore, &size_ignore,
       &byte_Ks, &size_K, &byte_C2Ws, &size_C2W,
       &byte_centers, &size_centers,
       &byte_IndexMap, &size_IndexMap,
       &byte_BConFaceIdx, &size_BConFaceIdx,
       &byte_BConFaceNum, &size_BConFaceNum,
       &byte_predefine, &size_predefine,
       &height, &width, &num_block, &patch_size, &max_tracingtile, 
       &tile_shape.x, &tile_shape.y, &tile_shape.z,
       &scene_min_corner.x, &scene_min_corner.y, &scene_min_corner.z, 
       &num_voxel, &voxel_size, &dilate_size, &dis_thresh, &need, &debug))
    {
        return NULL;
    }
    
    std::string model_path(s1);
    std::string img_path(s3);
    std::string dep_path(s5);
    float* Ks = (float*)byte_Ks;
    float* C2Ws = (float*)byte_C2Ws;
    num_camera = size_K / 36;

    float tile_size = num_voxel * voxel_size;

    std::cout << "\n============= Prepare training data =============" <<std::endl;
    std::cout << "Num Block: " << num_block << std::endl;
    std::cout << "Patch Size: " << patch_size << std::endl; 
    std::cout << "Mesh path：" << model_path << std::endl;
    std::cout << "depth path: " << dep_path << std::endl;
    std::cout << "image path：" << img_path << std::endl;
    std::cout << "num camera " << num_camera << std::endl;
    std::cout << "image size: H " << height << " W " << width << std::endl;
    std::cout << "voxel size: " << voxel_size << std::endl;
    std::cout << "num voxel: " << num_voxel << std::endl;
    std::cout << "num tiles: " << size_tileIdxs/4 << std::endl;
    std::cout << "max_tracingtile: " << max_tracingtile << std::endl;
    std::cout << "tile shape: " << tile_shape.x << " " << tile_shape.y << " " << tile_shape.z << std::endl;
    std::cout << "============= Prepare training data =============\n" <<std::endl;
;
    std::vector<float3> centers((float3*)byte_centers, (float3*)byte_centers+size_centers/12);;
    std::vector<int> IndexMap((int*)byte_IndexMap, (int*)byte_IndexMap+size_IndexMap/4);
    std::vector<int> BConFaceIdx((int*)byte_BConFaceIdx, (int*)byte_BConFaceIdx+size_BConFaceIdx/4);
    std::vector<int2> BConFaceNum((int2*)byte_BConFaceNum, (int2*)byte_BConFaceNum+size_BConFaceNum/8);
    std::vector<int> tileIdxs((int*)byte_tileIdxs, (int*)byte_tileIdxs+size_tileIdxs/4);
    std::vector<int> VisImg((int*)byte_VisImg, (int*)byte_VisImg + size_VisImg/4);
    std::vector<int> ignore((int*)byte_ignore, (int*)byte_ignore + size_ignore/4);
    std::vector<int> predefine_imgIdxs((int*)byte_predefine, (int*)byte_predefine + size_predefine/4);

    std::vector<float> data;
    std::vector<short> nodes_flag;
    for (int i=0; i<(int)tileIdxs.size() * num_voxel*num_voxel*num_voxel; i++)
    {
        nodes_flag.emplace_back(-1);
    }
    std::vector<int> tiles_flag; 


    std::cout << "first iteration" << std::endl;
    preparedata_patch( model_path, dep_path, img_path, predefine_imgIdxs,
                    tileIdxs, VisImg, ignore, centers, IndexMap,
                    BConFaceIdx, BConFaceNum, tile_shape, scene_min_corner,
                    max_tracingtile, Ks, C2Ws, num_camera, tile_size, num_block, height, width,
                    patch_size, num_voxel, voxel_size, dilate_size, dis_thresh, need, tiles_flag, nodes_flag, data);


    PyObject* Result = Py_BuildValue("y#y#y#",
                        (char*)&data[0], data.size()*sizeof(float),
                        (char*)&tiles_flag[0], tiles_flag.size()*sizeof(int),
                        (char*)&nodes_flag[0], nodes_flag.size()*sizeof(short));
    return Result;
}

PyObject* Pypreparedata_patch_sec(PyObject* self, PyObject* args)
{
    char *s3, *s5;
    char *byte_centers, *byte_IndexMap;
    char *byte_predefine;
    char *byte_voxels, *byte_nodes, *byte_SparseToGroup;
    int size_voxels, size_nodes, size_SparseToGroup, size_predefine;
    float sample_step;
    int max_tracingtile;
    char *byte_Ks, *byte_C2Ws;
    char *byte_tileIdxs, *byte_VisImg, *byte_ignore; 
    int size_tileIdxs, size_VisImg, size_ignore;
    int height, width, num_camera, num_block, patch_size;
    int size_K, size_C2W, size_imgIdx, size_centers, size_IndexMap;
    int num_voxel;
    float voxel_size;
    float dis_thresh;
    int need;
    // bool init_voxel;
    int3 tile_shape;
    float3 scene_min_corner;
    if (!PyArg_ParseTuple(args, "sss#s#s#s#s#s#s#s#s#s#s#iiiiiiifffifffip",
       &s3, &s5,
       &byte_voxels, &size_voxels, &byte_nodes, &size_nodes, &byte_SparseToGroup, &size_SparseToGroup,
       &byte_tileIdxs, &size_tileIdxs, 
       &byte_VisImg, &size_VisImg, &byte_ignore, &size_ignore,
       &byte_Ks, &size_K, &byte_C2Ws, &size_C2W,
       &byte_centers, &size_centers,
       &byte_IndexMap, &size_IndexMap,
       &byte_predefine, &size_predefine,
       &height, &width, &patch_size, &max_tracingtile, 
       &tile_shape.x, &tile_shape.y, &tile_shape.z,
       &scene_min_corner.x, &scene_min_corner.y, &scene_min_corner.z, 
       &num_voxel, &voxel_size, &dis_thresh, &sample_step, &need, &debug))
    {
        return NULL;
    }
    
    std::string img_path(s3);
    std::string diffuse_path(s5);

    float* Ks = (float*)byte_Ks;
    float* C2Ws = (float*)byte_C2Ws;
    num_camera = size_K / 36;

    float tile_size = (num_voxel - 2) * voxel_size;

    std::vector<float3> centers((float3*)byte_centers, (float3*)byte_centers+size_centers/12);
    std::vector<int> IndexMap((int*)byte_IndexMap, (int*)byte_IndexMap+size_IndexMap/4);
    std::vector<int> tileIdxs((int*)byte_tileIdxs, (int*)byte_tileIdxs+size_tileIdxs/4);
    std::vector<int> VisImg((int*)byte_VisImg, (int*)byte_VisImg + size_VisImg/4);
    std::vector<int> ignore((int*)byte_ignore, (int*)byte_ignore + size_ignore/4);
    std::vector<int> SparseToGroup((int*)byte_SparseToGroup, (int*)byte_SparseToGroup + size_SparseToGroup/4);
    std::vector<int> predefine_imgIdxs((int*)byte_predefine, (int*)byte_predefine + size_predefine/4);
    float* voxels = (float*)byte_voxels;
    short* nodes = (short*)byte_nodes;

    std::vector<float> data;
    std::vector<int> tiles_flag; 

    std::cout << "second iteration" << std::endl;
    preparedata_patch_sec( diffuse_path, img_path, predefine_imgIdxs,
                    tileIdxs, VisImg, ignore, centers, IndexMap, SparseToGroup,
                    voxels, nodes, sample_step, tile_shape, scene_min_corner,
                    max_tracingtile, Ks, C2Ws, num_camera, tile_size, height, width,
                    patch_size, num_voxel, voxel_size, dis_thresh, need, tiles_flag, data);   


    PyObject* Result = Py_BuildValue("y#y#",
                        (char*)&data[0], data.size()*sizeof(float),
                        (char*)&tiles_flag[0], tiles_flag.size()*sizeof(int));
    return Result;
}

static PyMethodDef gf_methods[] = {
    {"preparedata", Pypreparedata, METH_VARARGS},
    {"preparedata_v2", Pypreparedata_v2, METH_VARARGS},
    {"preparedata_patch", Pypreparedata_patch, METH_VARARGS},
    {"preparedata_patch_sec", Pypreparedata_patch_sec, METH_VARARGS},
    {NULL,NULL}
};

static struct PyModuleDef callModuleDef = {
    PyModuleDef_HEAD_INIT,
    "preparedata",
    "",
    -1,
    gf_methods
};

extern "C"

PyMODINIT_FUNC PyInit_preparedata(void){
    PyModule_Create(&callModuleDef);
}
