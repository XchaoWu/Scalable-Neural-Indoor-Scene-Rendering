#ifndef _plyIO_H
#define _plyIO_H

#include <thread>
#include <chrono>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstring>
#include <iterator>
#include <tinyply.h>
#include "cutil_math.h"

#define DEBUG false 

using namespace tinyply;

inline std::vector<uint8_t> read_file_binary(const std::string & pathToFile)
{
    std::ifstream file(pathToFile, std::ios::binary);
    std::vector<uint8_t> fileBufferBytes;

    if (file.is_open())
    {
        file.seekg(0, std::ios::end);
        size_t sizeBytes = file.tellg();
        file.seekg(0, std::ios::beg);
        fileBufferBytes.resize(sizeBytes);
        if (file.read((char*)fileBufferBytes.data(), sizeBytes)) return fileBufferBytes;
    }
    else throw std::runtime_error("could not open binary ifstream to path " + pathToFile);
    return fileBufferBytes;
}

struct memory_buffer : public std::streambuf
{
    char * p_start {nullptr};
    char * p_end {nullptr};
    size_t size;

    memory_buffer(char const * first_elem, size_t size)
        : p_start(const_cast<char*>(first_elem)), p_end(p_start + size), size(size)
    {
        setg(p_start, p_start, p_end);
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override
    {
        if (dir == std::ios_base::cur) gbump(static_cast<int>(off));
        else setg(p_start, (dir == std::ios_base::beg ? p_start : p_end) + off, p_end);
        return gptr() - p_start;
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override
    {
        return seekoff(pos, std::ios_base::beg, which);
    }
};

struct memory_stream : virtual memory_buffer, public std::istream
{
    memory_stream(char const * first_elem, size_t size)
        : memory_buffer(first_elem, size), std::istream(static_cast<std::streambuf*>(this)) {}
};

class manual_timer
{
    std::chrono::high_resolution_clock::time_point t0;
    double timestamp{ 0.0 };
public:
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    void stop() { timestamp = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() * 1000.0; }
    const double & get() { return timestamp; }
};

struct geometry
{
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<float2> texcoords;
    std::vector<uint3> triangles;
};

inline geometry make_cube_geometry()
{
    geometry cube;

    const struct CubeVertex { float3 position, normal; float2 texCoord; } verts[] = {
    { { -1, -1, -1 },{ -1, 0, 0 },{ 0, 0 } },{ { -1, -1, +1 },{ -1, 0, 0 },{ 1, 0 } },{ { -1, +1, +1 },{ -1, 0, 0 },{ 1, 1 } },{ { -1, +1, -1 },{ -1, 0, 0 },{ 0, 1 } },
    { { +1, -1, +1 },{ +1, 0, 0 },{ 0, 0 } },{ { +1, -1, -1 },{ +1, 0, 0 },{ 1, 0 } },{ { +1, +1, -1 },{ +1, 0, 0 },{ 1, 1 } },{ { +1, +1, +1 },{ +1, 0, 0 },{ 0, 1 } },
    { { -1, -1, -1 },{ 0, -1, 0 },{ 0, 0 } },{ { +1, -1, -1 },{ 0, -1, 0 },{ 1, 0 } },{ { +1, -1, +1 },{ 0, -1, 0 },{ 1, 1 } },{ { -1, -1, +1 },{ 0, -1, 0 },{ 0, 1 } },
    { { +1, +1, -1 },{ 0, +1, 0 },{ 0, 0 } },{ { -1, +1, -1 },{ 0, +1, 0 },{ 1, 0 } },{ { -1, +1, +1 },{ 0, +1, 0 },{ 1, 1 } },{ { +1, +1, +1 },{ 0, +1, 0 },{ 0, 1 } },
    { { -1, -1, -1 },{ 0, 0, -1 },{ 0, 0 } },{ { -1, +1, -1 },{ 0, 0, -1 },{ 1, 0 } },{ { +1, +1, -1 },{ 0, 0, -1 },{ 1, 1 } },{ { +1, -1, -1 },{ 0, 0, -1 },{ 0, 1 } },
    { { -1, +1, +1 },{ 0, 0, +1 },{ 0, 0 } },{ { -1, -1, +1 },{ 0, 0, +1 },{ 1, 0 } },{ { +1, -1, +1 },{ 0, 0, +1 },{ 1, 1 } },{ { +1, +1, +1 },{ 0, 0, +1 },{ 0, 1 } }};

    std::vector<uint4> quads = { { 0, 1, 2, 3 },{ 4, 5, 6, 7 },{ 8, 9, 10, 11 },{ 12, 13, 14, 15 },{ 16, 17, 18, 19 },{ 20, 21, 22, 23 } };

    for (auto & q : quads)
    {
        cube.triangles.push_back({ q.x,q.y,q.z });
        cube.triangles.push_back({ q.x,q.z,q.w });
    }

    for (int i = 0; i < 24; ++i)
    {
        cube.vertices.push_back(verts[i].position);
        cube.normals.push_back(verts[i].normal);
        cube.texcoords.push_back(verts[i].texCoord);
    }

    return cube;
}


void write_ply_example(const std::string & filename)
{
    geometry cube = make_cube_geometry();

    std::filebuf fb_binary;
    fb_binary.open(filename + "-binary.ply", std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + filename);

    std::filebuf fb_ascii;
    fb_ascii.open(filename + "-ascii.ply", std::ios::out);
    std::ostream outstream_ascii(&fb_ascii);
    if (outstream_ascii.fail()) throw std::runtime_error("failed to open " + filename);

    PlyFile cube_file;

    cube_file.add_properties_to_element("vertex", { "x", "y", "z" }, 
        Type::FLOAT32, cube.vertices.size(), reinterpret_cast<uint8_t*>(cube.vertices.data()), Type::INVALID, 0);

    cube_file.add_properties_to_element("vertex", { "nx", "ny", "nz" },
        Type::FLOAT32, cube.normals.size(), reinterpret_cast<uint8_t*>(cube.normals.data()), Type::INVALID, 0);

    cube_file.add_properties_to_element("vertex", { "u", "v" },
        Type::FLOAT32, cube.texcoords.size() , reinterpret_cast<uint8_t*>(cube.texcoords.data()), Type::INVALID, 0);

    cube_file.add_properties_to_element("face", { "vertex_indices" },
        Type::UINT32, cube.triangles.size(), reinterpret_cast<uint8_t*>(cube.triangles.data()), Type::UINT8, 3);

    cube_file.get_comments().push_back("generated by tinyply 2.3");

    // Write an ASCII file
    cube_file.write(outstream_ascii, false);

    // Write a binary file
    cube_file.write(outstream_binary, true);
}

void read_plyFile(const std::string & filepath, 
                   std::vector<float3> &vertex_array,
                   std::vector<int3> &face_array,
                   std::vector<float2> &uv_array,
                   const bool preload_into_memory = true)
{

#if DEBUG
    std::cout << "........................................................................\n";
    std::cout << "Now Reading: " << filepath << std::endl;
#endif 

    std::unique_ptr<std::istream> file_stream;
    std::vector<uint8_t> byte_buffer;

    try
    {
        // For most files < 1gb, pre-loading the entire file upfront and wrapping it into a 
        // stream is a net win for parsing speed, about 40% faster. 
        if (preload_into_memory)
        {
            byte_buffer = read_file_binary(filepath);
            file_stream.reset(new memory_stream((char*)byte_buffer.data(), byte_buffer.size()));
        }
        else
        {
            file_stream.reset(new std::ifstream(filepath, std::ios::binary));
        }

        if (!file_stream || file_stream->fail()) throw std::runtime_error("file_stream failed to open " + filepath);

        file_stream->seekg(0, std::ios::end);
        const float size_mb = file_stream->tellg() * float(1e-6);
        file_stream->seekg(0, std::ios::beg);

        PlyFile file;
        file.parse_header(*file_stream);

#if DEBUG
        std::cout << "\t[ply_header] Type: " << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
        for (const auto & c : file.get_comments()) std::cout << "\t[ply_header] Comment: " << c << std::endl;
        for (const auto & c : file.get_info()) std::cout << "\t[ply_header] Info: " << c << std::endl; 

        for (const auto & e : file.get_elements())
        {
            std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
            for (const auto & p : e.properties)
            {
                std::cout << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.propertyType].str << ")";
                if (p.isList) std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
                std::cout << std::endl;
            }
        }
#endif 

        // Because most people have their own mesh types, tinyply treats parsed data as structured/typed byte buffers. 
        // See examples below on how to marry your own application-specific data structures with this one. 
        std::shared_ptr<PlyData> vertices, normals, colors, texcoords, faces, tripstrip;

        // The header information can be used to programmatically extract properties on elements
        // known to exist in the header prior to reading the data. For brevity of this sample, properties 
        // like vertex position are hard-coded: 
        try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
        catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl;*/ }

        try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
        catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl;*/ }

        try { colors = file.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" }); }
        catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl;*/ }

        try { colors = file.request_properties_from_element("vertex", { "r", "g", "b", "a" }); }
        catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl;*/ }

        try { texcoords = file.request_properties_from_element("vertex", { "texture_u", "texture_v" }); }
        catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl;*/ }

        // Providing a list size hint (the last argument) is a 2x performance improvement. If you have 
        // arbitrary ply files, it is best to leave this 0. 
        try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
        catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl;*/ }

        // Tristrips must always be read with a 0 list size hint (unless you know exactly how many elements
        // are specifically in the file, which is unlikely); 
        try { tripstrip = file.request_properties_from_element("tristrips", { "vertex_indices" }, 0); }
        catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl;*/ }

        manual_timer read_timer;

        read_timer.start();
        file.read(*file_stream);
        read_timer.stop();

        const float parsing_time = static_cast<float>(read_timer.get()) / 1000.f;
        std::cout << "\tparsing " << size_mb << "mb in " << parsing_time << " seconds [" << (size_mb / parsing_time) << " MBps]" << std::endl;

        if (vertices)   std::cout << "\tRead " << vertices->count  << " total vertices "<< std::endl;
        if (normals)    std::cout << "\tRead " << normals->count   << " total vertex normals " << std::endl;
        if (colors)     std::cout << "\tRead " << colors->count << " total vertex colors " << std::endl;
        if (texcoords)  std::cout << "\tRead " << texcoords->count << " total vertex texcoords " << std::endl;
        if (faces)      std::cout << "\tRead " << faces->count     << " total faces (triangles) " << std::endl;
        if (tripstrip)  std::cout << "\tRead " << (tripstrip->buffer.size_bytes() / tinyply::PropertyTable[tripstrip->t].stride) << " total indicies (tristrip) " << std::endl;



        // std::cout << vertices->buffer.size_bytes() << std::endl;
        // vertex = Eigen::Map<Eigen::MatrixXf>(vertices->buffer.get(), (int)vertices->buffer.size_bytes(), 3);
        // // Example One: converting to your own application types
        // {
        if (vertices)
        {
            const size_t numVerticesBytes = vertices->buffer.size_bytes();
            std::vector<float3> verts(vertices->count);
            std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);
            vertex_array = verts;
        }

        if (faces)
        {
            const size_t numFacesBytes = faces->buffer.size_bytes();
            std::vector<int3> trifaces(faces->count);
            std::memcpy(trifaces.data(), faces->buffer.get(), numFacesBytes);
            face_array = trifaces;
        }

        if (texcoords)
        {
            const size_t numUvBytes = texcoords->buffer.size_bytes();
            std::vector<float2> triuv(texcoords->count);
            std::memcpy(triuv.data(), texcoords->buffer.get(), numUvBytes);
            uv_array = triuv;
        }
        // }

        // // Example Two: converting to your own application type
        // {
        //     std::vector<float3> verts_floats;
        //     std::vector<double3> verts_doubles;
        //     if (vertices->t == tinyply::Type::FLOAT32) { /* as floats ... */ }
        //     if (vertices->t == tinyply::Type::FLOAT64) { /* as doubles ... */ }
        // }
    }
    catch (const std::exception & e)
    {
        std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
    }
}



#endif 