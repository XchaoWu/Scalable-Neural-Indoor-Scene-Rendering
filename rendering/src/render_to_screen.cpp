#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "helper_gl.h"
#include <GLFW/glfw3.h> 


#ifdef _MSC_VER
#pragma warning (disable: 4505) // unreferenced local function has been removed
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "helper_cuda.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;





int width, height;
GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)
py::object faster_render; 
bool* group_enable_flag;
int num_group = 0;

// render image using CUDA
void render_frame()
{
    // map PBO to get CUDA device pointer
    float *d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                         cuda_pbo_resource));     
    faster_render.attr("set_frame_pointer")((long)d_output);
    faster_render.attr("render_one_frame")();
    faster_render.attr("memcpy")();
    
    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}


void draw_imgui(){
    // auto& cam = rend.camera;
    
    static bool show_demo_window = true;
    static bool show_another_window = false;
    static bool Unet = false;
    static bool low_reflection = true;
    static int render_mode = 1;
    static float trans_decay = 0.5f;
    static float focal_scale = 1.0f;
    static int counter = 0;
    static ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    static float reflection_early_stop = 0.01;
    static float diffuse_early_stop = 0.01;
    static int max_tracing_tile = 30;
    static float sample_scale = 0.2f;
    // static int num_thread = 256;
    static float move_scale = 0.08f;
    // static float relfection_far_scale = 1.0f;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    {

        ImGui::Begin("Control Panel");                          // Create a window called "Hello, world!" and append into it.

        ImGui::Text("Rendering mode");
        ImGui::RadioButton("Frame", &render_mode, 1); ImGui::SameLine();
        ImGui::RadioButton("Diffuse", &render_mode, 2); ImGui::SameLine();
        ImGui::RadioButton("Reflection", &render_mode, 3); ImGui::SameLine();
        ImGui::RadioButton("Depth", &render_mode, 4); 
        ImGui::Checkbox("Unet", &Unet); ImGui::SameLine();
        ImGui::Checkbox("LR reflection", &low_reflection); ImGui::SameLine();
        if (ImGui::Button("Snapshot"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
            faster_render.attr("export_frame")();



        ImGui::SliderFloat("focal", &focal_scale, 0.5f, 2.0f);
        ImGui::SliderFloat("move scale", &move_scale, 0.01f, 1.0f);
        ImGui::SliderFloat("trans decay", &trans_decay, 0.0f, 1.0f);   
        ImGui::SliderFloat("early stop R", &reflection_early_stop, 0.0f, 0.5f);   
        ImGui::SliderFloat("early stop D", &diffuse_early_stop, 0.0f, 0.5f); 
    
        ImGui::SliderInt("max tracing tile", &max_tracing_tile, 0, 30); 
        ImGui::SliderFloat("sample step", &sample_scale, 0.2f, 1.0f);
        // ImGui::SliderInt("num thread", &num_thread, 64, 1024); 


        ImGui::End();
    }

    // Rendering
    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    faster_render.attr("set_mode")(render_mode);
    faster_render.attr("set_unet")(Unet);
    faster_render.attr("set_trans_decay")(trans_decay);
    faster_render.attr("set_focal_scale")(focal_scale);
    faster_render.attr("enable_lr_reflection")(low_reflection);
    faster_render.attr("set_early_stop_R")(reflection_early_stop);
    faster_render.attr("set_early_stop_D")(diffuse_early_stop);
    faster_render.attr("set_max_tracing_tile")(max_tracing_tile);
    faster_render.attr("set_sample_scale")(sample_scale);
    // faster_render.attr("set_num_thread")(num_thread);
    faster_render.attr("set_move_scale")(move_scale);
}

void initPixelBuffer()
{
    if (pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(float)*3, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}


double ox, oy;
int buttonState = 0;

void mouse(GLFWwindow* window, int button, int action, int mods)
{
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) return;

    glfwGetCursorPos(window, &ox, &oy);

    if (action == GLFW_PRESS)
    {
        buttonState  |= 1<<button;

        if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            //TODO: no x y
            faster_render.attr("display_groupIdx")(ox, oy);

        }else if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            buttonState = 1;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        buttonState = 0;

       
    }
}

void scrollCallback(GLFWwindow *window, double xOffset, double yOffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xOffset, yOffset);
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (yOffset==1.0)
    {
        faster_render.attr("zoom")(1.0f/1.01f);
    }else if (yOffset == -1.0)
    {
        faster_render.attr("zoom")(1.01f);
    }
}
void motion(GLFWwindow* window, double x, double y)
{
    if (ImGui::GetIO().WantCaptureMouse) return;
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 1)
    {
        // left = rotate
        faster_render.attr("rotate")(-dx / 100.0f, -dy / 100.0f);
    }

    ox = x;
    oy = y;
}


// void move (unsigned char key, int x, int y)
void move(GLFWwindow* window, int key, int scancode, int action, int mods)
{
 switch (key)
 {
    case 'W':
        faster_render.attr("move_forward")();
        break;
    case 'S':
        faster_render.attr("move_back")();
        break;
    case 'A':
        faster_render.attr("move_left")();
        break;
    case 'D':
        faster_render.attr("move_right")();
        break;
    case 'Q':
        faster_render.attr("move_up")();
        break;
    case 'E':
        faster_render.attr("move_down")();
        break;
    default:
        break;
 }
}



void cleanup()
{
    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
}

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void render_to_screen(py::object render, int w, int h, int ngroup)
{
    faster_render = render;
    // orbit_camera = orbit_cam;
    width = w;
    height = h;

    num_group = ngroup;
    group_enable_flag = new bool[num_group];
    for (int i=0; i<num_group; i++)
    {
        group_enable_flag[i] = true;
    }
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    glfwInit();
    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(width, height, "Scalable Neural Indoor Scene Rendering Viewer", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

#ifdef True
// Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); 
    (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

#endif


    glfwSetMouseButtonCallback(window, mouse);
    glfwSetCursorPosCallback(window, motion);
    glfwSetKeyCallback(window, move);
    glfwSetScrollCallback(window, scrollCallback);

    initPixelBuffer();
    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        glClearColor(1.0, 0.0, 0.0, 0.0);

        render_frame();
        
        // display results
        glClear(GL_COLOR_BUFFER_BIT);
        // draw image from PBO
        glDisable(GL_DEPTH_TEST);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        // copy from pbo to texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 1);
        glVertex2f(-1, -1);
        glTexCoord2f(1, 1);
        glVertex2f(1, -1);
        glTexCoord2f(1, 0);
        glVertex2f(1, 1);
        glTexCoord2f(0, 0);
        glVertex2f(-1, 1);
        glEnd();
        glDisable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);

        
        draw_imgui();

        glfwSwapBuffers(window);
    }

#ifdef True
// Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
#endif
    glfwDestroyWindow(window);
    glfwTerminate();

    delete [] group_enable_flag;
}
