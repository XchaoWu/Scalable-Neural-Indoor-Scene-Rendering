# rm -r build 
# python setup.py build 
# cp build/lib.linux-x86_64-3.8/CUDA_EXT.cpython-38-x86_64-linux-gnu.so lib/CUDA_EXT.so

# rm -r build 
# python setup.py build 
# cp build/lib.linux-x86_64-3.8/RENDERING.cpython-38-x86_64-linux-gnu.so lib/RENDERING.so

nvcc -std=c++11 --shared --compiler-options "-fpic -shared" \
-I./include -c -o ./lib/libcuda_func.so cuda_func.cu
g++ -std=c++11 -fPIC -shared \
preparedata.cpp \
-I/home/yons/anaconda3/envs/py38/include/python3.8 \
-I/usr/local/include -I./include \
-I /usr/local/cuda/include \
-L /usr/local/cuda/lib64 \
-L ./lib -lcuda_func \
-L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
-L/usr/local/lib -ltinyply -lcnpy \
-o lib/preparedata.so \
-lcudart -lcuda


# g++ -std=c++11 -fPIC -shared \
# warp.cpp \
# -o lib/compute_grid.so  \
# -I /home/yons/anaconda3/envs/py38/include/python3.8

# g++ -std=c++11 -fPIC -shared \
# group_tiles.cpp \
# -I/home/yons/anaconda3/envs/py38/include/python3.8 \
# -I/usr/local/include -I./include \
# -I /usr/local/cuda/include \
# -L /usr/local/cuda/lib64 \
# -L ./lib -lcuda_func \
# -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
# -L/usr/local/lib -ltinyply -lcnpy \
# -o lib/group_tiles.so \
# -lcudart -lcuda

# g++ -std=c++11 -fPIC -shared \
# gen_texture.cpp \
# -I/home/yons/anaconda3/envs/py38/include/python3.8 \
# -I/usr/local/include -I./include \
# -I /usr/local/cuda/include \
# -L /usr/local/cuda/lib64 \
# -L ./lib -lcuda_func \
# -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
# -L/usr/local/lib -ltinyply -lcnpy \
# -o lib/gentexture.so \
# -lcudart -lcuda

# nvcc -std=c++11 -arch=sm_80 --shared --compiler-options "-fpic -shared" \
# -I./include -c -o ./lib/librendering.so rendering_kernel.cu
# g++ -std=c++11 -fPIC \
# rendering.cpp \
# -I/home/yons/anaconda3/envs/py38/include/python3.8 \
# -I/usr/local/include -I./include \
# -I /usr/local/cuda/include \
# -L /usr/local/cuda/lib64 \
# -L ./lib -lrendering \
# -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
# -L/usr/local/lib -ltinyply -lcnpy \
# -o rendering.out \
# -lcudart -lcuda
