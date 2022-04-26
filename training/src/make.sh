# Replace by your own python 
python_header=${CONDA_PREFIX}/include/python3.8
# The dir that you install tqdm, cnpy, tinyply 
dependency="/usr/local/lib"


rm -r build 
python setup.py build 
cp build/lib.linux-x86_64-3.8/CUDA_EXT.cpython-38-x86_64-linux-gnu.so lib/CUDA_EXT.so
echo "Build CUDA EXT successfully!"


nvcc -std=c++11 --shared --compiler-options "-fpic -shared" \
-I./include -c -o ./lib/libcuda_func.so cuda_func.cu
echo "Build CUDA functions successfully!"

g++ -std=c++11 -fPIC -shared \
preparedata.cpp \
-I $python_header \
-I /usr/local/include -I./include \
-I /usr/local/cuda/include \
-L /usr/local/cuda/lib64 \
-L ./lib -lcuda_func \
-L /usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
-L $dependency -ltinyply -lcnpy \
-o lib/preparedata.so \
-lcudart -lcuda
echo "Build preparedata successfully!"

g++ -std=c++11 -fPIC -shared \
warp.cpp \
-o lib/compute_grid.so  \
-I  $python_header
echo "Build compute_grid successfully!"

g++ -std=c++11 -fPIC -shared \
group_tiles.cpp \
-I $python_header \
-I /usr/local/include -I./include \
-I /usr/local/cuda/include \
-L /usr/local/cuda/lib64 \
-L ./lib -lcuda_func \
-L /usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
-L $dependency -ltinyply -lcnpy \
-o lib/group_tiles.so \
-lcudart -lcuda
echo "Build group_tiles successfully!"