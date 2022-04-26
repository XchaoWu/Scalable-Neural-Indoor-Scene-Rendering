# Replace by your own python 
python_header=${CONDA_PREFIX}/include/python3.8
# The dir that you install tqdm, cnpy, tinyply 
DEPENDENCY="/usr/local/lib"
CUDA_DIR="/usr/local/cuda"


BASEDIR=$(dirname "$0")
LIBDIR=$BASEDIR/lib
INCLUDE_DIR=$BASEDIR/include

if [ -d $LIBDIR ]; then
    echo file $LIBDIR exists!
else
    mkdir $LIBDIR
fi


# rm -r $BASEDIR/build 
# python setup.py build --build-lib $LIBDIR
# mv $LIBDIR/CUDA_EXT.*.so $LIBDIR/CUDA_EXT.so 
# echo "Build CUDA EXT successfully!"

nvcc -std=c++11 --shared --compiler-options "-fpic -shared" \
-I $INCLUDE_DIR -c -o $LIBDIR/libcuda_func.so cuda_func.cu
echo "Build CUDA functions successfully!"

g++ -std=c++11 -fPIC -shared \
preparedata.cpp \
-I $python_header \
-I $INCLUDE_DIR \
-I $CUDA_DIR/include \
-L $CUDA_DIR/lib64 \
-L $LIBDIR -lcuda_func \
-L /usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
-L $DEPENDENCY -ltinyply -lcnpy \
-o $LIBDIR/preparedata.so \
-lcudart -lcuda

if [$? == 0];then
    echo "Build preparedata successfully!"
else
    echo "[failed] Build preparedata!"
fi


g++ -std=c++11 -fPIC -shared \
warp.cpp \
-o $LIBDIR/compute_grid.so  \
-I $python_header

if [$? == 0];then
    echo "Build compute_grid successfully!"
else
    echo "[failed] Build compute_grid!"
fi


g++ -std=c++11 -fPIC -shared \
group_tiles.cpp \
-I $python_header \
-I $INCLUDE_DIR \
-I $CUDA_DIR/include \
-L $CUDA_DIR/lib64 \
-L $LIBDIR -lcuda_func \
-L /usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
-L $DEPENDENCY -ltinyply -lcnpy \
-o $LIBDIR/group_tiles.so \
-lcudart -lcuda

if [$? == 0];then
    echo "Build group_tiles successfully!"
else
    echo "[failed] Build group_tiles!"
fi