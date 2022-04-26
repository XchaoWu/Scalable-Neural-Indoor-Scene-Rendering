# Replace by your own python 
python_header=${CONDA_PREFIX}/include/python3.8
# The dir that you install tqdm, cnpy, tinyply 
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
# echo "Build CUDA_EXT successfully!"


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
-L /usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -ltinyply -lcnpy \
-o $LIBDIR/preparedata.so \
-lcudart -lcuda
echo "Build preparedata successfully!"



g++ -std=c++11 -fPIC -shared \
warp.cpp \
-o $LIBDIR/compute_grid.so  \
-I $python_header
echo "Build compute_grid successfully!"



g++ -std=c++11 -fPIC -shared \
group_tiles.cpp \
-I $python_header \
-I $INCLUDE_DIR \
-I $CUDA_DIR/include \
-L $CUDA_DIR/lib64 \
-L $LIBDIR -lcuda_func \
-L /usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -ltinyply -lcnpy \
-o $LIBDIR/group_tiles.so \
-lcudart -lcuda
echo "Build group_tiles successfully!"