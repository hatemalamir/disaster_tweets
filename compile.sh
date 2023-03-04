cd build
rm -r *
cd ..
cmake -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/Users/halamir/Documents/work/ml/lib/pytorch-install -H. -Bbuild
cd build
make disaster-tweets
