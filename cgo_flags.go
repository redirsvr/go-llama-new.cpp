package llama

/*
// Пути и LDFLAGS генерируются из build.conf (make cgo_flags.go)
// BUILD_BACKEND=cuda
#cgo CXXFLAGS: -std=c++17 -I${SRCDIR}/llama.cpp/include -I${SRCDIR}/llama.cpp/common -I${SRCDIR}/llama.cpp/ggml/include -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lbinding -L${SRCDIR}/llama.cpp/build/src -lllama -L${SRCDIR}/llama.cpp/build/common -lllama-common -lllama-common-base -L${SRCDIR}/llama.cpp/build/ggml/src -lggml -lggml-cpu -lggml-cuda -lggml-base -L${SRCDIR}/llama.cpp/build/vendor/cpp-httplib -lcpp-httplib -lstdc++ -lm -lpthread -fopenmp -ldl -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib/stubs -lcudart -lcublas -lcublasLt -lcuda
*/
import "C"
