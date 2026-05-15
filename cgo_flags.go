package llama

/*
// Пути генерируются из build.conf (make cgo_flags.go)
#cgo CXXFLAGS: -std=c++17 -I${SRCDIR}/llama.cpp/include -I${SRCDIR}/llama.cpp/common -I${SRCDIR}/llama.cpp/ggml/include -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lbinding -L${SRCDIR}/llama.cpp/build/src -lllama -L${SRCDIR}/llama.cpp/build/common -lllama-common -lllama-common-base -L${SRCDIR}/llama.cpp/build/ggml/src -lggml -lggml-cpu -lggml-base -L${SRCDIR}/llama.cpp/build/vendor/cpp-httplib -lcpp-httplib -lstdc++ -lm -lpthread -fopenmp -ldl
*/
import "C"
