package llama

// Флаги CGO генерируются из build.conf (пути к ./llama.cpp).
// При смене пути отредактируйте build.conf и этот файл.

/*
#cgo CXXFLAGS: -std=c++17 -I./llama.cpp/include -I./llama.cpp/common -I./llama.cpp/ggml/include -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lbinding -L./llama.cpp/build/src -lllama -L./llama.cpp/build/common -lllama-common -lllama-common-base -L./llama.cpp/build/ggml/src -lggml -lggml-cpu -lggml-base -L./llama.cpp/build/vendor/cpp-httplib -lcpp-httplib -lstdc++ -lm -lpthread -fopenmp -ldl
*/
import "C"
