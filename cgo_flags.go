package llama

// Флаги CGO генерируются из build.conf (пути к /home/admin/cpp/llama.cpp).
// При смене пути отредактируйте build.conf и этот файл.

/*
#cgo CXXFLAGS: -std=c++17 -I/home/admin/cpp/llama.cpp/include -I/home/admin/cpp/llama.cpp/common -I/home/admin/cpp/llama.cpp/ggml/include -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lbinding -L/home/admin/cpp/llama.cpp/build/src -lllama -L/home/admin/cpp/llama.cpp/build/common -lllama-common -lllama-common-base -L/home/admin/cpp/llama.cpp/build/ggml/src -lggml -lggml-cpu -lggml-base -L/home/admin/cpp/llama.cpp/build/vendor/cpp-httplib -lcpp-httplib -lstdc++ -lm -lpthread -fopenmp -ldl
*/
import "C"
