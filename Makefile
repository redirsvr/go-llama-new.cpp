.PHONY: all clean

MODDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include build.conf

LLAMA_ROOT  := $(MODDIR)/$(LLAMA_CPP_PATH)
LLAMA_BUILD := $(MODDIR)/$(LLAMA_BUILD_PATH)

LLAMA_INCLUDE := $(LLAMA_ROOT)/include
LLAMA_COMMON  := $(LLAMA_ROOT)/common
LLAMA_GGML    := $(LLAMA_ROOT)/ggml/include

CXXFLAGS := -std=c++17 -O3 -DNDEBUG -fPIC -pthread \
	-I$(LLAMA_INCLUDE) -I$(LLAMA_COMMON) -I$(LLAMA_GGML) -I$(MODDIR)

all: libbinding.a cgo_flags.go

# CGO: пути через ${SRCDIR}, чтобы go run/build находили библиотеки
cgo_flags.go: build.conf
	@REL=$$(grep '^LLAMA_CPP_PATH=' build.conf | cut -d= -f2); \
	printf '%s\n' \
		'package llama' \
		'' \
		'/*' \
		"// Пути генерируются из build.conf (make cgo_flags.go)" \
		"#cgo CXXFLAGS: -std=c++17 -I\$${SRCDIR}/$$REL/include -I\$${SRCDIR}/$$REL/common -I\$${SRCDIR}/$$REL/ggml/include -I\$${SRCDIR}" \
		"#cgo LDFLAGS: -L\$${SRCDIR} -lbinding -L\$${SRCDIR}/$$REL/build/src -lllama -L\$${SRCDIR}/$$REL/build/common -lllama-common -lllama-common-base -L\$${SRCDIR}/$$REL/build/ggml/src -lggml -lggml-cpu -lggml-base -L\$${SRCDIR}/$$REL/build/vendor/cpp-httplib -lcpp-httplib -lstdc++ -lm -lpthread -fopenmp -ldl" \
		'*/' \
		'import "C"' \
		> cgo_flags.go

$(LLAMA_BUILD)/src/libllama.a:
	cd $(LLAMA_BUILD) && cmake $(LLAMA_ROOT) -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && \
	cmake --build . --target llama llama-common -j$$(nproc)

binding.o: binding.cpp binding.h $(LLAMA_BUILD)/src/libllama.a
	$(CXX) $(CXXFLAGS) -c binding.cpp -o binding.o

libbinding.a: binding.o
	ar rcs libbinding.a binding.o
	@echo "OK: libbinding.a"

clean:
	rm -f binding.o libbinding.a
