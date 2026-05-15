.PHONY: all clean libbinding.a

include build.conf

LLAMA_INCLUDE := $(LLAMA_CPP_PATH)/include
LLAMA_COMMON  := $(LLAMA_CPP_PATH)/common
LLAMA_GGML    := $(LLAMA_CPP_PATH)/ggml/include

CXXFLAGS := -std=c++17 -O3 -DNDEBUG -fPIC -pthread \
	-I$(LLAMA_INCLUDE) -I$(LLAMA_COMMON) -I$(LLAMA_GGML) -I.

LDFLAGS_LIBS := \
	-L$(LLAMA_BUILD_PATH)/src -lllama \
	-L$(LLAMA_BUILD_PATH)/common -lllama-common \
	-L$(LLAMA_BUILD_PATH)/ggml/src -lggml -lggml-cpu -lggml-base \
	-L$(LLAMA_BUILD_PATH)/vendor/cpp-httplib -lcpp-httplib \
	-lpthread -fopenmp -ldl -lm -lstdc++

all: libbinding.a cgo_flags.go

# Обновить пути в cgo_flags.go из build.conf
cgo_flags.go: build.conf
	@LLAMA=$$(grep '^LLAMA_CPP_PATH=' build.conf | cut -d= -f2); \
	BUILD=$$(grep '^LLAMA_BUILD_PATH=' build.conf | cut -d= -f2); \
	printf '%s\n' \
		'package llama' \
		'' \
		'/*' \
		"#cgo CXXFLAGS: -std=c++17 -I$$LLAMA/include -I$$LLAMA/common -I$$LLAMA/ggml/include -I\$${SRCDIR}" \
		"#cgo LDFLAGS: -L\$${SRCDIR} -lbinding -L$$BUILD/src -lllama -L$$BUILD/common -lllama-common -lllama-common-base -L$$BUILD/ggml/src -lggml -lggml-cpu -lggml-base -L$$BUILD/vendor/cpp-httplib -lcpp-httplib -lstdc++ -lm -lpthread -fopenmp -ldl" \
		'*/' \
		'import "C"' \
		> cgo_flags.go

$(LLAMA_BUILD_PATH)/src/libllama.a:
	cd $(LLAMA_BUILD_PATH) && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && \
	cmake --build . --target llama llama-common -j$$(nproc)

binding.o: binding.cpp binding.h $(LLAMA_BUILD_PATH)/src/libllama.a
	$(CXX) $(CXXFLAGS) -c binding.cpp -o binding.o

libbinding.a: binding.o
	ar rcs libbinding.a binding.o
	@echo "Собрано: libbinding.a. Линковка llama.cpp — через cgo_flags.go."

clean:
	rm -f binding.o libbinding.a
