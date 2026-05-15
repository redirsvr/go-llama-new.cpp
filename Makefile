.PHONY: all clean help cpu cuda rocm

MODDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Не использовать BUILD_BACKEND из окружения (BUILD_BACKEND=cuda make и т.п.)
_ENV_BUILD_BACKEND := $(BUILD_BACKEND)
BUILD_BACKEND :=

include build.conf

ifndef BUILD_BACKEND
BUILD_BACKEND := cpu
endif

# Бэкенд: make cpu|cuda|rocm или BUILD_BACKEND в build.conf
BACKEND_GOAL := $(firstword $(filter cpu cuda rocm,$(MAKECMDGOALS)))
ifneq ($(BACKEND_GOAL),)
  BUILD_BACKEND := $(BACKEND_GOAL)
else
  ifneq ($(strip $(_ENV_BUILD_BACKEND)),)
    ifneq ($(strip $(_ENV_BUILD_BACKEND)),$(strip $(BUILD_BACKEND)))
      $(error Бэкенд задаётся через «make cpu», «make cuda», «make rocm» или BUILD_BACKEND в build.conf)
    endif
  endif
endif
CUDA_PATH     ?= /usr/local/cuda
ROCM_PATH     ?= /opt/rocm

LLAMA_ROOT  := $(MODDIR)/$(LLAMA_CPP_PATH)
LLAMA_BUILD := $(MODDIR)/$(LLAMA_BUILD_PATH)

LLAMA_INCLUDE := $(LLAMA_ROOT)/include
LLAMA_COMMON  := $(LLAMA_ROOT)/common
LLAMA_GGML    := $(LLAMA_ROOT)/ggml/include

# Проверка BACKEND
VALID_BACKENDS := cpu cuda rocm
ifeq ($(filter $(BUILD_BACKEND),$(VALID_BACKENDS)),)
$(error Неверный BUILD_BACKEND="$(BUILD_BACKEND)". Допустимо: cpu, cuda, rocm)
endif

# CMake: общие флаги + бэкенд
CMAKE_COMMON := -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
CMAKE_BACKEND :=
CMAKE_ENV :=
GGML_LINK_LIBS := -lggml -lggml-cpu -lggml-base
CGO_EXTRA_LDFLAGS :=

ifeq ($(BUILD_BACKEND),cpu)
  CMAKE_BACKEND :=
endif

ifeq ($(BUILD_BACKEND),cuda)
  CMAKE_BACKEND := -DGGML_CUDA=ON
  GGML_LINK_LIBS := -lggml -lggml-cpu -lggml-cuda -lggml-base
  CGO_EXTRA_LDFLAGS := -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib/stubs \
	-lcudart -lcublas -lcublasLt -lcuda
  ifneq ($(strip $(CMAKE_CUDA_ARCHITECTURES)),)
    CMAKE_BACKEND += -DCMAKE_CUDA_ARCHITECTURES="$(CMAKE_CUDA_ARCHITECTURES)"
  endif
endif

ifeq ($(BUILD_BACKEND),rocm)
  CMAKE_BACKEND := -DGGML_HIP=ON
  GGML_LINK_LIBS := -lggml -lggml-cpu -lggml-hip -lggml-base
  CGO_EXTRA_LDFLAGS := -L$(ROCM_PATH)/lib -lamdhip64 -lrocblas -lhipblas
  ifneq ($(strip $(ROCM_GPU_TARGETS)),)
    CMAKE_BACKEND += -DGPU_TARGETS="$(ROCM_GPU_TARGETS)"
  endif
  ROCM_CC  := $(ROCM_PATH)/bin/amdclang
  ROCM_CXX := $(ROCM_PATH)/bin/amdclang++
  ifneq ($(wildcard $(ROCM_CC)),)
    CMAKE_ENV := CC="$(ROCM_CC)" CXX="$(ROCM_CXX)"
  else
    ROCM_CC  := $(ROCM_PATH)/bin/hipcc
    ROCM_CXX := $(ROCM_PATH)/bin/hipcc
    ifneq ($(wildcard $(ROCM_CC)),)
      CMAKE_ENV := CC="$(ROCM_CC)" CXX="$(ROCM_CXX)"
    endif
  endif
endif

CMAKE_ARGS := $(CMAKE_COMMON) $(CMAKE_BACKEND)

CXXFLAGS := -std=c++17 -O3 -DNDEBUG -fPIC -pthread \
	-I$(LLAMA_INCLUDE) -I$(LLAMA_COMMON) -I$(LLAMA_GGML) -I$(MODDIR)

NPROC := $(shell nproc 2>/dev/null || echo 4)

all: libbinding.a cgo_flags.go
	@echo "Сборка завершена: BUILD_BACKEND=$(BUILD_BACKEND) LLAMA_BUILD_PATH=$(LLAMA_BUILD_PATH)"

cpu cuda rocm: all

help:
	@echo "go-llama-new.cpp — сборка нативного ядра и CGO-обёртки"
	@echo ""
	@echo "Сборка по бэкенду:"
	@echo "  make          — BUILD_BACKEND из build.conf"
	@echo "  make cpu      — только CPU"
	@echo "  make cuda     — NVIDIA CUDA + CPU offloading"
	@echo "  make rocm     — AMD ROCm/HIP + CPU offloading"
	@echo ""
	@echo "Пути (LLAMA_BUILD_PATH, CUDA_PATH, ROCM_GPU_TARGETS) — в build.conf."
	@echo "Для cuda/rocm рекомендуется отдельный LLAMA_BUILD_PATH:"
	@echo "  llama.cpp/build-cuda  или  llama.cpp/build-rocm"
	@echo ""
	@echo "Цели: all, cpu, cuda, rocm, clean, help"

# CGO: пути из build.conf, библиотеки — по BUILD_BACKEND (цель make cuda или build.conf)
cgo_flags.go: build.conf Makefile
	@REL=$$(grep '^LLAMA_CPP_PATH=' build.conf | cut -d= -f2); \
	BUILD=$$(grep '^LLAMA_BUILD_PATH=' build.conf | cut -d= -f2); \
	BACKEND="$(BUILD_BACKEND)"; \
	case "$$BACKEND" in \
	  cuda) GGML_LIBS="-lggml -lggml-cpu -lggml-cuda -lggml-base" ;; \
	  rocm) GGML_LIBS="-lggml -lggml-cpu -lggml-hip -lggml-base" ;; \
	  *)    GGML_LIBS="-lggml -lggml-cpu -lggml-base" ;; \
	esac; \
	EXTRA=""; \
	CUDA_PATH=$$(grep '^CUDA_PATH=' build.conf 2>/dev/null | cut -d= -f2); \
	ROCM_PATH=$$(grep '^ROCM_PATH=' build.conf 2>/dev/null | cut -d= -f2); \
	[ -z "$$CUDA_PATH" ] && CUDA_PATH=/usr/local/cuda; \
	[ -z "$$ROCM_PATH" ] && ROCM_PATH=/opt/rocm; \
	case "$$BACKEND" in \
	  cuda) EXTRA="-L$$CUDA_PATH/lib64 -L$$CUDA_PATH/lib/stubs -lcudart -lcublas -lcublasLt -lcuda" ;; \
	  rocm) EXTRA="-L$$ROCM_PATH/lib -lamdhip64 -lrocblas -lhipblas" ;; \
	esac; \
	printf '%s\n' \
		'package llama' \
		'' \
		'/*' \
		"// Пути и LDFLAGS генерируются из build.conf (make cgo_flags.go)" \
		"// BUILD_BACKEND=$$BACKEND" \
		"#cgo CXXFLAGS: -std=c++17 -I\$${SRCDIR}/$$REL/include -I\$${SRCDIR}/$$REL/common -I\$${SRCDIR}/$$REL/ggml/include -I\$${SRCDIR}" \
		"#cgo LDFLAGS: -L\$${SRCDIR} -lbinding -L\$${SRCDIR}/$$BUILD/src -lllama -L\$${SRCDIR}/$$BUILD/common -lllama-common -lllama-common-base -L\$${SRCDIR}/$$BUILD/ggml/src $$GGML_LIBS -L\$${SRCDIR}/$$BUILD/vendor/cpp-httplib -lcpp-httplib -lstdc++ -lm -lpthread -fopenmp -ldl $$EXTRA" \
		'*/' \
		'import "C"' \
		> cgo_flags.go

$(LLAMA_BUILD)/src/libllama.a:
	@mkdir -p $(LLAMA_BUILD)
	@echo "CMake: BUILD_BACKEND=$(BUILD_BACKEND) -> $(LLAMA_BUILD)"
	cd $(LLAMA_BUILD) && $(CMAKE_ENV) cmake $(LLAMA_ROOT) $(CMAKE_ARGS)
	cd $(LLAMA_BUILD) && cmake --build . --target llama llama-common -j$(NPROC)

binding.o: binding.cpp binding.h $(LLAMA_BUILD)/src/libllama.a
	$(CXX) $(CXXFLAGS) -c binding.cpp -o binding.o

libbinding.a: binding.o
	ar rcs libbinding.a binding.o
	@echo "OK: libbinding.a (BUILD_BACKEND=$(BUILD_BACKEND))"

clean:
	rm -f binding.o libbinding.a
