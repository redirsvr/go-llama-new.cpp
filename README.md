# go-llama-new.cpp

Ядро собирается из локальных исходников llama.cpp (не из submodule внутри репозитория). Пути к исходникам задаются в файле `build.conf`, переменные окружения для этого не используются.

## Требования

- **Go** 1.21 или новее (с поддержкой CGO)
- **Компилятор C++** с поддержкой C++17 (`g++` / `clang++`)
- **CMake** 3.14+
- **make**, **ar**
- **OpenMP** (обычно пакет `libgomp` в Linux)
- Инструменты сборки: `git`, `build-essential` (или аналог)

Для линковки также нужны статические библиотеки, которые CMake собирает из llama.cpp: `libllama.a`, `libllama-common.a`, `libllama-common-base.a`, `libggml*.a`, `libcpp-httplib.a`.

## Настройка путей

Отредактируйте `build.conf` в корне модуля:

```ini
# Пути к исходникам llama.cpp (без переменных окружения)
LLAMA_CPP_PATH=./llama.cpp
LLAMA_BUILD_PATH=./llama.cpp/build
```

| Параметр | Описание |
|----------|----------|
| `LLAMA_CPP_PATH` | Каталог с исходниками llama.cpp (`include/`, `common/`, `src/` и т.д.) |
| `LLAMA_BUILD_PATH` | Каталог сборки CMake (там появятся `build/src/libllama.a` и др.) |

После изменения `build.conf` выполните `make` — будет пересоздан `cgo_flags.go` с актуальными путями для CGO.

## Сборка

Сборка состоит из двух этапов: сначала нативное ядро llama.cpp, затем Go-модуль с C-обёрткой `binding`.

### 1. Сборка llama.cpp

```bash
mkdir -p ./llama.cpp/build
cd ./llama.cpp/build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF

cmake --build . --target llama llama-common -j"$(nproc)"
```

Проверка, что библиотеки на месте:

```bash
ls -la build/src/libllama.a
ls -la build/common/libllama-common.a
ls -la build/common/libllama-common-base.a
ls -la build/ggml/src/libggml.a
```

Цель `make` в каталоге модуля при необходимости запустит эту же сборку автоматически (см. `Makefile`).

### 2. Сборка C-обёртки (libbinding.a)

В каталоге модуля:

```bash
cd /path/to/go-llama-new.cpp
make
```

Будет выполнено:

1. Генерация `cgo_flags.go` из `build.conf`
2. Компиляция `binding.cpp` → `binding.o`
3. Создание архива `libbinding.a`

Очистка артефактов обёртки:

```bash
make clean
```

### 3. Сборка Go-модуля

```bash
go build ./...
```

Или пример:

```bash
go build -o llama-example ./examples/
go run ./examples/main.go /path/to/model.gguf "Привет, мир"
```

При первой сборке CGO скомпилирует `binding.cpp` ещё раз и слинкует его с библиотеками из `LLAMA_BUILD_PATH` (см. `cgo_flags.go`).

## Использование в своём проекте

```go
import llama "go-llama-new.cpp"

func main() {
    model, err := llama.New("/path/to/model.gguf",
        llama.SetContext(4096),
        llama.SetGPULayers(0),
    )
    if err != nil {
        panic(err)
    }
    defer model.Free()

    text, err := model.Predict("Привет",
        llama.SetTokens(128),
        llama.SetTemperature(0.8),
    )
    if err != nil {
        panic(err)
    }
    println(text)
}
```

В `go.mod` вашего проекта:

```go
require go-llama-new.cpp v0.0.0

replace go-llama-new.cpp => /path/to/go-llama-new.cpp
```

Перед `go build` в проекте-потребителе должны быть собраны llama.cpp и `libbinding.a` (шаги 1–2 выше).

## Опциональные теги сборки

Как в оригинальном go-llama.cpp:

| Тег | Назначение |
|-----|------------|
| `openblas` | Дополнительная линковка с OpenBLAS (`llama_openblas.go`) |
| `cublas` | CUDA (`llama_cublas.go`) — требует отдельной сборки llama.cpp с `GGML_CUDA=ON` |

Пример:

```bash
go build -tags openblas ./...
```

Для GPU нужно пересобрать llama.cpp с нужными опциями CMake (например `-DGGML_CUDA=ON`) и убедиться, что пути в `build.conf` указывают на эту сборку.

## Устранение неполадок

### `неопределённая ссылка на llama_compiler` / `llama_commit` / `llama_build_number`

Не слинкована `libllama-common-base.a`. Убедитесь, что в `cgo_flags.go` в `LDFLAGS` есть `-lllama-common-base`, и пересоберите:

```bash
make
go build ./...
```

### `cannot find -lllama` или `-lllama-common`

Проверьте `LLAMA_BUILD_PATH` в `build.conf` и выполните сборку llama.cpp (шаг 1).

### CGO отключён

```bash
go env CGO_ENABLED   # должно быть 1
```

Установите `gcc`/`g++`, если CGO выключен из-за отсутствия компилятора C.

### Изменили путь к llama.cpp

1. Обновите `build.conf`
2. `make` (обновит `cgo_flags.go` и `libbinding.a`)
3. `go build ./...`

## Структура репозитория

```
.
├── build.conf          # пути к llama.cpp
├── binding.h
├── binding.cpp         # C API для CGO
├── cgo_flags.go        # флаги CGO (генерируется make)
├── llama.go
├── options.go
├── Makefile
├── examples/main.go
└── README.md
```

## Лицензия

Следует лицензиям llama.cpp и исходного go-llama.cpp. Используйте в соответствии с условиями соответствующих проектов.
