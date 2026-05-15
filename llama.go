package llama

// #include "binding.h"
// #include <stdlib.h>
import "C"
import (
    "fmt"
    "os"
    "strings"
    "sync"
    "unsafe"
)

type LLama struct {
    state       unsafe.Pointer
    embeddings  bool
    contextSize int
}

func New(model string, opts ...ModelOption) (*LLama, error) {
    mo := NewModelOptions(opts...)
    modelPath := C.CString(model)
    defer C.free(unsafe.Pointer(modelPath))
    loraBase := C.CString(mo.LoraBase)
    defer C.free(unsafe.Pointer(loraBase))
    loraAdapter := C.CString(mo.LoraAdapter)
    defer C.free(unsafe.Pointer(loraAdapter))

    MulMatQ := true

    if mo.MulMatQ != nil {
        MulMatQ = *mo.MulMatQ
    }

    result := C.load_model(modelPath,
        C.int(mo.ContextSize), C.int(mo.Seed),
        C.bool(mo.F16Memory), C.bool(mo.MLock), C.bool(mo.Embeddings), C.bool(mo.MMap), C.bool(mo.LowVRAM),
        C.int(mo.NGPULayers), C.int(mo.NBatch), C.CString(mo.MainGPU), C.CString(mo.TensorSplit), C.bool(mo.NUMA),
        C.float(mo.FreqRopeBase), C.float(mo.FreqRopeScale),
        C.bool(MulMatQ), loraAdapter, loraBase, C.bool(mo.Perplexity),
    )

    if result == nil {
        return nil, fmt.Errorf("failed loading model")
    }

    ll := &LLama{state: result, contextSize: mo.ContextSize, embeddings: mo.Embeddings}
    return ll, nil
}

func (l *LLama) Free() {
    C.llama_binding_free_model(l.state)
}

func (l *LLama) LoadState(state string) error {
    d := C.CString(state)
    w := C.CString("rb")
    result := C.load_state(l.state, d, w)

    defer C.free(unsafe.Pointer(d))
    defer C.free(unsafe.Pointer(w))

    if result != 0 {
        return fmt.Errorf("error while loading state")
    }

    return nil
}

func (l *LLama) SaveState(dst string) error {
    d := C.CString(dst)
    w := C.CString("wb")

    C.save_state(l.state, d, w)

    defer C.free(unsafe.Pointer(d))
    defer C.free(unsafe.Pointer(w))

    _, err := os.Stat(dst)
    return err
}

// Token Embeddings
func (l *LLama) TokenEmbeddings(tokens []int, opts ...PredictOption) ([]float32, error) {
    if !l.embeddings {
        return []float32{}, fmt.Errorf("model loaded without embeddings")
    }

    po := NewPredictOptions(opts...)

    outSize := po.Tokens
    if po.Tokens == 0 {
        outSize = 9999999
    }

    floats := make([]float32, outSize)

    myArray := (*C.int)(C.malloc(C.size_t(len(tokens)) * C.sizeof_int))

    for i, v := range tokens {
        (*[1 << 31]int32)(unsafe.Pointer(myArray))[i] = int32(v)
    }

    params := C.llama_allocate_params(C.CString(""), C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
        C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
        C.bool(po.IgnoreEOS), C.bool(po.F16KV),
        C.int(po.Batch), C.int(po.NKeep), nil, C.int(0),
        C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
        C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
        C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
        C.CString(po.MainGPU), C.CString(po.TensorSplit),
        C.bool(po.PromptCacheRO),
        C.CString(po.Grammar),
        C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), C.CString(po.NegativePrompt),
        C.int(po.NDraft),
    )
    ret := C.get_token_embeddings(params, l.state, myArray, C.int(len(tokens)), (*C.float)(&floats[0]))
    C.free(unsafe.Pointer(myArray))
    C.llama_free_params(params)
    if ret != 0 {
        return floats, fmt.Errorf("embedding inference failed")
    }
    return floats, nil
}

// Embeddings
func (l *LLama) Embeddings(text string, opts ...PredictOption) ([]float32, error) {
    if !l.embeddings {
        return []float32{}, fmt.Errorf("model loaded without embeddings")
    }

    po := NewPredictOptions(opts...)

    input := C.CString(text)
    defer C.free(unsafe.Pointer(input))
    if po.Tokens == 0 {
        po.Tokens = 99999999
    }
    floats := make([]float32, po.Tokens)
    reverseCount := len(po.StopPrompts)
    reversePrompt := make([]*C.char, reverseCount)
    var pass **C.char
    for i, s := range po.StopPrompts {
        cs := C.CString(s)
        defer C.free(unsafe.Pointer(cs))
        reversePrompt[i] = cs
        pass = &reversePrompt[0]
    }

    params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
        C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
        C.bool(po.IgnoreEOS), C.bool(po.F16KV),
        C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
        C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
        C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
        C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
        C.CString(po.MainGPU), C.CString(po.TensorSplit),
        C.bool(po.PromptCacheRO),
        C.CString(po.Grammar),
        C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), C.CString(po.NegativePrompt),
        C.int(po.NDraft),
    )

    ret := C.get_embeddings(params, l.state, (*C.float)(&floats[0]))
    C.llama_free_params(params)
    if ret != 0 {
        return floats, fmt.Errorf("embedding inference failed")
    }

    return floats, nil
}

func (l *LLama) Eval(text string, opts ...PredictOption) error {
    po := NewPredictOptions(opts...)

    input := C.CString(text)
    defer C.free(unsafe.Pointer(input))
    if po.Tokens == 0 {
        po.Tokens = 99999999
    }

    reverseCount := len(po.StopPrompts)
    reversePrompt := make([]*C.char, reverseCount)
    var pass **C.char
    for i, s := range po.StopPrompts {
        cs := C.CString(s)
        defer C.free(unsafe.Pointer(cs))
        reversePrompt[i] = cs
        pass = &reversePrompt[0]
    }

    params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
        C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
        C.bool(po.IgnoreEOS), C.bool(po.F16KV),
        C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
        C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
        C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
        C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
        C.CString(po.MainGPU), C.CString(po.TensorSplit),
        C.bool(po.PromptCacheRO),
        C.CString(po.Grammar),
        C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), C.CString(po.NegativePrompt),
        C.int(po.NDraft),
    )
    ret := C.eval(params, l.state, input)
    C.llama_free_params(params)
    if ret != 0 {
        return fmt.Errorf("inference failed")
    }

    return nil
}

func (l *LLama) SpeculativeSampling(ll *LLama, text string, opts ...PredictOption) (string, error) {
    po := NewPredictOptions(opts...)

    if po.TokenCallback != nil {
        setCallback(l.state, po.TokenCallback)
    }

    input := C.CString(text)
    defer C.free(unsafe.Pointer(input))
    if po.Tokens == 0 {
        po.Tokens = 99999999
    }
    out := make([]byte, po.Tokens)

    reverseCount := len(po.StopPrompts)
    reversePrompt := make([]*C.char, reverseCount)
    var pass **C.char
    for i, s := range po.StopPrompts {
        cs := C.CString(s)
        defer C.free(unsafe.Pointer(cs))
        reversePrompt[i] = cs
        pass = &reversePrompt[0]
    }

    params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
        C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
        C.bool(po.IgnoreEOS), C.bool(po.F16KV),
        C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
        C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
        C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
        C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
        C.CString(po.MainGPU), C.CString(po.TensorSplit),
        C.bool(po.PromptCacheRO),
        C.CString(po.Grammar),
        C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), C.CString(po.NegativePrompt),
        C.int(po.NDraft),
    )
    ret := C.speculative_sampling(params, l.state, ll.state, (*C.char)(unsafe.Pointer(&out[0])), C.bool(po.DebugMode))
    C.llama_free_params(params)

    if po.TokenCallback != nil {
        setCallback(l.state, nil)
    }

    if ret != 0 {
        return "", fmt.Errorf("inference failed")
    }
    res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

    res = strings.TrimPrefix(res, " ")
    res = strings.TrimPrefix(res, text)
    res = strings.TrimPrefix(res, "\n")

    for _, s := range po.StopPrompts {
        res = strings.TrimRight(res, s)
    }

    return res, nil
}

func (l *LLama) Predict(text string, opts ...PredictOption) (string, error) {
    po := NewPredictOptions(opts...)

    if po.TokenCallback != nil {
        setCallback(l.state, po.TokenCallback)
    }

    input := C.CString(text)
    defer C.free(unsafe.Pointer(input))
    if po.Tokens == 0 {
        po.Tokens = 99999999
    }
    out := make([]byte, po.Tokens)

    reverseCount := len(po.StopPrompts)
    reversePrompt := make([]*C.char, reverseCount)
    var pass **C.char
    for i, s := range po.StopPrompts {
        cs := C.CString(s)
        defer C.free(unsafe.Pointer(cs))
        reversePrompt[i] = cs
        pass = &reversePrompt[0]
    }

    params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
        C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
        C.bool(po.IgnoreEOS), C.bool(po.F16KV),
        C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
        C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
        C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
        C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
        C.CString(po.MainGPU), C.CString(po.TensorSplit),
        C.bool(po.PromptCacheRO),
        C.CString(po.Grammar),
        C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), C.CString(po.NegativePrompt),
        C.int(po.NDraft),
    )
    ret := C.llama_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])), C.bool(po.DebugMode))
    C.llama_free_params(params)

    if po.TokenCallback != nil {
        setCallback(l.state, nil)
    }

    if ret != 0 {
        return "", fmt.Errorf("inference failed")
    }
    res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

    res = strings.TrimPrefix(res, " ")
    res = strings.TrimPrefix(res, text)
    res = strings.TrimPrefix(res, "\n")

    for _, s := range po.StopPrompts {
        res = strings.TrimRight(res, s)
    }

    return res, nil
}

func (l *LLama) TokenizeString(text string, opts ...PredictOption) (int32, []int32, error) {
    po := NewPredictOptions(opts...)

    input := C.CString(text)
    defer C.free(unsafe.Pointer(input))
    if po.Tokens == 0 {
        po.Tokens = 4096
    }
    out := make([]C.int, po.Tokens)

    var fakeDblPtr **C.char

    params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
        C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
        C.bool(po.IgnoreEOS), C.bool(po.F16KV),
        C.int(po.Batch), C.int(po.NKeep), fakeDblPtr, C.int(0),
        C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
        C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
        C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
        C.CString(po.MainGPU), C.CString(po.TensorSplit),
        C.bool(po.PromptCacheRO),
        C.CString(po.Grammar),
        C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), C.CString(po.NegativePrompt),
        C.int(po.NDraft),
    )

    tokRet := C.llama_tokenize_string(params, l.state, (*C.int)(unsafe.Pointer(&out[0])))
    C.llama_free_params(params)

    if tokRet < 0 {
        return int32(tokRet), []int32{}, fmt.Errorf("llama_tokenize_string returned negative count %d", tokRet)
    }

    gTokRet := int32(tokRet)

    gLenOut := min(len(out), int(gTokRet))

    goSlice := make([]int32, gLenOut)
    for i := 0; i < gLenOut; i++ {
        goSlice[i] = int32(out[i])
    }

    return gTokRet, goSlice, nil
}

func (l *LLama) SetTokenCallback(callback func(token string) bool) {
    setCallback(l.state, callback)
}

var (
    m         sync.RWMutex
    callbacks = map[uintptr]func(string) bool{}
)

//export tokenCallback
func tokenCallback(statePtr unsafe.Pointer, token *C.char) bool {
    m.RLock()
    defer m.RUnlock()

    if callback, ok := callbacks[uintptr(statePtr)]; ok {
        return callback(C.GoString(token))
    }

    return true
}

func setCallback(statePtr unsafe.Pointer, callback func(string) bool) {
    m.Lock()
    defer m.Unlock()

    if callback == nil {
        delete(callbacks, uintptr(statePtr))
    } else {
        callbacks[uintptr(statePtr)] = callback
    }
}
