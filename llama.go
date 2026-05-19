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
	mu          sync.Mutex
	closed      bool
}

func (l *LLama) lockState() (unsafe.Pointer, func(), error) {
	if l == nil {
		return nil, nil, fmt.Errorf("llama model is nil")
	}

	l.mu.Lock()
	if l.closed || l.state == nil {
		l.mu.Unlock()
		return nil, nil, fmt.Errorf("llama model is closed")
	}

	return l.state, l.mu.Unlock, nil
}

func lockTwoStates(a, b *LLama) (unsafe.Pointer, unsafe.Pointer, func(), error) {
	if a == nil || b == nil {
		return nil, nil, nil, fmt.Errorf("llama model is nil")
	}

	if a == b {
		state, unlock, err := a.lockState()
		if err != nil {
			return nil, nil, nil, err
		}
		return state, state, unlock, nil
	}

	first, second := a, b
	if uintptr(unsafe.Pointer(first)) > uintptr(unsafe.Pointer(second)) {
		first, second = second, first
	}

	first.mu.Lock()
	second.mu.Lock()

	unlock := func() {
		second.mu.Unlock()
		first.mu.Unlock()
	}

	if first.closed || first.state == nil || second.closed || second.state == nil {
		unlock()
		return nil, nil, nil, fmt.Errorf("llama model is closed")
	}

	return a.state, b.state, unlock, nil
}

func cStringForCall(s string, ptrs *[]unsafe.Pointer) *C.char {
	cs := C.CString(s)
	*ptrs = append(*ptrs, unsafe.Pointer(cs))
	return cs
}

func freeCStrings(ptrs []unsafe.Pointer) {
	for _, ptr := range ptrs {
		C.free(ptr)
	}
}

func allocatePredictParams(input *C.char, po PredictOptions, pass **C.char, reverseCount int, cstrs *[]unsafe.Pointer) unsafe.Pointer {
	return C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), cStringForCall(po.LogitBias, cstrs),
		cStringForCall(po.PathPromptCache, cstrs), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		cStringForCall(po.MainGPU, cstrs), cStringForCall(po.TensorSplit, cstrs),
		C.bool(po.PromptCacheRO),
		cStringForCall(po.Grammar, cstrs),
		C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), cStringForCall(po.NegativePrompt, cstrs),
		C.int(po.NDraft),
	)
}

func New(model string, opts ...ModelOption) (*LLama, error) {
	mo := NewModelOptions(opts...)
	modelPath := C.CString(model)
	defer C.free(unsafe.Pointer(modelPath))
	loraBase := C.CString(mo.LoraBase)
	defer C.free(unsafe.Pointer(loraBase))
	loraAdapter := C.CString(mo.LoraAdapter)
	defer C.free(unsafe.Pointer(loraAdapter))
	mainGPU := C.CString(mo.MainGPU)
	defer C.free(unsafe.Pointer(mainGPU))
	tensorSplit := C.CString(mo.TensorSplit)
	defer C.free(unsafe.Pointer(tensorSplit))

	MulMatQ := true

	if mo.MulMatQ != nil {
		MulMatQ = *mo.MulMatQ
	}

	result := C.load_model(modelPath,
		C.int(mo.ContextSize), C.int(mo.Seed),
		C.bool(mo.F16Memory), C.bool(mo.MLock), C.bool(mo.Embeddings), C.bool(mo.MMap), C.bool(mo.LowVRAM),
		C.int(mo.NGPULayers), C.int(mo.NBatch), mainGPU, tensorSplit, C.bool(mo.NUMA),
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
	state, unlock, err := l.lockState()
	if err != nil {
		return
	}
	defer unlock()

	C.llama_binding_free_model(state)
	l.state = nil
	l.closed = true
}

func (l *LLama) LoadState(state string) error {
	llamaState, unlock, err := l.lockState()
	if err != nil {
		return err
	}
	defer unlock()

	d := C.CString(state)
	w := C.CString("rb")
	result := C.load_state(llamaState, d, w)

	defer C.free(unsafe.Pointer(d))
	defer C.free(unsafe.Pointer(w))

	if result != 0 {
		return fmt.Errorf("error while loading state")
	}

	return nil
}

func (l *LLama) SaveState(dst string) error {
	llamaState, unlock, err := l.lockState()
	if err != nil {
		return err
	}
	defer unlock()

	d := C.CString(dst)
	w := C.CString("wb")

	C.save_state(llamaState, d, w)

	defer C.free(unsafe.Pointer(d))
	defer C.free(unsafe.Pointer(w))

	_, statErr := os.Stat(dst)
	return statErr
}

// Token Embeddings
func (l *LLama) TokenEmbeddings(tokens []int, opts ...PredictOption) ([]float32, error) {
	state, unlock, err := l.lockState()
	if err != nil {
		return nil, err
	}
	defer unlock()

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

	var cstrs []unsafe.Pointer
	defer func() { freeCStrings(cstrs) }()

	input := cStringForCall("", &cstrs)
	params := allocatePredictParams(input, po, nil, 0, &cstrs)
	ret := C.get_token_embeddings(params, state, myArray, C.int(len(tokens)), (*C.float)(&floats[0]))
	C.free(unsafe.Pointer(myArray))
	C.llama_free_params(params)
	if ret != 0 {
		return floats, fmt.Errorf("embedding inference failed")
	}
	return floats, nil
}

// Embeddings
func (l *LLama) Embeddings(text string, opts ...PredictOption) ([]float32, error) {
	state, unlock, err := l.lockState()
	if err != nil {
		return nil, err
	}
	defer unlock()

	if !l.embeddings {
		return []float32{}, fmt.Errorf("model loaded without embeddings")
	}

	po := NewPredictOptions(opts...)

	input := C.CString(text)
	defer C.free(unsafe.Pointer(input))
	nEmbd := int(C.llama_binding_n_embd(state))
	if nEmbd <= 0 {
		return nil, fmt.Errorf("invalid embedding dimension from model")
	}
	floats := make([]float32, nEmbd)
	reverseCount := len(po.StopPrompts)
	reversePrompt := make([]*C.char, reverseCount)
	var pass **C.char
	for i, s := range po.StopPrompts {
		cs := C.CString(s)
		defer C.free(unsafe.Pointer(cs))
		reversePrompt[i] = cs
		pass = &reversePrompt[0]
	}

	var cstrs []unsafe.Pointer
	defer func() { freeCStrings(cstrs) }()

	params := allocatePredictParams(input, po, pass, reverseCount, &cstrs)

	ret := C.get_embeddings(params, state, (*C.float)(&floats[0]))
	C.llama_free_params(params)
	if ret != 0 {
		return floats, fmt.Errorf("embedding inference failed")
	}

	return floats, nil
}

func (l *LLama) Eval(text string, opts ...PredictOption) error {
	state, unlock, err := l.lockState()
	if err != nil {
		return err
	}
	defer unlock()

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

	var cstrs []unsafe.Pointer
	defer func() { freeCStrings(cstrs) }()

	params := allocatePredictParams(input, po, pass, reverseCount, &cstrs)
	ret := C.eval(params, state, input)
	C.llama_free_params(params)
	if ret != 0 {
		return fmt.Errorf("inference failed")
	}

	return nil
}

func predictOutputBufSize(nTokens int) int {
	if nTokens <= 0 {
		nTokens = 1024
	}
	size := nTokens * 32
	if size < 65536 {
		size = 65536
	}
	const maxBuf = 4 * 1024 * 1024
	if size > maxBuf {
		size = maxBuf
	}
	return size
}

func (l *LLama) SpeculativeSampling(ll *LLama, text string, opts ...PredictOption) (string, error) {
	targetState, draftState, unlock, err := lockTwoStates(l, ll)
	if err != nil {
		return "", err
	}
	defer unlock()

	po := NewPredictOptions(opts...)

	if po.TokenCallback != nil {
		setCallback(targetState, po.TokenCallback)
	}

	input := C.CString(text)
	defer C.free(unsafe.Pointer(input))
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	out := make([]byte, predictOutputBufSize(po.Tokens))

	reverseCount := len(po.StopPrompts)
	reversePrompt := make([]*C.char, reverseCount)
	var pass **C.char
	for i, s := range po.StopPrompts {
		cs := C.CString(s)
		defer C.free(unsafe.Pointer(cs))
		reversePrompt[i] = cs
		pass = &reversePrompt[0]
	}

	var cstrs []unsafe.Pointer
	defer func() { freeCStrings(cstrs) }()

	params := allocatePredictParams(input, po, pass, reverseCount, &cstrs)
	ret := C.speculative_sampling(params, targetState, draftState, (*C.char)(unsafe.Pointer(&out[0])), C.int(len(out)), C.bool(po.DebugMode))
	C.llama_free_params(params)

	if po.TokenCallback != nil {
		setCallback(targetState, nil)
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
	state, unlock, err := l.lockState()
	if err != nil {
		return "", err
	}
	defer unlock()

	po := NewPredictOptions(opts...)

	if po.TokenCallback != nil {
		setCallback(state, po.TokenCallback)
	}

	input := C.CString(text)
	defer C.free(unsafe.Pointer(input))
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	out := make([]byte, predictOutputBufSize(po.Tokens))

	reverseCount := len(po.StopPrompts)
	reversePrompt := make([]*C.char, reverseCount)
	var pass **C.char
	for i, s := range po.StopPrompts {
		cs := C.CString(s)
		defer C.free(unsafe.Pointer(cs))
		reversePrompt[i] = cs
		pass = &reversePrompt[0]
	}

	var cstrs []unsafe.Pointer
	defer func() { freeCStrings(cstrs) }()

	params := allocatePredictParams(input, po, pass, reverseCount, &cstrs)
	ret := C.llama_predict(params, state, (*C.char)(unsafe.Pointer(&out[0])), C.int(len(out)), C.bool(po.DebugMode))
	C.llama_free_params(params)

	if po.TokenCallback != nil {
		setCallback(state, nil)
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
	state, unlock, err := l.lockState()
	if err != nil {
		return 0, nil, err
	}
	defer unlock()

	po := NewPredictOptions(opts...)

	input := C.CString(text)
	defer C.free(unsafe.Pointer(input))
	if po.Tokens == 0 {
		po.Tokens = 4096
	}
	out := make([]C.int, po.Tokens)

	var fakeDblPtr **C.char

	var cstrs []unsafe.Pointer
	defer func() { freeCStrings(cstrs) }()

	params := allocatePredictParams(input, po, fakeDblPtr, 0, &cstrs)

	tokRet := C.llama_tokenize_string(params, state, (*C.int)(unsafe.Pointer(&out[0])))
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
	state, unlock, err := l.lockState()
	if err != nil {
		return
	}
	defer unlock()

	setCallback(state, callback)
}

var (
	m         sync.RWMutex
	callbacks = map[uintptr]func(string) bool{}
)

//export tokenCallback
func tokenCallback(statePtr unsafe.Pointer, token *C.char) (keepGoing bool) {
	keepGoing = true
	defer func() {
		if recover() != nil {
			keepGoing = false
		}
	}()

	m.RLock()
	callback := callbacks[uintptr(statePtr)]
	m.RUnlock()

	if callback != nil {
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
