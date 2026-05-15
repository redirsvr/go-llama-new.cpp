package main

import (
    "bufio"
    "fmt"
    "os"
    "strings"

    llama "go-llama-new.cpp"
)

func main() {
    if len(os.Args) < 2 {
        fmt.Fprintf(os.Stderr, "usage: %s <model.gguf> [prompt]\n", os.Args[0])
        os.Exit(1)
    }

    modelPath := os.Args[1]
    prompt := "Hello"
    if len(os.Args) > 2 {
        prompt = strings.Join(os.Args[2:], " ")
    }

    l, err := llama.New(modelPath, llama.SetContext(512), llama.SetGPULayers(0))
    if err != nil {
        fmt.Fprintf(os.Stderr, "load model: %v\n", err)
        os.Exit(1)
    }
    defer l.Free()

    out, err := l.Predict(prompt, llama.SetTokens(64), llama.SetThreads(4))
    if err != nil {
        fmt.Fprintf(os.Stderr, "predict: %v\n", err)
        os.Exit(1)
    }

    fmt.Println(out)

    reader := bufio.NewReader(os.Stdin)
    fmt.Print("\nТокенизация (введите текст): ")
    line, _ := reader.ReadString('\n')
    _, tokens, err := l.TokenizeString(strings.TrimSpace(line))
    if err != nil {
        fmt.Fprintf(os.Stderr, "tokenize: %v\n", err)
        return
    }
    fmt.Printf("токенов: %d, ids: %v\n", len(tokens), tokens)
}
