#include "binding.h"

#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

struct llama_binding_state {
    common_init_result_ptr init;
    llama_model *           model = nullptr;
    llama_context *         ctx   = nullptr;
    common_sampler *        smpl  = nullptr;
    bool                    embeddings = false;
};

static llama_binding_state * binding_state(void * state_pr) {
    return static_cast<llama_binding_state *>(state_pr);
}

static void parse_tensor_split(const char * tensorsplit, float * out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = 0.0f;
    }
    if (tensorsplit == nullptr || tensorsplit[0] == '\0') {
        return;
    }
    std::string arg_next = tensorsplit;
    const std::regex regex{R"([,/]+)"};
    std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
    std::vector<std::string> split_arg{it, {}};
    for (size_t i = 0; i < split_arg.size() && i < n; ++i) {
        out[i] = std::stof(split_arg[i]);
    }
}

static void apply_model_load_options(
        common_params & params,
        int n_ctx,
        int n_seed,
        bool memory_f16,
        bool mlock,
        bool embeddings,
        bool mmap,
        int n_gpu,
        int n_batch,
        const char * maingpu,
        const char * tensorsplit,
        bool numa,
        float rope_freq_base,
        float rope_freq_scale,
        const char * lora,
        const char * lora_base,
        bool perplexity) {
    (void) lora_base;

    if (n_ctx > 0) {
        params.n_ctx = n_ctx;
    }
    if (n_seed >= 0) {
        params.sampling.seed = (uint32_t) n_seed;
    }
    params.use_mlock    = mlock;
    params.embedding    = embeddings;
    params.use_mmap     = mmap;
    params.n_gpu_layers = n_gpu;
    params.n_batch      = n_batch > 0 ? n_batch : params.n_batch;
    params.n_ubatch     = std::min(params.n_batch, params.n_ubatch);
    params.numa         = numa ? GGML_NUMA_STRATEGY_DISTRIBUTE : GGML_NUMA_STRATEGY_DISABLED;
    params.warmup       = false;
    // auto/all (-1/-2): подобрать слои и контекст под свободную VRAM (как ollama)
    params.fit_params   = (n_gpu < 0);

    if (rope_freq_base > 0.0f) {
        params.rope_freq_base = rope_freq_base;
    }
    if (rope_freq_scale > 0.0f) {
        params.rope_freq_scale = rope_freq_scale;
    }

    if (memory_f16) {
        params.cache_type_k = GGML_TYPE_F16;
        params.cache_type_v = GGML_TYPE_F16;
    }

    if (maingpu != nullptr && maingpu[0] != '\0') {
        params.main_gpu = std::stoi(maingpu);
    }

    parse_tensor_split(tensorsplit, params.tensor_split, sizeof(params.tensor_split) / sizeof(params.tensor_split[0]));

    if (perplexity) {
        params.compute_ppl = true;
    }

    if (lora != nullptr && lora[0] != '\0') {
        common_adapter_lora_info la;
        la.path  = lora;
        la.scale = 1.0f;
        params.lora_adapters.push_back(la);
    }
}

static bool check_antiprompt(
        const std::string & output,
        const std::vector<std::string> & antiprompt,
        bool interactive) {
    for (const auto & ap : antiprompt) {
        if (ap.empty()) {
            continue;
        }
        const size_t extra = interactive ? 0 : 2;
        const size_t search_start = output.length() > ap.length() + extra
            ? output.length() - ap.length() - extra
            : 0;
        if (output.find(ap, search_start) != std::string::npos) {
            return true;
        }
    }
    return false;
}

extern "C" {

void * load_model(
        const char * fname,
        int n_ctx,
        int n_seed,
        bool memory_f16,
        bool mlock,
        bool embeddings,
        bool mmap,
        bool low_vram,
        int n_gpu,
        int n_batch,
        const char * maingpu,
        const char * tensorsplit,
        bool numa,
        float rope_freq_base,
        float rope_freq_scale,
        bool mul_mat_q,
        const char * lora,
        const char * lora_base,
        bool perplexity) {
    (void) low_vram;
    (void) mul_mat_q;

    common_init();
    llama_backend_init();

    common_params params;
    params.model.path = fname;

    apply_model_load_options(
            params, n_ctx, n_seed, memory_f16, mlock, embeddings, mmap,
            n_gpu, n_batch, maingpu, tensorsplit, numa,
            rope_freq_base, rope_freq_scale, lora, lora_base, perplexity);

    llama_numa_init(params.numa);

    auto * binding = new llama_binding_state();
    binding->init = common_init_from_params(params);
    if (!binding->init || binding->init->context() == nullptr) {
        delete binding;
        return nullptr;
    }

    binding->model      = binding->init->model();
    binding->ctx        = binding->init->context();
    binding->smpl       = binding->init->sampler(0);
    binding->embeddings = embeddings;

    return binding;
}

void llama_binding_free_model(void * state_pr) {
    delete binding_state(state_pr);
}

int load_state(void * state_pr, char * statefile, char * modes) {
    (void) modes;
    auto * state = binding_state(state_pr);
    if (state == nullptr || state->ctx == nullptr) {
        return 1;
    }

    std::vector<llama_token> tokens(llama_n_ctx(state->ctx));
    size_t n_out = 0;
    if (!llama_state_load_file(state->ctx, statefile, tokens.data(), tokens.size(), &n_out)) {
        return 1;
    }
    return 0;
}

void save_state(void * state_pr, char * dst, char * modes) {
    (void) modes;
    auto * state = binding_state(state_pr);
    if (state == nullptr || state->ctx == nullptr) {
        return;
    }
    llama_state_save_file(state->ctx, dst, nullptr, 0);
}

void * llama_allocate_params(
        const char * prompt,
        int seed,
        int threads,
        int tokens,
        int top_k,
        float top_p,
        float temp,
        float repeat_penalty,
        int repeat_last_n,
        bool ignore_eos,
        bool memory_f16,
        int n_batch,
        int n_keep,
        const char ** antiprompt,
        int antiprompt_count,
        float tfs_z,
        float typical_p,
        float frequency_penalty,
        float presence_penalty,
        int mirostat,
        float mirostat_eta,
        float mirostat_tau,
        bool penalize_nl,
        const char * logit_bias,
        const char * session_file,
        bool prompt_cache_all,
        bool mlock,
        bool mmap,
        const char * maingpu,
        const char * tensorsplit,
        bool prompt_cache_ro,
        const char * grammar,
        float rope_freq_base,
        float rope_freq_scale,
        float negative_prompt_scale,
        const char * negative_prompt,
        int n_draft) {
    (void) tfs_z;
    (void) penalize_nl;
    (void) negative_prompt_scale;
    (void) negative_prompt;
    (void) memory_f16;

    auto * params = new common_params();
    params->prompt = prompt != nullptr ? prompt : "";
    params->n_predict = tokens;
    params->n_batch   = n_batch > 0 ? n_batch : params->n_batch;
    params->n_keep    = n_keep;
    params->use_mlock = mlock;
    params->use_mmap  = mmap;
    params->path_prompt_cache = session_file != nullptr ? session_file : "";
    params->prompt_cache_all  = prompt_cache_all;
    params->prompt_cache_ro   = prompt_cache_ro;

    if (rope_freq_base > 0.0f) {
        params->rope_freq_base = rope_freq_base;
    }
    if (rope_freq_scale > 0.0f) {
        params->rope_freq_scale = rope_freq_scale;
    }

    params->sampling.seed           = seed >= 0 ? (uint32_t) seed : LLAMA_DEFAULT_SEED;
    params->cpuparams.n_threads     = threads > 0 ? threads : 4;
    params->cpuparams_batch.n_threads = params->cpuparams.n_threads;
    params->sampling.top_k          = top_k;
    params->sampling.top_p          = top_p;
    params->sampling.temp           = temp;
    params->sampling.penalty_repeat = repeat_penalty;
    params->sampling.penalty_last_n = repeat_last_n;
    params->sampling.penalty_freq   = frequency_penalty;
    params->sampling.penalty_present = presence_penalty;
    params->sampling.typ_p          = typical_p > 0 ? typical_p : 1.0f;
    params->sampling.mirostat       = mirostat;
    params->sampling.mirostat_eta   = mirostat_eta;
    params->sampling.mirostat_tau   = mirostat_tau;
    params->sampling.ignore_eos     = ignore_eos;

    if (grammar != nullptr && grammar[0] != '\0') {
        params->sampling.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, grammar);
    }

    if (maingpu != nullptr && maingpu[0] != '\0') {
        params->main_gpu = std::stoi(maingpu);
    }
    parse_tensor_split(tensorsplit, params->tensor_split, sizeof(params->tensor_split) / sizeof(params->tensor_split[0]));

    if (antiprompt_count > 0 && antiprompt != nullptr) {
        params->antiprompt = create_vector(antiprompt, antiprompt_count);
    }

    if (logit_bias != nullptr && logit_bias[0] != '\0') {
        std::stringstream ss(logit_bias);
        llama_token key;
        char sign = 0;
        std::string value_str;
        if (ss >> key >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
            params->sampling.logit_bias.push_back({key, std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f)});
        }
    }

    params->speculative.draft.n_max = n_draft > 0 ? n_draft : params->speculative.draft.n_max;

    return params;
}

void llama_free_params(void * params_ptr) {
    delete static_cast<common_params *>(params_ptr);
}

int eval(void * params_ptr, void * state_pr, char * text) {
    auto * params = static_cast<common_params *>(params_ptr);
    auto * state  = binding_state(state_pr);
    if (state == nullptr || state->ctx == nullptr) {
        return 1;
    }

    std::string str = text != nullptr ? text : params->prompt;
    auto embd = common_tokenize(state->ctx, str, true, true);
    if (embd.empty()) {
        return 1;
    }

    int n_past = 0;
    if (!common_prompt_batch_decode(state->ctx, embd, n_past, params->n_batch, "", false)) {
        return 1;
    }
    return 0;
}

int get_embeddings(void * params_ptr, void * state_pr, float * res_embeddings) {
    auto * params = static_cast<common_params *>(params_ptr);
    auto * state  = binding_state(state_pr);
    if (state == nullptr || state->ctx == nullptr || !state->embeddings) {
        return 1;
    }

    auto embd = common_tokenize(state->ctx, params->prompt, true, true);
    if (!embd.empty()) {
        int n_past = 0;
        if (!common_prompt_batch_decode(state->ctx, embd, n_past, params->n_batch, "", false)) {
            return 1;
        }
    }

    const int n_embd = llama_model_n_embd(state->model);
    const float * emb = llama_get_embeddings_ith(state->ctx, -1);
    if (emb == nullptr) {
        emb = llama_get_embeddings(state->ctx);
    }
    if (emb == nullptr) {
        return 1;
    }

    for (int i = 0; i < n_embd; ++i) {
        res_embeddings[i] = emb[i];
    }
    return 0;
}

int get_token_embeddings(void * params_ptr, void * state_pr, int * tokens, int tokenSize, float * res_embeddings) {
    auto * params = static_cast<common_params *>(params_ptr);
    auto * state  = binding_state(state_pr);
    if (state == nullptr || state->ctx == nullptr) {
        return 1;
    }

    std::string text;
    for (int i = 0; i < tokenSize; ++i) {
        text += common_token_to_piece(state->ctx, tokens[i]);
    }
    params->prompt = text;
    return get_embeddings(params_ptr, state_pr, res_embeddings);
}

int llama_tokenize_string(void * params_ptr, void * state_pr, int * result) {
    auto * params = static_cast<common_params *>(params_ptr);
    auto * state  = binding_state(state_pr);
    if (state == nullptr || state->ctx == nullptr) {
        return -1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const int32_t max_tokens = params->n_ctx > 0 ? params->n_ctx : 4096;

    return llama_tokenize(
            vocab,
            params->prompt.c_str(),
            (int32_t) params->prompt.size(),
            reinterpret_cast<llama_token *>(result),
            max_tokens,
            add_bos,
            true);
}

int llama_predict(void * params_ptr, void * state_pr, char * result, int result_cap, bool debug) {
    auto * params = static_cast<common_params *>(params_ptr);
    auto * state  = binding_state(state_pr);
    if (state == nullptr || state->ctx == nullptr || state->smpl == nullptr) {
        return 1;
    }

    llama_context * ctx   = state->ctx;
    llama_model *   model = state->model;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_memory_t mem = llama_get_memory(ctx);

    common_sampler_ptr smpl_ptr(common_sampler_init(model, params->sampling));
    if (!smpl_ptr) {
        return 1;
    }
    common_sampler * smpl = smpl_ptr.get();

    const int n_ctx = llama_n_ctx(ctx);
    if (params->n_predict < 0) {
        params->n_predict = 128;
    }

    llama_set_n_threads(ctx, params->cpuparams.n_threads, params->cpuparams_batch.n_threads);

    std::string path_session = params->path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        session_tokens.resize(n_ctx);
        size_t n_out = 0;
        if (std::ifstream(path_session).good()) {
            llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size(), &n_out);
            session_tokens.resize(n_out);
        }
    }

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> embd_inp = common_tokenize(ctx, params->prompt, add_bos, true);
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_vocab_bos(vocab));
    }

    if ((int) embd_inp.size() > n_ctx - 4) {
        return 1;
    }

    if (params->n_keep < 0 || params->n_keep > (int) embd_inp.size()) {
        params->n_keep = (int) embd_inp.size();
    }

    common_sampler_reset(smpl);

    int n_past             = 0;
    int n_remain           = params->n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;
    bool is_antiprompt     = false;
    bool need_save_session = !path_session.empty() && !params->prompt_cache_ro;

    std::vector<llama_token> embd;
    std::string res;

    while (n_remain > 0 && !is_antiprompt) {
        if (!embd.empty()) {
            const int max_embd_size = n_ctx - 4;
            if ((int) embd.size() > max_embd_size) {
                embd.resize(max_embd_size);
            }

            if (n_past + (int) embd.size() >= n_ctx) {
                const int n_left    = n_past - params->n_keep;
                const int n_discard = n_left / 2;
                llama_memory_seq_rm(mem, 0, params->n_keep, params->n_keep + n_discard);
                llama_memory_seq_add(mem, 0, params->n_keep + n_discard, n_past, -n_discard);
                n_past -= n_discard;
                path_session.clear();
            }

            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for (; i < embd.size(); ++i) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }
                    n_past++;
                    n_session_consumed++;
                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            if (!embd.empty()) {
                const bool save_now = need_save_session && n_consumed >= (int) embd_inp.size();
                if (!common_prompt_batch_decode(ctx, embd, n_past, params->n_batch, path_session, save_now)) {
                    return 1;
                }
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
                need_save_session  = false;
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed) {
            const llama_token id = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, id, true);
            embd.push_back(id);

            auto piece = common_token_to_piece(ctx, id);
            if (!tokenCallback(state_pr, const_cast<char *>(piece.c_str()))) {
                break;
            }

            --n_remain;

            if (llama_vocab_is_eog(vocab, id)) {
                break;
            }
        } else {
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                common_sampler_accept(smpl, embd_inp[n_consumed], false);
                ++n_consumed;
                if ((int) embd.size() >= params->n_batch) {
                    break;
                }
            }
        }

        // Только сгенерированные токены (не промпт) — иначе каждый токен дублировался в res.
        if ((int) embd_inp.size() <= n_consumed) {
            for (const auto id : embd) {
                res += common_token_to_piece(ctx, id);
            }
        }

        if ((int) embd_inp.size() <= n_consumed && !params->antiprompt.empty()) {
            is_antiprompt = check_antiprompt(res, params->antiprompt, false);
        }
    }

    if (!path_session.empty() && params->prompt_cache_all && !params->prompt_cache_ro) {
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    if (debug) {
        common_perf_print(ctx, smpl);
    }

    if (result != nullptr && result_cap > 0) {
        size_t n = res.size();
        size_t maxn = (size_t) result_cap - 1;
        if (n > maxn) {
            n = maxn;
        }
        std::memcpy(result, res.c_str(), n);
        result[n] = '\0';
    }

    return 0;
}

int speculative_sampling(void * params_ptr, void * target_model, void * draft_model, char * result, int result_cap, bool debug) {
    auto * params = static_cast<common_params *>(params_ptr);
    auto * tgt    = binding_state(target_model);
    auto * dft    = binding_state(draft_model);
    if (tgt == nullptr || dft == nullptr || tgt->ctx == nullptr || dft->ctx == nullptr) {
        return 1;
    }

    llama_context * ctx_tgt = tgt->ctx;
    llama_context * ctx_dft = dft->ctx;
    const llama_vocab * vocab = llama_model_get_vocab(tgt->model);

    common_sampler_ptr smpl_ptr(common_sampler_init(tgt->model, params->sampling));
    if (!smpl_ptr) {
        return 1;
    }
    common_sampler * smpl_tgt = smpl_ptr.get();

    auto inp = common_tokenize(ctx_tgt, params->prompt, true, true);
    const int max_tokens = llama_n_ctx(ctx_tgt) - 4;
    if ((int) inp.size() > max_tokens) {
        return 1;
    }

    int n_past_tgt = 0;
    int n_past_dft = 0;
    if (!inp.empty()) {
        if (!common_prompt_batch_decode(ctx_tgt, inp, n_past_tgt, params->n_batch, "", false)) {
            return 1;
        }
        if (!common_prompt_batch_decode(ctx_dft, inp, n_past_dft, params->n_batch, "", false)) {
            return 1;
        }
    }

    const int n_draft = params->speculative.draft.n_max > 0 ? params->speculative.draft.n_max : 16;
    int n_predict = 0;
    std::string res;
    bool has_eos = false;

    std::vector<llama_token> drafted;
    std::vector<llama_token> last_tokens(llama_n_ctx(ctx_tgt), 0);
    for (auto id : inp) {
        last_tokens.erase(last_tokens.begin());
        last_tokens.push_back(id);
    }

    while (n_predict < params->n_predict && !has_eos) {
        int i_dft = 0;
        while (true) {
            const llama_token id = common_sampler_sample(smpl_tgt, ctx_tgt, -1);
            common_sampler_accept(smpl_tgt, id, true);

            last_tokens.erase(last_tokens.begin());
            last_tokens.push_back(id);

            auto piece = common_token_to_piece(ctx_tgt, id);
            if (!tokenCallback(draft_model, const_cast<char *>(piece.c_str()))) {
                break;
            }
            res += piece;

            if (llama_vocab_is_eog(vocab, id)) {
                has_eos = true;
            }

            ++n_predict;

            if (i_dft < (int) drafted.size() && id == drafted[i_dft]) {
                ++i_dft;
                continue;
            }

            llama_token dft_id = id;
            llama_batch batch = llama_batch_get_one(&dft_id, 1);
            if (llama_decode(ctx_dft, batch) != 0) {
                return 1;
            }
            ++n_past_dft;

            drafted.clear();
            drafted.push_back(id);
            break;
        }

        if (n_predict >= params->n_predict || has_eos) {
            break;
        }

        int n_past_cur = n_past_dft;
        for (int i = 0; i < n_draft; ++i) {
            float * logits = llama_get_logits(ctx_dft);
            const int n_vocab = llama_vocab_n_tokens(vocab);

            llama_token draft_id = 0;
            float max_logit = logits[0];
            for (llama_token t = 1; t < n_vocab; ++t) {
                if (logits[t] > max_logit) {
                    max_logit = logits[t];
                    draft_id = t;
                }
            }
            drafted.push_back(draft_id);

            if (i == n_draft - 1) {
                break;
            }

            llama_batch batch = llama_batch_get_one(&draft_id, 1);
            if (llama_decode(ctx_dft, batch) != 0) {
                return 1;
            }
            ++n_past_cur;
        }

        llama_batch batch = llama_batch_get_one(drafted.data(), (int32_t) drafted.size());
        if (llama_decode(ctx_tgt, batch) != 0) {
            return 1;
        }
        ++n_past_tgt;

        if (!drafted.empty()) {
            drafted.erase(drafted.begin());
        }
    }

    if (debug) {
        common_perf_print(ctx_tgt, smpl_tgt);
        common_perf_print(ctx_dft, nullptr);
    }

    if (result != nullptr && result_cap > 0) {
        size_t n = res.size();
        size_t maxn = (size_t) result_cap - 1;
        if (n > maxn) {
            n = maxn;
        }
        std::memcpy(result, res.c_str(), n);
        result[n] = '\0';
    }

    return 0;
}

} // extern "C"

std::vector<std::string> create_vector(const char ** strings, int count) {
    std::vector<std::string> vec;
    for (int i = 0; i < count; ++i) {
        vec.emplace_back(strings[i]);
    }
    return vec;
}

void delete_vector(std::vector<std::string> * vec) {
    delete vec;
}
