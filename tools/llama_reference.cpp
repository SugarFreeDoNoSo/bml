// Independent inference oracle using the pinned upstream llama.cpp API.
// Build against the reference checkout; this is never linked into BML output.
#include "llama.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

struct options {
    std::string model;
    std::string prompt;
    std::string tokens;
    int32_t predict = 4;
    int32_t context = 128;
    bool logits = false;
    bool has_prompt = false;
    bool has_tokens = false;
};

static int32_t integer(const std::string & text) {
    size_t consumed = 0;
    const long long value = std::stoll(text, &consumed);
    if (consumed != text.size() || value < 0 || value > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("invalid nonnegative integer: " + text);
    }
    return static_cast<int32_t>(value);
}

static options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--logits") {
            result.logits = true;
            continue;
        }
        if (i + 1 == argc) {
            throw std::runtime_error("missing value for " + key);
        }
        const std::string value = argv[++i];
        if (key == "--model") result.model = value;
        else if (key == "--prompt") { result.prompt = value; result.has_prompt = true; }
        else if (key == "--tokens") { result.tokens = value; result.has_tokens = true; }
        else if (key == "--predict") result.predict = integer(value);
        else if (key == "--context") result.context = integer(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (result.model.empty() || result.has_prompt == result.has_tokens || result.context == 0) {
        throw std::runtime_error("use --model FILE, either --prompt TEXT or --tokens CSV, and positive --context");
    }
    return result;
}

static std::vector<llama_token> prompt_tokens(const options & opt, const llama_vocab * vocab) {
    std::vector<llama_token> ids;
    if (opt.has_tokens) {
        size_t start = 0;
        while (start <= opt.tokens.size()) {
            const size_t end = opt.tokens.find(',', start);
            ids.push_back(integer(opt.tokens.substr(start, end == std::string::npos ? end : end - start)));
            if (end == std::string::npos) break;
            start = end + 1;
        }
    } else {
        if (opt.prompt.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("prompt is too long");
        }
        const int32_t size = static_cast<int32_t>(opt.prompt.size());
        const int32_t required = llama_tokenize(vocab, opt.prompt.data(), size, nullptr, 0, true, false);
        if (required == std::numeric_limits<int32_t>::min()) {
            throw std::runtime_error("tokenizer output overflows the reference API");
        }
        ids.resize(static_cast<size_t>(required < 0 ? -required : required));
        const int32_t actual = llama_tokenize(vocab, opt.prompt.data(), size, ids.data(),
                                            static_cast<int32_t>(ids.size()), true, false);
        if (actual < 0) throw std::runtime_error("reference tokenization failed");
        ids.resize(static_cast<size_t>(actual));
    }
    if (ids.empty()) throw std::runtime_error("prompt has no tokens");
    for (llama_token id : ids) {
        if (id < 0 || id >= llama_vocab_n_tokens(vocab)) {
            throw std::runtime_error("input token is outside the reference vocabulary");
        }
    }
    if (ids.size() + static_cast<size_t>(opt.predict) > static_cast<size_t>(opt.context)) {
        throw std::runtime_error("prompt and generation exceed context");
    }
    return ids;
}

static void evaluate(llama_context * ctx, llama_batch_ext * batch,
                     const std::vector<llama_token> & tokens, llama_pos first_position) {
    llama_batch_ext_clear(batch);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const int32_t index = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos position = first_position + static_cast<llama_pos>(i);
        if (index < 0 || !llama_batch_ext_set_pos(batch, index, &position)) {
            throw std::runtime_error("reference batch construction failed");
        }
    }
    if (!llama_batch_ext_set_output_logits(batch, static_cast<int32_t>(tokens.size()) - 1, true)) {
        throw std::runtime_error("reference logits request failed");
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
        throw std::runtime_error("reference decode failed");
    }
}

static std::vector<float> logits(llama_context * ctx, int32_t count) {
    const float * source = llama_get_logits_ith(ctx, -1);
    if (!source) throw std::runtime_error("reference returned no logits");
    std::vector<float> values(source, source + count);
    for (float value : values) {
        if (!std::isfinite(value)) throw std::runtime_error("reference produced non-finite logits");
    }
    return values;
}

static std::vector<uint8_t> token_bytes(const llama_vocab * vocab, llama_token id) {
    const int32_t required = llama_token_to_piece(vocab, id, nullptr, 0, 0, false);
    std::vector<char> piece(static_cast<size_t>(required < 0 ? -required : required));
    const int32_t written = llama_token_to_piece(vocab, id, piece.data(), static_cast<int32_t>(piece.size()), 0, false);
    if (written < 0) throw std::runtime_error("reference piece decoding failed");
    return std::vector<uint8_t>(piece.begin(), piece.begin() + written);
}

static json run(const options & opt) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_extra_bufts = false;
    model_params.check_tensors = true;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(opt.model.c_str(), model_params), llama_model_free);
    if (!model) throw std::runtime_error("could not load reference model");
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const auto prompt = prompt_tokens(opt, vocab);

    llama_context_params params = llama_context_default_params();
    params.n_ctx = static_cast<uint32_t>(opt.context);
    params.n_batch = static_cast<uint32_t>(opt.context);
    params.n_ubatch = static_cast<uint32_t>(opt.context);
    params.n_seq_max = 1;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.type_k = GGML_TYPE_F32;
    params.type_v = GGML_TYPE_F32;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.offload_kqv = false;
    params.op_offload = false;
    params.no_perf = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), params), llama_free);
    if (!ctx) throw std::runtime_error("could not create reference context");
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
        llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    if (!batch) throw std::runtime_error("could not create reference batch");

    const auto start = std::chrono::steady_clock::now();
    evaluate(ctx.get(), batch.get(), prompt, 0);
    auto current = logits(ctx.get(), llama_vocab_n_tokens(vocab));
    const auto prefilled = std::chrono::steady_clock::now();
    json rows = json::array();
    if (opt.logits) rows.push_back(current);
    std::vector<llama_token> generated;
    std::vector<uint8_t> bytes;
    for (int32_t step = 0; step < opt.predict; ++step) {
        const llama_token next = static_cast<llama_token>(std::max_element(current.begin(), current.end()) - current.begin());
        generated.push_back(next);
        const auto piece = token_bytes(vocab, next);
        bytes.insert(bytes.end(), piece.begin(), piece.end());
        evaluate(ctx.get(), batch.get(), {next}, static_cast<llama_pos>(prompt.size()) + step);
        current = logits(ctx.get(), llama_vocab_n_tokens(vocab));
        if (opt.logits) rows.push_back(current);
        if (next == llama_vocab_eos(vocab)) break;
    }
    const auto finished = std::chrono::steady_clock::now();
    const std::string text(bytes.begin(), bytes.end());
    return {
        {"prompt_tokens", prompt}, {"tokens", generated}, {"logits", rows},
        {"eos_id", llama_vocab_eos(vocab)},
        {"text", text}, {"bytes", bytes},
        {"timing", {
            {"prefill_ms", std::chrono::duration<double, std::milli>(prefilled - start).count()},
            {"decode_ms", std::chrono::duration<double, std::milli>(finished - prefilled).count()}
        }},
        {"reference", {
            {"threads", 1}, {"context", opt.context}, {"kv_dtype", "f32"},
            {"eos_id", llama_vocab_eos(vocab)},
            {"flash_attention", false}, {"weight_repacking", false},
            {"parse_special", false}, {"add_special", true}
        }}
    };
}

int main(int argc, char ** argv) {
    try {
        const auto opt = parse_options(argc, argv);
        llama_backend_init();
        const auto result = run(opt);
        llama_backend_free();
        // bytes remains exact if the final token ends inside a UTF-8 codepoint.
        std::cout << result.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "llama reference: " << error.what() << '\n';
        return 1;
    }
}
