#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>
#include <string> 
#include <fstream> 
#include <random> 
#include <algorithm>
#include <iostream>
#include <numeric> 

// define hparams for miniGPT
struct mgpt2_hparams {
    int32_t n_vocab  = 1000;
    int32_t n_embd   = 128;
    int32_t n_ctx    = 64;
    int32_t n_heads  = 4;
    int32_t n_layers = 2;
    float   eps      = 1e-5f;
};


struct mgpt2_layer {
    // Layer Normalization
    std::vector<float> ln_1_g;
    std::vector<float> ln_1_b;

    std::vector<float> ln_2_g;
    std::vector<float> ln_2_b;

    // Attention 
    std::vector<float> wq;
    std::vector<float> wk;
    std::vector<float> wv;
    std::vector<float> wo;

    // Feed Forward
    std::vector<float> w_fc;
    std::vector<float> b_fc;
    std::vector<float> w_proj;
    std::vector<float> b_proj;
    
};

struct mgpt2_model {
    mgpt2_hparams hparams;

    //normalization 
    std::vector<float> ln_f_g;
    std::vector<float> ln_f_b;

    std::vector<float> wte; // token embedding
    std::vector<float> wpe; // positional embedding
    std::vector<float> lm_head; // language model head 
    
    std::vector<mgpt2_layer> layers;


};


std::random_device rd;
std::mt19937 gen(rd());

std::normal_distribution<float> dist(0.0f, 0.02f);

void init_weights(std::vector<float>& weights) {
    for (float& x: weights) {
        x = dist(gen);
    }
}

void initialize_model(mgpt2_model& model) {

    // token embedding 
    model.wte.resize(model.hparams.n_vocab * model.hparams.n_embd);

    // pos embedding
    model.wpe.resize(model.hparams.n_ctx * model.hparams.n_embd);

    // transformer layers 
    model.layers.resize(model.hparams.n_layers);

    for (int i = 0; i < model.hparams.n_layers; i++) {
        model.layers[i].ln_1_g.resize(model.hparams.n_embd);
        model.layers[i].ln_1_b.resize(model.hparams.n_embd);

        model.layers[i].ln_2_g.resize(model.hparams.n_embd);
        model.layers[i].ln_2_b.resize(model.hparams.n_embd);


        model.layers[i].wq.resize(model.hparams.n_embd * model.hparams.n_embd);
        model.layers[i].wk.resize(model.hparams.n_embd * model.hparams.n_embd);
        model.layers[i].wv.resize(model.hparams.n_embd * model.hparams.n_embd);
        model.layers[i].wo.resize(model.hparams.n_embd * model.hparams.n_embd);

        model.layers[i].w_fc.resize(model.hparams.n_embd * 4 * model.hparams.n_embd);
        model.layers[i].b_fc.resize(4 * model.hparams.n_embd);

        model.layers[i].w_proj.resize(4 * model.hparams.n_embd * model.hparams.n_embd);
        model.layers[i].b_proj.resize(model.hparams.n_embd);

    }

    // layer normalization
    model.ln_f_g.resize(model.hparams.n_embd);
    model.ln_f_b.resize(model.hparams.n_embd);

    // language model head 
    model.lm_head.resize(model.hparams.n_embd * model.hparams.n_vocab);


    // intialize weights 
    init_weights(model.wte);
    init_weights(model.wpe); 
    init_weights(model.lm_head);

    for (int i = 0; i < model.hparams.n_layers; i++) {
    
        init_weights(model.layers[i].wq);
        init_weights(model.layers[i].wk);
        init_weights(model.layers[i].wv);
        init_weights(model.layers[i].wo);

        init_weights(model.layers[i].w_fc);
        init_weights(model.layers[i].w_proj);
    }


    for (int i = 0; i < model.hparams.n_layers; i++) {
        std::fill(model.layers[i].ln_1_g.begin(), model.layers[i].ln_1_g.end(), 1.0f);
        std::fill(model.layers[i].ln_1_b.begin(), model.layers[i].ln_1_b.end(), 0.0f);

        std::fill(model.layers[i].ln_2_g.begin(), model.layers[i].ln_2_g.end(), 1.0f);
        std::fill(model.layers[i].ln_2_b.begin(), model.layers[i].ln_2_b.end(), 0.0f);
    }

}


std::vector<float> get_embeddings(const mgpt2_model& model, const std::vector<int>& tokens) {

    std::vector<float> final_embeddings;
    final_embeddings.resize(tokens.size() * model.hparams.n_embd);

    for (int pos = 0; pos < tokens.size(); pos++ ) {
        // first index of embedding
        int tok_idx = tokens[pos] * model.hparams.n_embd;
        int pos_idx = pos * model.hparams.n_embd;

        for (int i = 0; i < model.hparams.n_embd; i++ ) {
            final_embeddings[pos * model.hparams.n_embd + i] = 
                model.wte[tok_idx + i] + 
                model.wpe[pos_idx + i];
        }
    }
    return final_embeddings;
}

// tokenizer
std::vector<int> tokenize(const std::string& text) {
    std::vector<int> tokens;

    for (char x : text) {        
        int token_id;
        if (x == ' ') {
            token_id = 26;
        } else {
            token_id = x - 'a';
        }
        tokens.push_back(token_id);
    }
    return tokens;
}

// layernorm
std::vector<float> layer_norm(
        const std::vector<float>& input, 
        const std::vector<float>& gamma, 
        const std::vector<float>& beta, 
        int n_embd,
        float eps
        ){

    int num_tokens = input.size() / n_embd;
    std::vector<float> output(input.size());
    
    for (int pos = 0; pos < num_tokens; pos ++){
        // mean 
        float sum = 0.0f;
        for (int i = 0; i < n_embd; i++){
            sum += input[pos * n_embd + i];
        }

        float mean = sum / n_embd;

        //variance
        float variance = 0.0f;

        for (int i = 0; i < n_embd; i++){
            float diff = input[pos * n_embd + i] - mean;
            variance += diff * diff;
        }
        variance /= n_embd;


        //normalize 
        for (int i = 0; i < n_embd; i++){
            float normalize = (input[pos * n_embd + i] - mean) / std::sqrt(variance + eps);
            output[pos * n_embd + i] = normalize * gamma[i] + beta[i];
        }
    } 

    return output;

}




int main() {
    mgpt2_model model;

    initialize_model(model);

    /** std::string text;
    std::getline(std::cin, text);

    std::vector<int> tokens = tokenize(text); 
    
    std::cout << "Number of tokens: " << tokens.size() << "\n";
    for (int token: tokens) {
        std::cout<< token << " ";
    }
    std::vector<int> tokens = {2, 5};

    std::vector<float> embeddings = get_embeddings(model, tokens);

    for (float x: embeddings) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    **/ 

    std::vector<float> input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        10.0f, 20.0f, 30.0f, 40.0f
    };

    // LayerNorm parameters
    std::vector<float> gamma(4, 1.0f);
    std::vector<float> beta(4, 0.0f);

    float eps = 1e-5f;

    std::vector<float> output =
        layer_norm(input, gamma, beta, 4, eps);

    // Print results
    for (int pos = 0; pos < 2; pos++) {
        std::cout << "Token " << pos << ": ";

        for (int i = 0; i < 4; i++) {
            std::cout << output[pos * 4 + i] << " ";
        }

        std::cout << "\n";
    }

    return 0;
}
