#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <vector>
#include <string> 
#include <fstream> 
#include <random> 
#include <algorithm>
#include <iostream>
#include <numeric> 
#include <stdexcept>


struct Matrix {
    int rows;
    int cols;
    std::vector<float> data;

    Matrix() = default;

    Matrix(int r, int c):
        rows(r), cols(c), data(r * c) {}
};


Matrix matmul(const Matrix& A, const Matrix& B){
    Matrix C(A.rows, B.cols);

    if (A.cols != B.rows){
        throw std::runtime_error("matrix dimensions doesn't match");
    }

    for (int row = 0; row < A.rows; row++){
        for (int col = 0; col < B.cols; col++){
            float sum = 0.0f;
            for (int i = 0; i < A.cols; i++){
                sum += A.data[row * A.cols + i] * B.data[i * B.cols + col];
            }
            C.data[row * C.cols + col] = sum;
        }
    }
    return C;
}

Matrix split_head(const Matrix& A, int head, int n_heads) {
    int head_dim = A.cols / n_heads;
    int head_offset = head *  head_dim;

    Matrix A_head(A.rows, head_dim);

    for (int row = 0; row < A.rows; row++) {
        for (int i = 0; i < head_dim; i++) {
            A_head.data[row * head_dim + i] = 
                A.data[row * A.cols + head_offset + i];
        }
    }
    return A_head;
}

Matrix transpose(const Matrix& A) {
    Matrix B(A.cols, A.rows);

    for (int row = 0; row < A.rows; row++) {
        for (int col = 0; col < A.cols; col++) {
            B.data[col * B.cols + row] =
                A.data[row * A.cols + col];
        }
    }

    return B;
}


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
    Matrix wq;
    Matrix wk;
    Matrix wv;
    Matrix wo;

    // Feed Forward
    Matrix w_fc;
    std::vector<float> b_fc;
    Matrix w_proj;
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

void init_weights(Matrix& weights) {
    for (float& x: weights.data) {
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


        model.layers[i].wq = Matrix(model.hparams.n_embd, model.hparams.n_embd);
        model.layers[i].wk = Matrix(model.hparams.n_embd, model.hparams.n_embd);
        model.layers[i].wv = Matrix(model.hparams.n_embd, model.hparams.n_embd);
        model.layers[i].wo = Matrix(model.hparams.n_embd, model.hparams.n_embd);

        model.layers[i].w_fc = Matrix(model.hparams.n_embd, 4 * model.hparams.n_embd);
        model.layers[i].b_fc.resize(4 * model.hparams.n_embd);

        model.layers[i].w_proj = Matrix(4 * model.hparams.n_embd, model.hparams.n_embd);
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


Matrix softmax(const Matrix& scores) {
    Matrix output(scores.rows, scores.cols);

    for (int row = 0; row < scores.rows; row++){
        float max_val = scores.data[row * scores.cols];
        for (int col = 0; col < scores.cols; col++){
            max_val = std::max(max_val, scores.data[row * scores.cols + col]);
        }

        for (int col = 0; col < scores.cols; col++){
            output.data[row * output.cols + col] = scores.data[row * scores.cols + col] - max_val;
        }

        for (int col = 0; col < scores.cols; col++){
            output.data[row * output.cols + col] = std::exp(output.data[row * output.cols + col]);
        }

        float sum = 0.0f;
        for (int col = 0; col < scores.cols; col++) {
            sum += output.data[row * output.cols + col];
        }

        for (int col = 0; col < scores.cols; col++) {
            output.data[row * output.cols + col] /= sum;
        }
    }

    return output;
}

Matrix self_attention(
        const Matrix& input,
        const mgpt2_layer& layer,
        int n_embds,
        int n_heads
    ) {

    Matrix Q = matmul(input, layer.wq);
    Matrix K = matmul(input, layer.wk);
    Matrix V = matmul(input, layer.wv);

     Matrix combined(input.rows, n_embds);

    for (int head = 0; head < n_heads; head++){

        Matrix Q_head = split_head(Q, head, n_heads);
        Matrix K_head = split_head(K, head, n_heads);
        Matrix V_head = split_head(V, head, n_heads);

        Matrix K_head_T = transpose(K_head);
        Matrix scores = matmul(Q_head, K_head_T);

        int head_dim = n_embds / n_heads;
        float scale = std::sqrt(static_cast<float>(head_dim));

        for (float& x : scores.data) {
            x /= scale;
        }

        for (int row = 0; row < scores.rows; row++){
            for (int col = 0; col < scores.cols; col++){
                if (col > row) {
                    scores.data[row * scores.cols + col] = -INFINITY;
                }
            }
        }

        Matrix attention_weights = softmax(scores);

        Matrix at_out = matmul(attention_weights, V_head);

        int head_offset = head * head_dim;
       
        for (int row = 0; row < at_out.rows; row++) {
            for (int i = 0; i < head_dim; i++) {
                combined.data[row * combined.cols + head_offset + i] =
                    at_out.data[row * at_out.cols + i];
            }
        }
    }

    Matrix projected = matmul(combined, layer.wo);
    
    return projected;
}




int main() {
    mgpt2_model model;

    initialize_model(model);

    int num_tokens = 5;

    Matrix input(
        num_tokens,
        model.hparams.n_embd
    );

    // Fill input with random values
    for (float& x : input.data) {
        x = dist(gen);
    }

    Matrix output = self_attention(
        input,
        model.layers[0],
        model.hparams.n_embd,
        model.hparams.n_heads
    );

    std::cout << "\nSelf-attention test complete!\n";



    return 0;

}
