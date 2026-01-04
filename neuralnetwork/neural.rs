use std::f32::consts::PI;

/// Represents a tensor with shape and data
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    /// Create a new tensor with given shape
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            shape,
            data: vec![0.0; size],
        }
    }

    /// Create a tensor with random initialization
    pub fn random(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let mut tensor = Tensor { shape, data: vec![0.0; size] };
        
        for i in 0..size {
            tensor.data[i] = (i as f32).sin() * 0.1;
        }
        tensor
    }

    /// Get size of tensor
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Reshape tensor
    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(self.size(), new_size, "Shape mismatch in reshape");
        self.shape = new_shape;
    }
}

/// Layer Normalization for stabilizing training
pub struct LayerNorm {
    gamma: Tensor,
    beta: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        LayerNorm {
            gamma: {
                let mut t = Tensor::new(vec![hidden_size]);
                for i in 0..hidden_size {
                    t.data[i] = 1.0;
                }
                t
            },
            beta: Tensor::new(vec![hidden_size]),
            eps,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let batch_size = x.shape[0];
        let hidden_size = x.shape[1];
        let mut output = x.clone();

        for b in 0..batch_size {
            let mut mean = 0.0;
            for h in 0..hidden_size {
                mean += x.data[b * hidden_size + h];
            }
            mean /= hidden_size as f32;

            let mut variance = 0.0;
            for h in 0..hidden_size {
                let diff = x.data[b * hidden_size + h] - mean;
                variance += diff * diff;
            }
            variance /= hidden_size as f32;

            for h in 0..hidden_size {
                let idx = b * hidden_size + h;
                let normalized = (x.data[idx] - mean) / (variance.sqrt() + self.eps);
                output.data[idx] = normalized * self.gamma.data[h] + self.beta.data[h];
            }
        }
        output
    }
}

/// Positional Encoding for Transformer
pub struct PositionalEncoding {
    pe: Tensor,
}

impl PositionalEncoding {
    pub fn new(max_seq_len: usize, d_model: usize) -> Self {
        let mut pe = Tensor::new(vec![max_seq_len, d_model]);

        for pos in 0..max_seq_len {
            for i in 0..d_model {
                let div_term = (10000.0 as f32).powf((2.0 * (i / 2) as f32) / d_model as f32);
                if i % 2 == 0 {
                    pe.data[pos * d_model + i] = ((pos as f32) / div_term).sin();
                } else {
                    pe.data[pos * d_model + i] = ((pos as f32) / div_term).cos();
                }
            }
        }

        PositionalEncoding { pe }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let seq_len = x.shape[1];
        let d_model = x.shape[2];
        let mut output = x.clone();

        for s in 0..seq_len {
            for d in 0..d_model {
                if s < self.pe.shape[0] {
                    output.data[s * d_model + d] += self.pe.data[s * d_model + d];
                }
            }
        }
        output
    }
}

/// Multi-Head Attention mechanism
pub struct MultiHeadAttention {
    num_heads: usize,
    d_model: usize,
    d_k: usize,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
}

impl MultiHeadAttention {
    pub fn new(num_heads: usize, d_model: usize) -> Self {
        let d_k = d_model / num_heads;
        
        MultiHeadAttention {
            num_heads,
            d_model,
            d_k,
            w_q: Tensor::random(vec![d_model, d_model]),
            w_k: Tensor::random(vec![d_model, d_model]),
            w_v: Tensor::random(vec![d_model, d_model]),
            w_o: Tensor::random(vec![d_model, d_model]),
        }
    }

    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
        let seq_len = query.shape[1];
        let batch_size = query.shape[0];
        let mut output = Tensor::new(vec![batch_size, seq_len, self.d_model]);

        // Simplified attention mechanism
        for b in 0..batch_size {
            for q in 0..seq_len {
                let mut attention_scores = vec![0.0; seq_len];
                let mut score_sum = 0.0;

                for k in 0..seq_len {
                    let mut score = 0.0;
                    for d in 0..self.d_k.min(self.d_model) {
                        let q_idx = b * seq_len * self.d_model + q * self.d_model + d;
                        let k_idx = b * seq_len * self.d_model + k * self.d_model + d;
                        if q_idx < query.data.len() && k_idx < key.data.len() {
                            score += query.data[q_idx] * key.data[k_idx];
                        }
                    }
                    score /= (self.d_k as f32).sqrt();
                    score = score.exp();
                    attention_scores[k] = score;
                    score_sum += score;
                }

                // Normalize scores and compute attention output
                for d in 0..self.d_model {
                    let mut attn_out = 0.0;
                    for k in 0..seq_len {
                        let weight = attention_scores[k] / score_sum.max(1e-8);
                        let v_idx = b * seq_len * self.d_model + k * self.d_model + d;
                        if v_idx < value.data.len() {
                            attn_out += weight * value.data[v_idx];
                        }
                    }
                    let out_idx = b * seq_len * self.d_model + q * self.d_model + d;
                    if out_idx < output.data.len() {
                        output.data[out_idx] = attn_out;
                    }
                }
            }
        }

        output
    }
}

/// Feed-Forward Network
pub struct FeedForward {
    w1: Tensor,
    w2: Tensor,
    bias1: Tensor,
    bias2: Tensor,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        FeedForward {
            w1: Tensor::random(vec![d_model, d_ff]),
            w2: Tensor::random(vec![d_ff, d_model]),
            bias1: Tensor::new(vec![d_ff]),
            bias2: Tensor::new(vec![d_model]),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let batch_size = x.shape[0];
        let seq_len = x.shape[1];
        let d_model = x.shape[2];
        let d_ff = self.w1.shape[1];

        // First layer with ReLU activation
        let mut hidden = Tensor::new(vec![batch_size, seq_len, d_ff]);
        for b in 0..batch_size {
            for s in 0..seq_len {
                for f in 0..d_ff {
                    let mut sum = self.bias1.data[f];
                    for d in 0..d_model {
                        let x_idx = b * seq_len * d_model + s * d_model + d;
                        let w_idx = d * d_ff + f;
                        if x_idx < x.data.len() && w_idx < self.w1.data.len() {
                            sum += x.data[x_idx] * self.w1.data[w_idx];
                        }
                    }
                    hidden.data[b * seq_len * d_ff + s * d_ff + f] = sum.max(0.0); // ReLU
                }
            }
        }

        // Second layer
        let mut output = Tensor::new(vec![batch_size, seq_len, d_model]);
        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..d_model {
                    let mut sum = self.bias2.data[d];
                    for f in 0..d_ff {
                        let h_idx = b * seq_len * d_ff + s * d_ff + f;
                        let w_idx = f * d_model + d;
                        if h_idx < hidden.data.len() && w_idx < self.w2.data.len() {
                            sum += hidden.data[h_idx] * self.w2.data[w_idx];
                        }
                    }
                    output.data[b * seq_len * d_model + s * d_model + d] = sum;
                }
            }
        }

        output
    }
}

/// Transformer Block (Attention + Feed-Forward)
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        TransformerBlock {
            attention: MultiHeadAttention::new(num_heads, d_model),
            feed_forward: FeedForward::new(d_model, d_ff),
            norm1: LayerNorm::new(d_model, 1e-6),
            norm2: LayerNorm::new(d_model, 1e-6),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Self-attention with residual connection
        let attn_out = self.attention.forward(x, x, x);
        let mut attn_normalized = Tensor::new(vec![x.shape[0], x.shape[1], x.shape[2]]);
        
        for i in 0..x.data.len() {
            attn_normalized.data[i] = x.data[i] + attn_out.data[i];
        }
        
        let attn_norm = self.norm1.forward(&attn_normalized);

        // Feed-forward with residual connection
        let ff_out = self.feed_forward.forward(&attn_norm);
        let mut output = Tensor::new(vec![x.shape[0], x.shape[1], x.shape[2]]);
        
        for i in 0..attn_norm.data.len() {
            output.data[i] = attn_norm.data[i] + ff_out.data[i];
        }
        
        self.norm2.forward(&output)
    }
}

/// Deep Neural Network for LLM
pub struct DeepNeuralNetworkLLM {
    embedding: Tensor,
    pos_encoding: PositionalEncoding,
    transformer_blocks: Vec<TransformerBlock>,
    output_projection: Tensor,
    vocab_size: usize,
    d_model: usize,
    num_layers: usize,
}

impl DeepNeuralNetworkLLM {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        d_ff: usize,
        max_seq_len: usize,
    ) -> Self {
        let mut blocks = Vec::new();
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new(d_model, num_heads, d_ff));
        }

        DeepNeuralNetworkLLM {
            embedding: Tensor::random(vec![vocab_size, d_model]),
            pos_encoding: PositionalEncoding::new(max_seq_len, d_model),
            transformer_blocks: blocks,
            output_projection: Tensor::random(vec![d_model, vocab_size]),
            vocab_size,
            d_model,
            num_layers,
        }
    }

    /// Embed input tokens
    fn embed_tokens(&self, token_ids: &[usize]) -> Tensor {
        let seq_len = token_ids.len();
        let mut embeddings = Tensor::new(vec![1, seq_len, self.d_model]);

        for (s, &token_id) in token_ids.iter().enumerate() {
            if token_id < self.vocab_size {
                for d in 0..self.d_model {
                    embeddings.data[s * self.d_model + d] = 
                        self.embedding.data[token_id * self.d_model + d];
                }
            }
        }

        embeddings
    }

    /// Forward pass through the deep network
    pub fn forward(&self, token_ids: &[usize]) -> Tensor {
        // Embedding
        let mut x = self.embed_tokens(token_ids);

        // Add positional encoding
        x = self.pos_encoding.forward(&x);

        // Pass through transformer blocks
        for block in &self.transformer_blocks {
            x = block.forward(&x);
        }

        // Output projection to vocabulary
        let seq_len = x.shape[1];
        let mut logits = Tensor::new(vec![1, seq_len, self.vocab_size]);

        for s in 0..seq_len {
            for v in 0..self.vocab_size {
                let mut sum = 0.0;
                for d in 0..self.d_model {
                    let x_idx = s * self.d_model + d;
                    let w_idx = d * self.vocab_size + v;
                    if x_idx < x.data.len() && w_idx < self.output_projection.data.len() {
                        sum += x.data[x_idx] * self.output_projection.data[w_idx];
                    }
                }
                logits.data[s * self.vocab_size + v] = sum;
            }
        }

        logits
    }

    /// Get model configuration
    pub fn config(&self) -> String {
        format!(
            "DeepNeuralNetworkLLM Config:\n\
             - Vocabulary Size: {}\n\
             - Model Dimension: {}\n\
             - Number of Layers: {}\n\
             - Total Parameters: ~{}",
            self.vocab_size,
            self.d_model,
            self.num_layers,
            self.vocab_size * self.d_model + self.num_layers * self.d_model * 4
        )
    }

    /// Forward pass with argmax decoding
    pub fn predict_next_token(&self, token_ids: &[usize]) -> usize {
        let logits = self.forward(token_ids);
        let last_pos_logits = &logits.data[(logits.shape[1] - 1) * self.vocab_size..];
        
        let mut max_idx = 0;
        let mut max_val = last_pos_logits[0];
        for (i, &val) in last_pos_logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        max_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::new(vec![2, 3]);
        assert_eq!(tensor.size(), 6);
    }

    #[test]
    fn test_llm_forward() {
        let llm = DeepNeuralNetworkLLM::new(100, 64, 4, 2, 256, 128);
        let tokens = vec![1, 2, 3, 4, 5];
        let output = llm.forward(&tokens);
        assert_eq!(output.shape[1], 5);
        assert_eq!(output.shape[2], 100);
    }

    #[test]
    fn test_transformer_block() {
        let block = TransformerBlock::new(64, 4, 256);
        let x = Tensor::new(vec![1, 5, 64]);
        let output = block.forward(&x);
        assert_eq!(output.shape, vec![1, 5, 64]);
    }
}
