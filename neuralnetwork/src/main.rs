mod neural;
use neural::DeepNeuralNetworkLLM;

fn main() {
    println!("🧠 Initializing Deep Neural Network LLM...\n");
    
    let llm = DeepNeuralNetworkLLM::new(
        100,    // vocab_size
        64,     // d_model
        4,      // num_heads
        2,      // num_layers
        256,    // d_ff
        128,    // max_seq_len
    );

    println!("{}\n", llm.config());

    // Example: predict next token
    let input_tokens = vec![1, 2, 3, 4, 5];
    println!("Input tokens: {:?}", input_tokens);
    
    let next_token = llm.predict_next_token(&input_tokens);
    println!("Predicted next token: {}\n", next_token);

    // Run forward pass
    let output = llm.forward(&input_tokens);
    println!("Output shape: {:?}", output.shape);
    println!("✅ Neural network forward pass completed successfully!");
}
