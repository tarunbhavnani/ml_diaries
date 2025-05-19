import numpy as np

# 1️⃣ Step 1: Word Embeddings (turn words into vectors)
word_to_vec = {
    "She": np.random.randn(8),  # 8-dimensional vector for each word
    "loves": np.random.randn(8),
    "chocolate": np.random.randn(8)
}

sentence = ["She", "loves", "chocolate"]
embedded_sentence = np.array([word_to_vec[word] for word in sentence])  # Shape: (3, 8)

# 2️⃣ Step 2: Positional Encoding
def positional_encoding(seq_length, d_model):
    pos = np.arange(seq_length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angles = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    pos_encoding = np.zeros((seq_length, d_model))
    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])  # Apply sine to even indices
    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])  # Apply cosine to odd indices
    return pos_encoding

pos_encoding = positional_encoding(3, 8)
input_with_pos = embedded_sentence + pos_encoding  # Adding positional encoding

# 3️⃣ Step 3: Self-Attention Mechanism
def scaled_dot_product_attention(Q, K, V):
    scores = np.matmul(Q, K.T) / np.sqrt(Q.shape[-1])  # Compute attention scores
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)  # Softmax
    return np.matmul(attention_weights, V), attention_weights  # Return new values


# Create Query, Key, and Value matrices
W_q, W_k, W_v = np.random.randn(8, 8), np.random.randn(8, 8), np.random.randn(8, 8)
Q, K, V = input_with_pos @ W_q, input_with_pos @ W_k, input_with_pos @ W_v

# Compute attention output
attn_output, attn_weights = scaled_dot_product_attention(Q, K, V)


# 4️⃣ Step 4: Feed-Forward Network
def feed_forward(x, d_ff=16):
    W1, W2 = np.random.randn(8, d_ff), np.random.randn(d_ff, 8)  # Two layers
    return np.maximum(0, x @ W1) @ W2  # ReLU activation + second linear transformation

ffn_output = feed_forward(attn_output)

# 5️⃣ Step 5: Layer Normalization
def layer_norm(x, epsilon=1e-6):
    mean, std = x.mean(), x.std()
    return (x - mean) / (std + epsilon)

output = layer_norm(ffn_output + attn_output)  # Add & Normalize

print("Final Encoder Output:\n", output)


# =============================================================================
# multihead attention
# =============================================================================

import numpy as np

# 1️⃣ Step 1: Word Embeddings (turn words into vectors)
word_to_vec = {
    "She": np.random.randn(8),
    "loves": np.random.randn(8),
    "chocolate": np.random.randn(8)
}

sentence = ["She", "loves", "chocolate"]
embedded_sentence = np.array([word_to_vec[word] for word in sentence])  # Shape: (3, 8)

# 2️⃣ Step 2: Positional Encoding
def positional_encoding(seq_length, d_model):
    pos = np.arange(seq_length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angles = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    pos_encoding = np.zeros((seq_length, d_model))
    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])  # Sine for even indices
    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])  # Cosine for odd indices
    return pos_encoding

pos_encoding = positional_encoding(3, 8)
input_with_pos = embedded_sentence + pos_encoding  # (3, 8)

# 3️⃣ Step 3: Multi-Head Attention (Multiple attention heads)
def split_into_heads(x, num_heads):
    batch_size, seq_len, d_model = x.shape
    depth = d_model // num_heads  # Size of each head (d_model is divisible by num_heads)
    
    # Reshape the input into (batch_size, num_heads, seq_len, depth)
    x = x.reshape(batch_size, seq_len, num_heads, depth)
    
    # Transpose the shape to (batch_size, num_heads, seq_len, depth)
    return x

def multi_head_attention(Q, K, V, num_heads=2):
    batch_size, seq_len, d_model = Q.shape
    depth = d_model // num_heads

    # Split Q, K, V into multiple heads
    Q = split_into_heads(Q, num_heads)
    K = split_into_heads(K, num_heads)
    V = split_into_heads(V, num_heads)
    
    # Compute attention for each head
    attention_outputs = []
    for i in range(num_heads):
        scores = np.matmul(Q[:, :, i, :], K[:, :, i, :].transpose(0, 1, 3, 2)) / np.sqrt(depth)  # Scaled dot product
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)  # Softmax
        attention_output = np.matmul(attention_weights, V[:, :, i, :])
        attention_outputs.append(attention_output)
    
    # Concatenate outputs from all heads
    attention_output = np.concatenate(attention_outputs, axis=-1)  # Concatenate along depth dimension
    return attention_output

# Create Query, Key, and Value matrices (3D, not 2D!)
W_q, W_k, W_v = np.random.randn(8, 8), np.random.randn(8, 8), np.random.randn(8, 8)
Q, K, V = np.matmul(input_with_pos, W_q), np.matmul(input_with_pos, W_k), np.matmul(input_with_pos, W_v)

# Check the shapes of Q, K, V to confirm they are 3D
print("Q shape:", Q.shape)  # Should be (batch_size, seq_len, d_model)
print("K shape:", K.shape)
print("V shape:", V.shape)

# Apply multi-head attention
multihead_attn_output = multi_head_attention(Q, K, V, num_heads=2)

# 4️⃣ Step 4: Feed-Forward Network
def feed_forward(x, d_ff=16):
    W1, W2 = np.random.randn(8, d_ff), np.random.randn(d_ff, 8)
    return np.maximum(0, x @ W1) @ W2  # ReLU activation

ffn_output = feed_forward(multihead_attn_output)

# 5️⃣ Step 5: Layer Normalization
def layer_norm(x, epsilon=1e-6):
    mean, std = x.mean(), x.std()
    return (x - mean) / (std + epsilon)

output = layer_norm(ffn_output + multihead_attn_output)  # Add & Normalize

print("Final Encoder Output with Multi-Head Attention:\n", output)