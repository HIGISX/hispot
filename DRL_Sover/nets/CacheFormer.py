import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple, Optional, List, Tuple

from nets.FacilitiesRepresentationEncoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.functions import sample_many


def set_decode_type(model, decode_type):
    """
    Set the decode type for a model, handling DataParallel wrapper.
    
    Args:
        model: The model to set decode type for
        decode_type: The type of decoding to use
    """
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached.
    This class allows for efficient indexing of multiple Tensors at once.
    
    Attributes:
        node_embeddings: Precomputed node embeddings
        context_node_projected: Projected context for nodes
        glimpse_key: Keys for glimpse attention mechanism
        glimpse_val: Values for glimpse attention mechanism
        logit_key: Keys for final logit computation
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        """
        Efficient indexing that returns a new AttentionModelFixed with indexed tensors.
        
        Args:
            key: Index or slice to apply to tensors
            
        Returns:
            New AttentionModelFixed with indexed tensors
        """
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return tuple.__getitem__(self, key)


def _get_attention_node_data(fixed):
    """
    Extract attention-related data from fixed context.
    
    Args:
        fixed: AttentionModelFixed context
        
    Returns:
        Tuple of (glimpse_key, glimpse_val, logit_key)
    """
    return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key


class PositionalEncoding(nn.Module):
    """
    Positional encoding module that adds positional information to input embeddings.
    Uses sine and cosine functions of different frequencies to encode position.
    """
    
    def __init__(self, d_model, max_len=50):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model embeddings
            max_len: Maximum sequence length to support
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return x


class CachedTransformerDecoderLayer(nn.Module):
    """
    A Transformer decoder layer that supports key-value caching.
    The forward method accepts and returns a past_key_value tuple for efficient autoregressive generation.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        """
        Initialize cached transformer decoder layer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
        """
        super(CachedTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional key-value caching.
        
        Args:
            tgt: Target sequence tensor
            memory: Memory tensor for cross-attention
            past_key_value: Optional cached key-value pairs from previous steps
            
        Returns:
            Tuple of (output, current_key_value) for caching
        """
        query = tgt

        # Handle key-value caching for autoregressive generation
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], tgt], dim=0)
            value = torch.cat([past_key_value[1], tgt], dim=0)
        else:
            key = value = tgt

        current_key_value = (key, value)

        # Self-attention with residual connection and normalization
        tgt2, _ = self.self_attn(query, key, value)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention with memory
        tgt2, _ = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward network with residual connection and normalization
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, current_key_value


class CachedTransformerDecoder(nn.Module):
    """
    A complete Transformer decoder that supports key-value caching across multiple layers.
    """
    
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        """
        Initialize cached transformer decoder.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            num_layers: Number of decoder layers
            dropout: Dropout probability
        """
        super(CachedTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            CachedTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
                ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through all decoder layers with key-value caching.
        
        Args:
            tgt: Target sequence tensor
            memory: Memory tensor for cross-attention
            kv_cache: Optional list of cached key-value pairs for each layer
            
        Returns:
            Tuple of (output, new_kv_cache) for continued caching
        """
        output = tgt
        new_kv_cache = []

        if kv_cache is None:
            kv_cache = [None] * self.num_layers

        # Process through each decoder layer
        for i, layer in enumerate(self.layers):
            output, new_cache = layer(output, memory, past_key_value=kv_cache[i])
            new_kv_cache.append(new_cache)

        return output, new_kv_cache


class AttentionModel(nn.Module):
    """
    Attention-based model for solving combinatorial optimization problems.
    This model uses a graph attention encoder and a cached transformer decoder
    to generate solutions step by step.
    """

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 n_decode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 dy=False):
        """
        Initialize attention model.
        
        Args:
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension for feedforward networks
            problem: Problem instance to solve
            n_encode_layers: Number of encoder layers
            n_decode_layers: Number of decoder layers
            tanh_clipping: Clipping value for tanh activation
            mask_inner: Whether to mask inner attention
            mask_logits: Whether to mask logits
            normalization: Type of normalization
            n_heads: Number of attention heads
            checkpoint_encoder: Whether to use gradient checkpointing
            shrink_size: Size for shrinking (unused)
            dy: Dynamic flag (unused)
        """
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.is_MCLP = problem.NAME == 'MCLP'
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        # Initialize embedding layers
        node_dim = 5
        self.init_embed = nn.Linear(node_dim, embedding_dim)

        # Graph attention encoder for processing input
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # Positional encoding for decoder
        self.pos_encoder = PositionalEncoding(embedding_dim)

        # Cached transformer decoder
        self.transformer_decoder = CachedTransformerDecoder(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            num_layers=n_decode_layers
        )

        # Projection layers for attention computation
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Learnable placeholder for first decoder step
        self.first_placeholder = nn.Parameter(torch.Tensor(embedding_dim))
        self.first_placeholder.data.uniform_(-1, 1)

    def set_decode_type(self, decode_type, temp=None):
        """
        Set the decoding strategy and temperature.
        
        Args:
            decode_type: Type of decoding ('greedy' or 'sampling')
            temp: Temperature for sampling (optional)
        """
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        Forward pass through the model.
        
        Args:
            input: Input data dictionary
            return_pi: Whether to return the solution sequence
            
        Returns:
            Cost and log-likelihood, optionally the solution sequence
        """
        if self.checkpoint_encoder and self.training:
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))

        _log_p, pi = self._inner(input, embeddings)

        cost = -self.problem.get_total_num(input, pi)
        ll = _log_p.sum(dim=-1)

        if return_pi:
            return cost, ll, pi
        return cost, ll

    def _init_embed(self, input):
        """
        Initialize embeddings from input facilities.
        
        Args:
            input: Input data dictionary
            
        Returns:
            Initial embeddings
        """
        return self.init_embed(input['facilities'])

    def _precompute(self, embeddings):
        """
        Precompute fixed attention data for efficiency.
        
        Args:
            embeddings: Node embeddings
            
        Returns:
            AttentionModelFixed context with precomputed data
        """
        graph_embed = embeddings.mean(1)
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed),
            self._make_heads(glimpse_val_fixed),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _inner(self, input, embeddings):
        """
        Inner loop for generating solution step by step.
        
        Args:
            input: Input data
            embeddings: Node embeddings
            
        Returns:
            Tuple of (log probabilities, solution sequence)
        """
        outputs = []
        sequences = []
        batch_size = embeddings.size(0)

        state = self.problem.make_state(input)
        fixed = self._precompute(embeddings)
        memory = fixed.node_embeddings.permute(1, 0, 2)
        kv_cache = None
        decoder_input = self.first_placeholder.view(1, 1, -1).expand(1, batch_size, -1)

        # Generate solution step by step
        for t in range(state.p):
            # Add positional encoding and decode
            decoder_input_with_pos = decoder_input + self.pos_encoder.pe[t:t + 1]
            decoder_output, kv_cache = self.transformer_decoder(
                tgt=decoder_input_with_pos,
                memory=memory,
                kv_cache=kv_cache
            )

            query = decoder_output.transpose(0, 1)

            # Compute attention and select next node
            glimpse_K, glimpse_V, logit_K = _get_attention_node_data(fixed)
            mask = state.get_mask()
            log_p, _ = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
            logp_selected, selected_idx = self._select_node(log_p.exp().squeeze(1), mask.squeeze(1))

            # Update state and prepare next decoder input
            state = state.update(selected_idx)
            decoder_input = fixed.node_embeddings.gather(
                1,
                selected_idx.view(-1, 1, 1).expand(-1, 1, self.embedding_dim)
            ).permute(1, 0, 2)

            outputs.append(logp_selected)
            sequences.append(selected_idx)

        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def _select_node(self, probs, mask):
        """
        Select next node based on probabilities and mask.
        
        Args:
            probs: Probability distribution over nodes
            mask: Mask indicating valid nodes
            
        Returns:
            Tuple of (log probability of selected node, selected node index)
        """
        assert (probs == probs).all(), "Probs should not contain any nans"
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), "Decode greedy: infeasible action"
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"

        logp = probs.gather(1, selected.unsqueeze(-1)).squeeze(-1).log()
        logp = torch.clamp(logp, min=-100)
        return logp, selected

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        """
        Compute attention logits from query to multiple keys.
        
        Args:
            query: Query tensor
            glimpse_K: Key tensor for glimpse attention
            glimpse_V: Value tensor for glimpse attention
            logit_K: Key tensor for final logit computation
            mask: Mask for invalid positions
            
        Returns:
            Tuple of (logits, glimpse output)
        """
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Compute attention compatibility scores
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Apply attention weights to values
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Compute final logits
        final_Q = glimpse
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # Apply clipping and masking
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        Sample multiple solutions for the same input.
        
        Args:
            input: Input data (batch_size, graph_size, node_dim)
            batch_rep: Number of batch repetitions
            iter_rep: Number of iteration repetitions
            
        Returns:
            Multiple solution samples
        """
        return sample_many(
            lambda input: self._inner(*input),
            lambda input, pi: -self.problem.get_total_num(input[0], pi),
            (input, self.embedder(self._init_embed(input))[0]),
            batch_rep, iter_rep
        )

    def _make_heads(self, v, num_steps=None):
        """
        Reshape tensor to support multi-head attention.
        
        Args:
            v: Input tensor
            num_steps: Number of steps (optional)
            
        Returns:
            Reshaped tensor for multi-head attention
        """
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )