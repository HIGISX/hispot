import torch
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):
    """
    Skip connection module that adds the input to the output of a sub-module.
    This helps with gradient flow in deep networks and can improve training stability.
    """

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        """
        Forward pass: adds input to the output of the sub-module.
        
        Args:
            input: Input tensor
            
        Returns:
            Tensor with skip connection applied
        """
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism as described in "Attention Is All You Need".
    This module computes attention weights across multiple heads and applies them to values.
    """
    
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        """
        Initialize multi-head attention module.
        
        Args:
            n_heads: Number of attention heads
            input_dim: Input dimension
            embed_dim: Output embedding dimension
            val_dim: Value dimension (defaults to embed_dim // n_heads)
            key_dim: Key dimension (defaults to val_dim)
        """
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        # Scaling factor for attention scores to prevent softmax saturation
        self.norm_factor = 1 / math.sqrt(key_dim)

        # Learnable weight matrices for queries, keys, and values
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        # Output projection matrix
        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):
        """
        Initialize parameters using Xavier uniform initialization.
        """
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            q: Query tensor (batch_size, n_query, input_dim)
            h: Input data tensor (batch_size, graph_size, input_dim), defaults to q for self-attention
            mask: Attention mask (batch_size, n_query, graph_size) to prevent attention to certain positions
            
        Returns:
            Output tensor after attention computation (batch_size, n_query, embed_dim)
        """
        if h is None:
            h = q  # compute self-attention

        # Validate input dimensions
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        # Flatten tensors for matrix multiplication
        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # Reshape for multi-head computation
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, keys, and values for each head
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Compute attention compatibility scores
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Apply mask if provided to prevent attention to masked positions
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        # Apply softmax to get attention weights
        attn = torch.softmax(compatibility, dim=-1)

        # Handle masked positions by setting attention weights to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        # Apply attention weights to values
        heads = torch.matmul(attn, V)

        # Project output through final linear layer
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):
    """
    Normalization layer that supports both batch normalization and instance normalization.
    """

    def __init__(self, embed_dim, normalization='batch'):
        """
        Initialize normalization layer.
        
        Args:
            embed_dim: Embedding dimension
            normalization: Type of normalization ('batch' or 'instance')
        """
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):
        """
        Initialize normalization parameters using Xavier uniform initialization.
        """
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        Apply normalization to input tensor.
        
        Args:
            input: Input tensor
            
        Returns:
            Normalized tensor
        """
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):
    """
    Complete multi-head attention layer with skip connections and normalization.
    This layer combines attention mechanism with feed-forward network and residual connections.
    """

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        """
        Initialize multi-head attention layer.
        
        Args:
            n_heads: Number of attention heads
            embed_dim: Embedding dimension
            feed_forward_hidden: Hidden dimension of feed-forward network
            normalization: Type of normalization to use
        """
        super(MultiHeadAttentionLayer, self).__init__(
            # First sub-layer: Multi-head attention with skip connection
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            # Normalization after attention
            Normalization(embed_dim, normalization),
            # Second sub-layer: Feed-forward network with skip connection
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            # Normalization after feed-forward
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    """
    Graph attention encoder that processes graph-structured data using multiple attention layers.
    This encoder can be used for facility location problems and other graph-based tasks.
    """

    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        """
        Initialize graph attention encoder.
        
        Args:
            n_heads: Number of attention heads in each layer
            embed_dim: Output embedding dimension
            n_layers: Number of attention layers
            node_dim: Input node dimension (if None, input is assumed to already be in embed_dim)
            normalization: Type of normalization to use
            feed_forward_hidden: Hidden dimension of feed-forward networks
        """
        super(GraphAttentionEncoder, self).__init__()

        # Linear layer to map input to embedding space if needed
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        # Stack of attention layers
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):
        """
        Forward pass through the graph attention encoder.
        
        Args:
            x: Input tensor (batch_size, graph_size, node_dim or embed_dim)
            mask: Attention mask (currently not supported)
            
        Returns:
            Tuple of:
                - Node embeddings (batch_size, graph_size, embed_dim)
                - Graph-level embedding (batch_size, embed_dim) - average of node embeddings
        """
        assert mask is None, "TODO mask not yet supported!"

        # Map input to embedding space if needed
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        # Pass through all attention layers
        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim) - node embeddings
            h.mean(dim=1),  # (batch_size, embed_dim) - graph-level embedding (average pooling)
        )
