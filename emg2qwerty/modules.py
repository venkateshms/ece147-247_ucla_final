# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn

# Import the PositionalEncoding class from utils instead of lightning
from emg2qwerty.utils import PositionalEncoding


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class LSTMEncoder(nn.Module):
    """A simple LSTM-based encoder for sequence-to-sequence tasks.
    
    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden state in the LSTM.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout probability (applied between LSTM layers).
        bidirectional (bool): Whether to use a bidirectional LSTM.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False,  # Following TNC convention (time-first)
        )
        
        # Output projection layer to map from LSTM output to desired output size
        # If bidirectional, we need to account for doubled hidden size
        self.output_factor = 2 if bidirectional else 1
        self.output_size = hidden_size * self.output_factor
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (T, N, input_size)
        outputs, _ = self.lstm(inputs)
        # outputs shape: (T, N, hidden_size * output_factor)
        return outputs  # (T, N, output_size)


class RNNEncoder(nn.Module):
    """A simple RNN-based encoder for sequence-to-sequence tasks.
    
    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden state in the RNN.
        num_layers (int): Number of RNN layers.
        dropout (float): Dropout probability (applied between RNN layers).
        bidirectional (bool): Whether to use a bidirectional RNN.
        nonlinearity (str): The non-linearity to use: 'tanh' or 'relu'.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        nonlinearity: str = "tanh",
    ) -> None:
        super().__init__()
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False,  # Following TNC convention (time-first)
            nonlinearity=nonlinearity,
        )
        
        # Output projection layer to map from RNN output to desired output size
        # If bidirectional, we need to account for doubled hidden size
        self.output_factor = 2 if bidirectional else 1
        self.output_size = hidden_size * self.output_factor
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (T, N, input_size)
        outputs, _ = self.rnn(inputs)
        # outputs shape: (T, N, hidden_size * output_factor)
        return outputs  # (T, N, output_size)


class TransformerEncoder(nn.Module):
    """A Transformer-based encoder for sequence-to-sequence tasks.
    
    Args:
        input_size (int): Size of the input features.
        d_model (int): Dimension of the model (embedding dimension).
        nhead (int): Number of attention heads.
        num_layers (int): Number of transformer encoder layers.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout probability.
        activation (str): Activation function to use in the feedforward network.
        positional_encoding (bool): Whether to use positional encoding.
        max_seq_length (int): Maximum sequence length for positional encoding.
        chunk_size (int): Size of chunks for processing long sequences. If 0, no chunking is used.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        positional_encoding: bool,
        max_seq_length: int = 5000,  # Default to 5000 like LSTM
        chunk_size: int = 0,  # Default to no chunking
    ) -> None:
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, d_model)
        self.chunk_size = chunk_size
        
        # Use the imported PositionalEncoding class
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length) if positional_encoding else nn.Identity()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,  # (T, N, E) format
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input to d_model dimensions
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        if self.chunk_size > 0 and x.size(0) > self.chunk_size:
            # Process long sequences in chunks to avoid memory issues
            outputs = []
            seq_len, batch_size, d_model = x.shape
            
            for i in range(0, seq_len, self.chunk_size):
                end_idx = min(i + self.chunk_size, seq_len)
                chunk = x[i:end_idx]
                chunk_output = self.transformer_encoder(chunk)
                outputs.append(chunk_output)
                
            x = torch.cat(outputs, dim=0)
        else:
            # Process the entire sequence at once
            x = self.transformer_encoder(x)
        
        return x


class ConvTransformerEncoder(nn.Module):
    """A hybrid Convolutional-Transformer encoder that first applies convolutional layers 
    for local feature extraction and then uses a Transformer for modeling long-range dependencies.
    
    Args:
        input_size (int): Size of the input features.
        d_model (int): Dimension of the model (embedding dimension).
        nhead (int): Number of attention heads.
        num_layers (int): Number of transformer encoder layers.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout probability.
        activation (str): Activation function to use in the feedforward network.
        positional_encoding (bool): Whether to use positional encoding.
        max_seq_length (int): Maximum sequence length for positional encoding.
        chunk_size (int): Size of chunks for processing long sequences. If 0, no chunking is used.
        cnn_channels (list): List of channel sizes for the CNN layers.
        kernel_size (int): Kernel size for the CNN layers.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        positional_encoding: bool,
        max_seq_length: int = 5000,
        chunk_size: int = 0,
        cnn_channels: Sequence[int] = (64, 128),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        
        # CNN layers for local feature extraction
        cnn_layers = []
        in_channels = input_size
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Projection layer from CNN output to transformer dimension
        self.proj = nn.Linear(cnn_channels[-1], d_model)
        
        self.chunk_size = chunk_size
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length) if positional_encoding else nn.Identity()
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,  # (T, N, E) format
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (T, N, input_size)
        T, N, C = x.shape
        
        # Reshape for CNN (expects NCL format)
        x = x.permute(1, 2, 0)  # (N, C, T)
        
        # Apply CNN layers
        x = self.cnn(x)  # (N, cnn_channels[-1], T')
        
        # Reshape back to TNC format for transformer
        x = x.permute(2, 0, 1)  # (T', N, cnn_channels[-1])
        
        # Project to d_model dimension
        x = self.proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder with optional chunking
        if self.chunk_size > 0 and x.size(0) > self.chunk_size:
            outputs = []
            seq_len, batch_size, d_model = x.shape
            
            for i in range(0, seq_len, self.chunk_size):
                end_idx = min(i + self.chunk_size, seq_len)
                chunk = x[i:end_idx]
                chunk_output = self.transformer_encoder(chunk)
                outputs.append(chunk_output)
                
            x = torch.cat(outputs, dim=0)
        else:
            x = self.transformer_encoder(x)
        
        return x
