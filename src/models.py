# deep learning libraries
import torch

# other libraries
import math

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """

        """
        outputs = inputs + self.pe[:inputs.size(0)]
        return self.dropout(outputs)


class BaseDailyElectricTransformer(torch.nn.Module):
    """
    This class is based in a Transformer Encoder and a multi layer perceptron. Each day (24 prices) is a "word"

    Attributes
    ----------
    values_embeddings : torch.nn.Sequential
        embeddings for past values
    positional_encoding : PositionalEncoding
        positional encoding for the transformer encoder
    transformer_encoder : torch.nn.TransformerEncoder
        transformer encoder
    features_embeddings : torch.nn.Sequential
        embeddings for features
    mlp : torch.nn.Module : torch.nn.Sequential
        multi layer perceptron for the model

    Methods
    -------
    forward -> torch.Tensor
    """

    def __init__(self, embedding_dim: int = 32, num_heads: int = 8, dim_feedforward: int = 128, num_layers: int = 6,
                 normalize_first: bool = False, dropout: float = 0.2, activation: str = 'relu') -> None:
        """
        Constructor for BaseElectricTransformer class

        Parameters
        ----------
        embedding_dim: int, Optional
            dimensions for embeddings. Default: 32
        num_heads: int, Optional
            number of heads for the transformer. Default: 8
        dim_feedforward: int, Optional
            dimensions of feedforward layers. Default: 128
        num_layers: int, Optional
            number of layers for the transformer. Default: 6
        normalize_first : bool, Optional
            normalize_first argument for transformer encoder. Default: False
        dropout: float, Optional
            dropout rate. Default: 0.2

        Returns
        -------
        None
        """

        # call torch.nn.Module constructor
        super().__init__()

        # define activation function
        if activation == 'relu':
            activation_function = torch.nn.ReLU()
        else:
            activation_function = torch.nn.GELU()

        # define inputs embeddings
        self.values_embeddings = torch.nn.Sequential(
            torch.nn.Linear(24, embedding_dim),
            activation_function,
        )

        # define transformer layers
        self.positional_encoding = PositionalEncoding(embedding_dim)
        encoder_layers = torch.nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, dropout,
                                                          batch_first=True, norm_first=normalize_first,
                                                          activation=activation)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers)

        # define current features embeddings
        self.features_embeddings = torch.nn.Sequential(
            torch.nn.Linear(48, embedding_dim),
            activation_function
        )

        # define multi layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * embedding_dim),
            torch.nn.Linear(2 * embedding_dim, dim_feedforward),
            torch.nn.Dropout(dropout),
            activation_function,
            torch.nn.Linear(dim_feedforward, 24)
        )

    def forward(self, values: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        This method computes the output for the BaseElectricTransformer object.

        Parameters
        ----------
        values : torch.Tensor
            past prices. Dimensions: [batch, sequence length, 1]
        features : torch.Tensor
            current features. Dimensions: [batch, 24, 2]

        Returns
        -------
        torch.Tensor
            batch of prices. Dimensions: [batch, sequence_length]
        """

        # compute inputs embeddings
        values_embeddings = self.values_embeddings(values.reshape((values.size(0), -1, 24)))

        # compute output of lstm and embeddings for the hour
        transformer_inputs = self.positional_encoding(values_embeddings)
        transformer_outputs = self.transformer_encoder(transformer_inputs)

        # compute current features embeddings
        features_embeddings = self.features_embeddings(features.reshape(features.size(0), -1, 48))

        # concat everything and compute output of linear layer
        mlp_inputs = torch.concat((features_embeddings, transformer_outputs), dim=2)
        outputs = self.mlp(mlp_inputs)

        return outputs.reshape((values.size(0), -1))


class sMAPELoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        differences = 2 * torch.abs(targets - predictions)
        loss_value = torch.mean(differences / (torch.abs(targets) + torch.abs(predictions)))

        return loss_value
