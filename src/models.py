# deep learning libraries
import torch

# other libraries
import math

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


class PositionalEncoding(torch.nn.Module):
    @torch.no_grad()
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        This method is the constructor of PositionalEncoding class

        Args:
            d_model: dimension of the inputs of the model
            dropout: rate of dropout. Defaults to 0.1.
            max_len: maximum length. Defaults to 5000.
        """

        # call super class constructor
        super().__init__()

        # define dropout layer
        self.dropout = torch.nn.Dropout(p=dropout)

        # define positional encoding layer
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.pe: torch.Tensor = torch.nn.Parameter(torch.zeros(max_len, 1, d_model))
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass

        Args:
            inputs: inputs tensor. Dimensions: [].

        Returns:
            outputs tensor. Dimensions:
        """

        # compute positional encoding
        outputs = inputs + self.pe[: inputs.size(0)]
        return self.dropout(outputs)


class BaseDailyElectricTransformer(torch.nn.Module):
    """
    This class is based in a Transformer Encoder and a multi layer perceptron. Each day (24 prices) is a "word"

    Args:
        values_embeddings: embeddings for past values
        positional_encoding: positional encoding for the transformer encoder
        transformer_encoder: transformer encoder
        features_embeddings: embeddings for features
        mlp: multi layer perceptron for the model
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        num_heads: int = 8,
        dim_feedforward: int = 128,
        num_layers: int = 6,
        normalize_first: bool = False,
        dropout: float = 0.2,
        activation: str = "relu",
    ) -> None:
        """
        Constructor for BaseElectricTransformer class

        Args:
            embedding_dim: dimensions for embeddings. Defaults to 32.
            num_heads: number of heads for the transformer. Defaults to 8.
            dim_feedforward: dimensions of feedforward layers. Defaults to 128.
            num_layers: number of layers for the transformer. Defaults to 6.
            normalize_first: normalize_first indicator for transformer encoder. Defaults to False.
            dropout: dropout rate. Defaults to 0.2.
            activation: activation function to use. Defaults to "relu".
        """

        # call torch.nn.Module constructor
        super().__init__()

        # define activation function
        activation_function: torch.nn.Module
        if activation == "relu":
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
        encoder_layers = torch.nn.TransformerEncoderLayer(
            embedding_dim,
            num_heads,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=normalize_first,
            activation=activation,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layers, num_layers
        )

        # define current features embeddings
        self.features_embeddings = torch.nn.Sequential(
            torch.nn.Linear(48, embedding_dim), activation_function
        )

        # define multi layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * embedding_dim),
            torch.nn.Linear(2 * embedding_dim, dim_feedforward),
            torch.nn.Dropout(dropout),
            activation_function,
            torch.nn.Linear(dim_feedforward, 24),
        )

    def forward(self, values: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        This method computes the output for the BaseElectricTransformer object.

        Args:
            values: past prices. Dimensions: [batch, sequence length, 1]
            features: current features. Dimensions: [batch, 24, 2]

        Returns:
            batch of prices. Dimensions: [batch, sequence_length]
        """

        # compute inputs embeddings
        values_embeddings = self.values_embeddings(
            values.reshape((values.size(0), -1, 24))
        )

        # compute output of lstm and embeddings for the hour
        transformer_inputs = self.positional_encoding(values_embeddings)
        transformer_outputs = self.transformer_encoder(transformer_inputs)

        # compute current features embeddings
        features_embeddings = self.features_embeddings(
            features.reshape(features.size(0), -1, 48)
        )

        # concat everything and compute output of linear layer
        mlp_inputs = torch.concat((features_embeddings, transformer_outputs), dim=2)
        outputs = self.mlp(mlp_inputs)

        return outputs.reshape((values.size(0), -1))


class sMAPELoss(torch.nn.Module):
    def __init__(self) -> None:
        """
        Constructor sMAPELoss
        """

        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass

        Args:
            predictions: tensor with the predictions. Dimensions: [*]
            targets: tensor with the targets. Dimensions: same as the predictions.

        Returns:
            tensor with the losses. Dimensions: [*].
        """

        # compute loss value
        differences = 2 * torch.abs(targets - predictions)
        loss_value = torch.mean(
            differences / (torch.abs(targets) + torch.abs(predictions))
        )

        return loss_value
