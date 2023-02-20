import torch.nn as nn

from set_transformer.blocks import ISAB, PMA, SAB


class SetTransformer(nn.Module):
    def __init__(self, in_dim, out_dim=None, embed_type="invariant"):
        """
        Arguments:
            in_dim: an integer.
            out_dim: an integer.
            embed_type: a string, either "invariant" or "equivariant".
        """
        super().__init__()

        d = 128
        m = 16  # number of inducing points
        h = 4  # number of heads
        k = 4  # number of seed vectors
        self.embed_type = embed_type

        self.embed = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(ISAB(d, m, h, RFF(d), RFF(d)), ISAB(d, m, h, RFF(d), RFF(d)))
        if self.embed_type.startswith("i"):  # invariant
            self.decoder = nn.Sequential(PMA(d, k, h, RFF(d)), SAB(d, h, RFF(d)))
            self.predictor = nn.Linear(k * d, out_dim)
        else:  # self.embed_type.startswith("e"):  # equivariant
            self.predictor = nn.Linear(d, out_dim)

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, in_dim].
        Returns:
            if embed_type is "invariant", returns a float tensor with shape [b, out_dim].
            if embed_type is "equivariant", returns a float tensor with shape [b, n, out_dim].
        """

        x = self.embed(x)  # shape [b, n, d]
        x_enc = self.encoder(x)  # shape [b, n, d]

        if self.embed_type.startswith("i"):  # invariant
            x_dec = self.decoder(x_enc)
            b, k, d = x_dec.shape
            return self.predictor(x_dec.view(b, k * d))

        else:  # elif self.embed_type.startswith("e"):  # equivariant
            return self.predictor(x_enc)


class RFF(nn.Module):
    """
    Row-wise FeedForward layers.
    """

    def __init__(self, d):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.layers(x)
