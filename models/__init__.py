from .pe_lstm import PELSTM
from .pi_lstm import PILSTM
from .pi_vrnn import PIVRNN
from .player_ball import PlayerBall
from .team_ball import TeamBall


def load_model(model_name, params, parser=None):
    model_name = model_name.lower()

    if model_name == "pe_lstm":
        return PELSTM(params, parser)
    elif model_name == "pi_lstm":
        return PILSTM(params, parser)
    elif model_name == "pi_vrnn":
        return PIVRNN(params, parser)
    elif model_name == "player_ball":
        return PlayerBall(params, parser)
    elif model_name == "team_ball":
        return TeamBall(params, parser)
    else:
        raise NotImplementedError
