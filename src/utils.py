from mace.calculators.mace import MACECalculator


def get_layer_specific_feature_slices(calc: MACECalculator) -> list[slice]:
    num_layers = calc.models[0].num_interactions
    irreps_out = calc.models[0].products[0].linear.__dict__["irreps_out"]
    l_max = irreps_out.lmax
    features_per_layer = irreps_out.dim // (l_max + 1) ** 2
    slices = [slice(0, (i + 1) * features_per_layer) for i in range(num_layers)]
    return slices
