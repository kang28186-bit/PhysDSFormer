from .physdsformer import PhysDSFormer, physdsformer_base, physdsformer_tiny


MODEL_REGISTRY = {
    "physdsformer-tiny": physdsformer_tiny,
    "physdsformer-base": physdsformer_base,
}


def build_model(name):
    try:
        return MODEL_REGISTRY[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError("Unknown model '{}'. Choose from: {}".format(name, choices)) from exc


__all__ = [
    "PhysDSFormer",
    "physdsformer_tiny",
    "physdsformer_base",
    "build_model",
]

