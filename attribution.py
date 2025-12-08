import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple cache so we don't reload HF models every time
_MODEL_CACHE = {}


def _get_model_and_tokenizer(model_name):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=True,
    ).to(DEVICE)
    model.eval()

    _MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def _l2_activation_saliency(hidden_states):
    """
    hidden_states: tuple of (layer0,...,layerL), each of shape [1, seq, dim].
    Returns list of {layer_index, scores[list[float]]}.
    """
    layers = []
    for idx, h in enumerate(hidden_states):
        h_seq = h[0]  # [seq, dim]
        scores = torch.norm(h_seq, dim=-1)
        layers.append(
            {
                "layer_index": idx,
                "scores": scores.detach().cpu().tolist(),
            }
        )
    return layers


def _attention_rollout(attentions):
    """
    attentions: tuple[L] of tensors [1, num_heads, seq, seq]
    """
    rollout = None
    layers = []

    for idx, att in enumerate(attentions):
        # [1, H, seq, seq] -> [seq, seq]
        att_mean = att.mean(dim=1)[0]
        eye = torch.eye(att_mean.size(-1), device=att_mean.device)
        att_aug = att_mean + eye
        att_aug = att_aug / att_aug.sum(dim=-1, keepdim=True)

        if rollout is None:
            rollout = att_aug
        else:
            rollout = att_aug @ rollout

        scores = rollout[0]  # attention received by tokens from CLS
        layers.append(
            {
                "layer_index": idx,
                "scores": scores.detach().cpu().tolist(),
            }
        )

    return layers


def _grad_input_saliency(hidden_states, target_layer):
    """
    Approx Grad × Input on a chosen layer using CLS energy objective.
    We backprop from the squared L2 norm of CLS at target_layer
    back to that layer's hidden states.
    """
    # allow grads on intermediate activations
    for h in hidden_states:
        h.retain_grad()

    # CLS vector at target layer
    cls_vec = hidden_states[target_layer][0, 0, :]
    scalar = (cls_vec ** 2).sum()
    scalar.backward()

    grad = hidden_states[target_layer].grad
    if grad is None:
        # fallback: just activation norm
        h_seq = hidden_states[target_layer][0]
        scores = torch.norm(h_seq, dim=-1)
    else:
        g_seq = grad[0]
        h_seq = hidden_states[target_layer][0]
        scores = (g_seq * h_seq).sum(dim=-1).abs()

    return scores.detach().cpu().tolist()


def compute_token_attributions(text, model_name, method="activation"):
    """
    Compute per-token scores across layers for a single text.
    method: 'activation', 'attention', or 'grad_input'
    """
    tokenizer, model = _get_model_and_tokenizer(model_name)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    # NOTE: we do NOT call requires_grad_ on encoded tensors;
    # only float tensors can require grad, and we only need grads
    # on the hidden states created by the model forward pass.
    outputs = model(**encoded)
    hidden_states = outputs.hidden_states  # tuple[L+1]
    attentions = outputs.attentions        # tuple[L]

    if method == "activation":
        layers = _l2_activation_saliency(hidden_states)

    elif method == "attention":
        layers = _attention_rollout(attentions)

    elif method == "grad_input":
        mid = len(hidden_states) // 2
        scores = _grad_input_saliency(hidden_states, target_layer=mid)
        layers = [
            {
                "layer_index": mid,
                "scores": scores,
            }
        ]
    else:
        layers = _l2_activation_saliency(hidden_states)

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    method_map = {
        "activation": "Activation norm (‖h_token‖₂)",
        "attention": "Attention rollout (CLS → token)",
        "grad_input": "Grad × Input (CLS energy)",
    }

    return {
        "tokens": tokens,
        "layers": layers,
        "method": method,
        "method_readable": method_map.get(method, method),
    }
