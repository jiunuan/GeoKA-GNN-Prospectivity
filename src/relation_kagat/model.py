"""Distance-conditioned Edge-KAN plus independent Same/Along/Cross channels."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import softmax
from .kan import KANLinear


class RelationKAGAT(nn.Module):
    def __init__(self, continuous_count, cardinalities, settings, geochemistry_indices=()):
        super().__init__()
        self.continuous_count = continuous_count
        self.cardinalities = list(cardinalities)
        self.missing_count = 3
        self.geochemistry_indices = list(geochemistry_indices)
        if any(i < 0 or i >= continuous_count for i in self.geochemistry_indices):
            raise ValueError("Geochemistry feature index out of bounds")
        self.settings = dict(settings)
        self.hidden = int(settings["hidden_channels"])
        self.heads = int(settings["attention_heads"])
        self.dropout = float(settings.get("dropout", 0.3))
        self.strength = float(settings["relation_strength"])
        self.regularization_scale = float(settings["relation_regularization_scale"])
        self.chunk_size = int(settings["edge_chunk_size"])
        self.graph_gate_max = float(settings["graph_gate_max"])
        initial = float(settings["graph_gate_initial"])
        if self.hidden <= 0 or self.heads <= 0 or self.hidden % self.heads:
            raise ValueError("hidden_channels must be positive and divisible by attention_heads")
        if not 0 <= self.dropout < 1 or not 0 < self.strength <= 1:
            raise ValueError("Invalid dropout or relation_strength")
        if not 0 < initial <= self.graph_gate_max <= 1 or initial == 1:
            raise ValueError("Invalid graph gate bounds")
        if self.chunk_size < 1 or self.regularization_scale <= 0:
            raise ValueError("Chunk size and regularization scale must be positive")
        options = dict(grid_size=settings["grid_size"], spline_order=settings["spline_order"])
        embedding_dim = settings["embedding_dim"]
        self.embeddings = nn.ModuleList(nn.Embedding(c, embedding_dim, padding_idx=0)
                                        for c in cardinalities)
        inputs = continuous_count + len(cardinalities) * embedding_dim + self.missing_count
        width, relation_dim = settings["encoder_width"], settings["relation_dim"]
        self.node_first = KANLinear(inputs, width, **options)
        self.node_second = KANLinear(width, self.hidden, **options)
        self.relation_first = KANLinear(2 * self.hidden + 5, relation_dim, **options)
        self.relation_second = KANLinear(relation_dim, relation_dim, **options)
        self.edge_selector = KANLinear(relation_dim, 1, **options)
        self.attention_source = nn.Linear(self.hidden, self.heads, bias=False)
        self.attention_target = nn.Linear(self.hidden, self.heads, bias=False)
        self.attention_relation = nn.Linear(relation_dim, self.heads, bias=False)
        self.message = KANLinear(self.hidden + relation_dim, self.hidden, **options)
        self.update = KANLinear(2 * self.hidden, self.hidden, **options)
        self.output_norm = nn.LayerNorm(self.hidden)
        self.prediction_head = KANLinear(self.hidden, 1, **options)
        self.graph_gate = nn.Parameter(torch.tensor(math.log(initial / (1 - initial))))
        self.geo_enabled = True
        self.residual_scale = float(settings["geo_residual_scale"])
        self.geo_regularization_scale = float(settings["geo_regularization_scale"])
        gain = float(settings["geo_gain_initial"])
        if not 0 < gain < 1 or self.residual_scale <= 0 or self.geo_regularization_scale <= 0:
            raise ValueError("Invalid geological channel gain/scale")
        self.geo_encoders = nn.ModuleList(KANLinear(1, 4, **options) for _ in range(3))
        self.geo_bias = nn.ModuleList(nn.Linear(4, self.heads, bias=False) for _ in range(3))
        self.geo_selector = nn.ModuleList(nn.Linear(4, 1, bias=False) for _ in range(3))
        self.geo_messages = nn.ModuleList(nn.Linear(4, self.hidden, bias=False) for _ in range(3))
        self.geo_context = nn.Linear(2 * self.hidden, 3)
        nn.init.zeros_(self.geo_context.weight)
        nn.init.zeros_(self.geo_context.bias)
        for layer in [*self.geo_bias, *self.geo_selector, *self.geo_messages]:
            nn.init.zeros_(layer.weight)
        self.geo_logit_gain = nn.Parameter(torch.full((3,), math.log(gain / (1 - gain))))

    def _encode_relation(self, source, target, attributes):
        inputs = torch.cat((torch.abs(source - target), source * target, attributes), -1)
        return self.strength * self.relation_second(torch.tanh(self.relation_first(inputs)))

    def _encode_message(self, source, relation):
        return self.message(torch.cat((source, relation), -1))

    def _gate_geochemistry(self, continuous, missing):
        if not self.geochemistry_indices:
            return continuous
        gated = continuous.clone()
        gated[:, self.geochemistry_indices] *= missing[:, 1:2]
        return gated

    def gains(self):
        return self.geo_logit_gain.sigmoid()

    def _chunks(self, function, *tensors):
        parts = []
        for start in range(0, len(tensors[0]), self.chunk_size):
            args = tuple(t[start:start + self.chunk_size] for t in tensors)
            parts.append(checkpoint(function, *args, use_reentrant=False)
                         if self.training else function(*args))
        return torch.cat(parts)

    def forward(self, x, edge_index, edge_attr):
        count, cats = self.continuous_count, len(self.cardinalities)
        if x.ndim != 2 or x.shape[1] != count + cats + 3:
            raise ValueError("Node features must be continuous, categories and three availability indicators")
        if edge_index.shape[0] != 2 or edge_attr.shape != (edge_index.shape[1], 4):
            raise ValueError("Expected distance, same igneous body, along fault and cross fault")
        categories = [emb(x[:, count + i].long()) for i, emb in enumerate(self.embeddings)]
        missing = x[:, count + cats:]
        continuous = self._gate_geochemistry(x[:, :count], missing)
        inputs = torch.cat((torch.tanh(continuous / 3), *categories, missing), -1)
        stable = self.node_second(torch.tanh(self.node_first(inputs)))
        h = F.dropout(stable, self.dropout, self.training)
        source, target = edge_index
        aggregate = torch.zeros_like(h)
        residual = torch.zeros_like(h)
        if source.numel():
            # Preserve the original five-slot backbone: only distance is active.
            # The three geological attributes are encoded independently below.
            backbone = torch.cat((edge_attr[:, :1], edge_attr.new_zeros((len(edge_attr), 4))), -1)
            relation = self._chunks(self._encode_relation, h[source], h[target], backbone)
            gate_logits = self.edge_selector(relation).squeeze(-1)
            energy = F.leaky_relu(self.attention_source(h)[source]
                                  + self.attention_target(h)[target]
                                  + self.attention_relation(relation), 0.2)
            bias, selector = torch.zeros_like(energy), torch.zeros_like(gate_logits)
            context = self.geo_context(torch.cat((stable[source], stable[target]), -1)).sigmoid()
            content = torch.tanh(stable[source] - stable[target])
            gains, channels = self.gains(), []
            if self.geo_enabled:
                for j, encoder in enumerate(self.geo_encoders):
                    value = edge_attr[:, j + 1:j + 2]
                    z = F.linear(F.silu(value), encoder.base_weight) + F.linear(
                        (encoder.b_splines(value) - encoder.b_splines(torch.zeros_like(value[:1]))).flatten(1),
                        encoder.spline_weight.flatten(1))
                    bias = bias + gains[j] * self.geo_bias[j](z)
                    selector = selector + gains[j] * self.geo_selector[j](z).squeeze(-1)
                    channels.append(gains[j] * self.geo_messages[j](z) * context[:, j:j + 1] * content)
            alpha = softmax(energy + bias, target, num_nodes=len(h))
            gate = (gate_logits + selector).sigmoid()
            messages = self._chunks(self._encode_message, h[source], relation)
            messages = messages.reshape(-1, self.heads, self.hidden // self.heads)
            weighted = messages * alpha[:, :, None] * gate[:, None, None]
            aggregate.index_add_(0, target, weighted.reshape(-1, self.hidden))
            for j, message in enumerate(channels):
                weights = edge_attr[:, j + 1]
                denominator = weights.new_zeros(len(h)).index_add_(0, target, weights)
                normalized = weights / denominator[target].clamp_min(1e-8)
                residual = residual.index_add(0, target, normalized[:, None] * message)
        weight = torch.sigmoid(self.graph_gate).clamp(max=self.graph_gate_max)
        updated = self.output_norm(h + weight * self.update(torch.cat((h, aggregate), -1))
                                   + self.residual_scale * residual)
        return self.prediction_head(torch.tanh(updated)).squeeze(-1)

    def regularization(self):
        l1, smooth = [], []
        for name, layer in self.named_children():
            if isinstance(layer, KANLinear):
                scale = self.regularization_scale if name in {
                    "relation_first", "relation_second", "edge_selector"} else 1.0
                l1.append(scale * layer.spline_weight.abs().mean())
                smooth.append(scale * torch.diff(layer.spline_weight, n=2, dim=-1).square().mean())
        for layer in self.geo_encoders:
            l1.append(self.geo_regularization_scale * layer.spline_weight.abs().mean())
            smooth.append(self.geo_regularization_scale * torch.diff(layer.spline_weight, n=2, dim=-1).square().mean())
        return torch.stack(l1).sum(), torch.stack(smooth).sum()


def target_subgraph(x, edges, attrs, targets):
    """All incoming edges of target nodes; exact in evaluation for one GAT layer."""
    targets = torch.as_tensor(targets, dtype=torch.long, device=x.device)
    keep = torch.isin(edges[1], targets)
    selected = edges[:, keep]
    active = torch.unique(torch.cat((targets, selected.flatten())), sorted=True)
    remap = torch.full((len(x),), -1, dtype=torch.long, device=x.device)
    remap[active] = torch.arange(len(active), device=x.device)
    return x[active], remap[selected], attrs[keep], remap[targets]


@torch.no_grad()
def predict_logits(model, x, edges, attrs, targets, batch_size):
    model.eval()
    parts = []
    for start in range(0, len(targets), batch_size):
        xx, ee, aa, selected = target_subgraph(x, edges, attrs, targets[start:start + batch_size])
        parts.append(model(xx, ee, aa)[selected].cpu())
    return torch.cat(parts)
