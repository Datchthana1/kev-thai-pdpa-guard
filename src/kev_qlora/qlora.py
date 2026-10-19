"""4-bit NF4 QLoRA backbone for kev's DecisionModel.

`QLoRADecisionModel` is DecisionModel.__init__ re-done with three differences; everything else (LoRA config, targets incl. the
Gated DeltaNet projections, alpha = 2r, dropout .05, PointerHead, row/prefix forward code) is inherited unchanged:

  1. the backbone is loaded with BitsAndBytesConfig(nf4, double quant, bf16 compute) straight onto the GPU (device_map);
     the tiny DeltaNet gate projections in_proj_a / in_proj_b stay bf16 (QUANT_SKIP): quantising 32-wide gates hurts and
     saves nothing;
  2. k-bit preparation: frozen base, input grads on (needed for gradient checkpointing), gradient checkpointing.
     peft.prepare_model_for_kbit_training is deliberately NOT used verbatim: it upcasts every bf16 parameter to fp32,
     which would turn the 248k x 2560 embedding table into 2.5 GB. Embeddings and norms stay bf16 (as Kev's own
     --weights_dtype bf16 path); the LoRA adapter is fp32 (peft autocast_adapter_dtype) and the forward runs under bf16 autocast;
  3. `.to(device)` is never called on the quantised backbone.

The saved adapter (adapter_model.safetensors + adapter_config.json) and head.pt use the standard layout, so
kev.checkpoint.Checkpoint(run).load(device, LoadOptions(dtype=torch.bfloat16)) serves it unmodified, un-quantised.
"""
import torch
import torch.nn as nn

from . import compat

QUANT_SKIP = ["in_proj_a", "in_proj_b"]


def bnb_config(skip=QUANT_SKIP):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=list(skip) or None)


def device_map_for(device):
    d = str(device)
    if d == "cuda":
        return {"": torch.cuda.current_device()}
    if d.startswith("cuda:"):
        return {"": int(d.split(":")[1])}
    return {"": d}


def load_quantized_backbone(name, revision, device, attn):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=torch.bfloat16, attn_implementation=attn,
        quantization_config=bnb_config(), device_map=device_map_for(device)).model


def prepare_kbit(lm, checkpointing=True):
    """Freeze, make the (frozen) embedding output require grad, enable non-reentrant gradient checkpointing."""
    for p in lm.parameters():
        p.requires_grad_(False)
    if checkpointing:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm.enable_input_require_grads()
    lm.config.use_cache = False
    return lm


def make_class():
    from kev.model import DecisionModel, PointerHead, is_hybrid, pad_id
    from transformers import AutoModelForCausalLM

    class QLoRADecisionModel(DecisionModel):
        def __init__(self, name, tok, device, lora=None, revision=None, attn=None, head_dim=256, option_isolation=False,
                     special_embeddings=False, lora_targets="all", dtype=torch.float32, weights=None, direct_load=False,
                     load_4bit=True, checkpointing=True):
            nn.Module.__init__(self)
            if weights:
                raise ValueError("QLoRA does not support full-weight checkpoints")
            attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")
            if load_4bit:
                self.lm = load_quantized_backbone(name, revision, device, attn)
            else:   # same code path minus the quantiser (CPU smoke tests)
                self.lm = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype, attn_implementation=attn).model
            self.pad_id = pad_id(tok)
            self.hybrid = is_hybrid(self.lm.config)
            if self.hybrid and option_isolation:
                raise ValueError("option_isolation needs the packed mask; not available on hybrid backbones")
            self.option_isolation = option_isolation
            self.loaded_4bit = load_4bit
            prepare_kbit(self.lm, checkpointing)
            if lora:
                from peft import LoraConfig, get_peft_model
                extra = {}
                if special_embeddings:
                    from kev.model import SPECIAL
                    extra = {"trainable_token_indices": {"embed_tokens": [tok.convert_tokens_to_ids(t) for t in SPECIAL]}}
                targets = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                           "dense": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                           "attn": ["q_proj", "k_proj", "v_proj", "o_proj"], "qv": ["q_proj", "v_proj"]}[lora_targets]
                if self.hybrid and lora_targets in ("all", "attn"):
                    targets = targets + ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
                cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora, lora_dropout=0.05,
                                 target_modules=targets, **extra)
                self.lm = get_peft_model(self.lm, cfg)
            self.head = PointerHead(self.lm.config.hidden_size, dp=head_dim)
            self.device = device
            if load_4bit:
                self.head.to(device)   # never self.to(): it would try to move the quantised backbone
            else:
                self.to(device)

    return QLoRADecisionModel


def load_adapter_inference(run, device, load_4bit=True, dtype=torch.bfloat16):
    """Load a run dir / hub id (adapter + head.pt) for scoring. load_4bit: NF4 backbone + unmerged adapter (what fits 8 GB);
    otherwise the standard Kev loader in `dtype`. -> (tokenizer, model) in eval mode with the calibration temperature applied."""
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import load_tokenizer
    ck = Checkpoint(run)
    if not load_4bit:
        return ck.load(device, LoadOptions(dtype=dtype))
    from peft import PeftModel
    cls = make_class()
    meta = ck.meta
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    m = cls(meta.base, tok, device, lora=None, revision=meta.base_revision, head_dim=meta.head_dim,
            option_isolation=meta.option_isolation, load_4bit=True, checkpointing=False)
    m.lm = PeftModel.from_pretrained(m.lm, ck.path)
    m.head.load_state_dict(meta.head)
    m.head.temperature = meta.temperature
    m.eval()
    return tok, m


def install(load_4bit=True):
    """Patch kev.train so that `DecisionModel(...)` builds a QLoRADecisionModel. Returns a holder dict that keeps the model."""
    compat.set_env_defaults()
    compat.install_resource_shim()
    compat.install_fcntl_shim()
    import kev.train as kt
    cls = make_class()
    holder = {}

    def factory(*a, **kw):
        kw.pop("direct_load", None)
        m = cls(*a, load_4bit=load_4bit, checkpointing=True, **kw)
        holder["model"] = m
        return m

    kt.DecisionModel = factory
    return holder
