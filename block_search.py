from typing import Optional, Tuple
import argparse
import json
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed; set_seed(42)
import utils
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for computation (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default="bf16",
        help="Data type for computation ('bf16', 'fp32', 'fp64').",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer",
    )
    parser.add_argument(
        "--ppl-search-path",
        type=str,
        help="Path to save the perplexity search results.",
        default="ppls",
    )
    parser.add_argument(
        "--del-block-num",
        type=int,
        help="Number of blocks to delete.",
        default=0,
    )
    parser.add_argument(
        "--block-type",
        type=str,
        help="Block type for searching ('mha', 'mlp', 'mix').",
        choices=["mha", "mlp", "mix"],
        default="mix",
    )
    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset for calibration.",
        choices=["wikitext2", "alpaca"],
        default="alpaca",
    )
    parser.add_argument(
        "--cal-nsamples",
        type=int,
        help="Number of samples for calibration.",
        default=128,
    )
    parser.add_argument(
        "--ppl-eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity."
    )
    parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    return parser.parse_args()


class MaskedLlamaDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = None
        self.mlp = None
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.mask_block = ""

    def setting_layer(self, layer):
        if "mha" not in self.mask_block:
            self.input_layernorm = layer.input_layernorm
            self.self_attn = layer.self_attn
        else:
            self.input_layernorm = None
            self.self_attn = None
        if "mlp" not in self.mask_block:
            self.post_attention_layernorm = layer.post_attention_layernorm
            self.mlp = layer.mlp
        else:
            self.post_attention_layernorm = None
            self.mlp = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        if "mha" not in self.mask_block:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = residual.to(hidden_states.device) + hidden_states
        else:
            self_attn_weights = None
            present_key_value = None

        if "mlp" not in self.mask_block:
        # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual.to(hidden_states.device) + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def get_model_params(model):
    return sum(int(p.nelement()) for p in model.parameters())


@torch.no_grad
def block_search_by_ppl(args, model, test_loader=None, model_size=None):
    # Initialize best results dictionary
    best_results = {}

    # Split blocks into MHA and MLP lists
    mha_block_ids = list(range(model.config.num_hidden_layers)) if args.block_type != "mlp" else [] # You can use BI to reduce the search space if needed
    mlp_block_ids = list(range(model.config.num_hidden_layers)) if args.block_type != "mha" else []

    logging.info(f"mha_block_ids: {mha_block_ids}")
    logging.info(f"mlp_block_ids: {mlp_block_ids}")

    # iterate search process
    current_sequence = set()
    current_ppl = float('inf')

    pbar = tqdm(range(1, args.del_block_num+1), desc=f"searching block del order based on {args.cal_dataset} ppl")
    for del_num in pbar:
        best_candidate = None
        best_candidate_ppl = float('inf')

        candidate_blocks = [("mha", mha_id) for mha_id in mha_block_ids if ("mha", mha_id) not in current_sequence] \
                         + [("mlp", mlp_id) for mlp_id in mlp_block_ids if ("mlp", mlp_id) not in current_sequence] 

        for block_type, block_id in candidate_blocks:
            candidate_sequence = frozenset(current_sequence) | {(block_type, block_id)}
            del_layer_dict = apply_block_masks(model, candidate_sequence)
            candidate_ppl = utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
            revert_block_masks(model, del_layer_dict)

            if candidate_ppl < best_candidate_ppl:
                best_candidate_ppl = candidate_ppl
                best_candidate = candidate_sequence

        if best_candidate is not None:
            current_sequence = best_candidate
            current_ppl = best_candidate_ppl

        del_order_list = list(current_sequence)
        best_results[str(del_num)] = sorted(del_order_list, key=lambda x: x[1], reverse=False)

        print(f"best_ppl: {current_ppl}")
        print(f"best_seq ({del_num}): {sorted(del_order_list, key=lambda x: x[1], reverse=False)}")

    file_name = f"{args.ppl_search_path}/{args.model_path.split('/')[-1]}_{args.block_type}_{args.cal_dataset}_ns_{args.cal_nsamples}_del_order_list.json"
    with open(file_name, "w") as f:
        json.dump(best_results, f)
    logging.info(f"del_order_list path: {file_name}")


def apply_block_masks(model, seq):
    del_layer_dict = {}
    for block_type, block_id in seq:
        chosen_layer = model.model.layers[block_id]
        if isinstance(chosen_layer, MaskedLlamaDecoderLayer):
            chosen_layer.mask_block += block_type
            chosen_layer.setting_layer(del_layer_dict[str(block_id)])
        else:
            new_layer = MaskedLlamaDecoderLayer()
            new_layer.mask_block += block_type
            new_layer.setting_layer(chosen_layer)
            del_layer_dict[str(block_id)] = chosen_layer
            model.model.layers[block_id] = new_layer
    return del_layer_dict


def revert_block_masks(model, del_layer_dict):
    for k, v in del_layer_dict.items():
        layer_id = int(k)
        model.model.layers[layer_id] = v


def main() -> None:
    args = parse_args()
    logging.info(args)
    logging.info(f"PyTorch device: {args.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    if args.compute_dtype == "bf16":
        compute_dtype = torch.bfloat16
    elif args.compute_dtype == "fp32":
        compute_dtype = torch.float32
    elif args.compute_dtype == "fp64":
        compute_dtype = torch.float64
    else:
        raise NotImplementedError("Unsupported compute type.")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=compute_dtype, trust_remote_code=True, device_map="auto", use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model_size = get_model_params(model)
    logging.info(f"original model size: {model_size/1e9:.3f}B")

    dataset = utils.get_dataset(args.cal_dataset)
    test_dataset = dataset["test"]
    sampled_test_dataset = test_dataset.select(random.sample(range(len(test_dataset)), args.cal_nsamples))
    test_loader = utils.prepare_test_dataloader(
        dataset=sampled_test_dataset, 
        tokenizer=tokenizer, 
        seqlen=args.ppl_eval_seqlen,
        batch_size=args.ppl_eval_batch_size
    )

    block_search_by_ppl(args, model, test_loader, model_size)


if __name__ == "__main__":
    main()
