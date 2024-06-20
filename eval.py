from typing import Optional, Tuple
import argparse
import json
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed; set_seed(42)
import utils
import lm_eval
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks


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
        "--do-eval",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer",
    )
    parser.add_argument(
        "--ppl-search-file",
        type=str,
        help="...",
    )
    parser.add_argument(
        "--del-block-num",
        type=int,
        help="Number of blocks to delete.",
        default=0,
    )
    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset for calibration.",
        choices=["wikitext2", "alpaca"],
        default="wikitext2",
    )
    parser.add_argument(
        "--ppl-eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity."
    )
    parser.add_argument("--ppl-eval-batch-size", type=int, default=2, help="Batch size for evaluating the perplexity.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluating with lm eval harness.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "winogrande", "hellaswag", "arc_easy", "arc_challenge"],
    )
    parser.add_argument('--num-fewshot', type=int, default=0, help="Number of fewshots for all tasks.")
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


@torch.no_grad
def remove_redundant_blocks(args, model):
    del_block_list = json.load(open(args.ppl_search_file, "r"))[str(args.del_block_num)]
    logging.info(f"chosen del_block_list: {del_block_list}")
    apply_block_masks(model, del_block_list)


def eval(args, model, tokenizer):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    
    task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)

    logging.info(f"Selected Tasks: {task_names}")

    results = lm_eval.simple_evaluate(hflm, tasks=task_names, num_fewshot=args.num_fewshot, batch_size=args.batch_size)['results']

    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    logging.info(json.dumps(metric_vals, indent=4))

    def calculate_avg_accuracy(task_names, results):
        n_tasks = len(task_names)
        acc_cumul = sum(result.get('acc_norm,none', result['acc,none']) for task, result in results.items())
        return acc_cumul / n_tasks

    acc_avg = calculate_avg_accuracy(task_names, results)
    logging.info(f"Average accuracy across tasks: {acc_avg}")


def main() -> None:
    initialize_tasks()
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

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=compute_dtype, trust_remote_code=True, use_cache=False, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model_size = get_model_params(model)
    logging.info(f"original model size: {model_size/1e9:.3f}B")

    dataset = utils.get_dataset(args.cal_dataset)
    test_dataset = dataset["test"]
    test_loader = utils.prepare_test_dataloader(
        dataset=test_dataset, 
        tokenizer=tokenizer, 
        seqlen=args.ppl_eval_seqlen,
        batch_size=args.ppl_eval_batch_size
    )

    remove_redundant_blocks(args, model)
    logging.info(f"pruned model size: {get_model_params(model)/1e9:.3f}B")
    logging.info(f"pruning ratio: {(1- get_model_params(model)/model_size) * 100:.2f}")

    dataset_ppl = utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    logging.info(f'model ppl: {dataset_ppl:.4f}')

    if args.do_eval:
        eval(args, model, tokenizer)

if __name__ == "__main__":
    main()
