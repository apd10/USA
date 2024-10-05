import torch
import torch.nn as nn
from datasets import load_dataset
import argparse
from tqdm import tqdm
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import math
import gc 
import numpy as np
DEBUG_SAMPLES = 10
from USA import USA


def loss_function(yhat ,ytarget, beta):
    weight = ytarget * (beta - 1) + torch.ones_like(ytarget)
    loss = nn.functional.binary_cross_entropy(yhat.reshape(-1), ytarget.reshape(-1), weight = weight.reshape(-1))
    return loss


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def load_data(name):
    
    if name == "billsum":
        dataset = load_dataset("FiscalNote/billsum")
    elif name == "qmsum":
        dataset = load_dataset("ioeddk/qmsum")
    elif name.startswith("lb-"):
        dname = name.split('-')[1]
        dataset = load_dataset("THUDM/LongBench", dname)
    else:
        raise NotImplementedError
    
    return dataset

def load_model(name):
    if name == "llama3-inst":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    else:
        raise NotImplementedError
    return model, tokenizer

def prompt_builder(model_name, data_name):
    
    if model_name == "llama3-inst" and data_name == "billsum":
        def builder (instance):
            messages = [{"role": "user", "content": 'Please summarize the following data: {}'.format(instance['text'])},
            {"role" : "assistant", "content": 'Tilte:{} \n Summary :{}'.format(instance['title'], instance['summary'])},
            {"role" : "user", "content" : 'Provide one line summary'}]
            return messages
        return builder
    elif model_name == "llama3-inst" and data_name == "qmsum":

        def builder (instance):
            messages = [{"role": "user", "content": '{}'.format(instance['text'].split("Answer:")[0])},
            {"role" : "assistant", "content": '{}'.format(instance['answer'])}]
            return messages
        return builder
    elif model_name == "llama3-inst" and data_name.startswith("lb-"):
        def builder (instance):
            import pdb
            pdb.set_trace()
            messages = [{"role": "user", "content": '{}'.format(instance['text'].split("Answer:")[0])},
            {"role" : "assistant", "content": '{}'.format(instance['answer'])}]
            return messages
        return builder
    else:
        raise NotImplementedError


def compute_target_mask(attn_weights, mode):
    with torch.no_grad():
        if mode.startswith("top"):
            topk = int(mode.split('-')[1])
            mask = torch.zeros_like(attn_weights)
            _,idx = torch.topk(attn_weights, dim=-1, k=topk) # [B,A,S,T]
            view_idx = idx.view(-1,idx.shape[-1])
            view_idx = view_idx + torch.arange(view_idx.shape[0], device=view_idx.device).reshape(-1,1) * attn_weights.shape[-1]
            mask.view(-1)[view_idx.view(-1)] = 1.0
            return mask
        elif mode.startswith("coverage"):
            cov = float(mode.split('-')[1])
            mask = torch.zeros_like(attn_weights)
            values,idx = torch.sort(attn_weights, descending=True, dim=-1) # [B,A,S,T]
            values = (torch.cumsum(values, dim=-1) < cov)
            view_idx = idx.view(-1,idx.shape[-1])
            values_view = values.view(-1, values.shape[-1])
            view_idx = view_idx + torch.arange(view_idx.shape[0], device=view_idx.device).reshape(-1,1) * attn_weights.shape[-1]
            mask.view(-1)[view_idx.view(-1)[values_view.view(-1)]] = 1.0
            return mask
        else:
            raise NotImplementedError

def get_kq_mask(output):
    B,A,S1,S2 = output.shape
    mask  = torch.ones_like(output)
    # do not consider first 64 tokens
    mask[:,:,:,:64] = 0.
    # do not consider last 64 tokens and causal mask
    mask = mask * (torch.tril(mask, diagonal=-64) !=0).float()
    return mask


def evaluate_coverage(model, usas, test_data_loader, target_mode, num_queries, num_samples=10):
    recalls = torch.zeros(len(usas), device="cuda:0")
    precisions = torch.zeros(len(usas), device="cuda:0")
    num = 0
    for (batch_idx, batch) in tqdm(enumerate(test_data_loader), total=min(num_samples, test_data_loader.__len__())):
        if batch_idx == num_samples:
             break
        with torch.no_grad():        
            output_dict = model(input_ids = batch['input_ids'].to("cuda:1"),
                    attention_mask = batch['attention_mask'].to("cuda:1"),
                    past_key_values=DynamicCache(),
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True)

        total_loss = 0
        losses = []
        for i in range(len(usas)):
            usa = usas[i].to("cuda:0")
            K = output_dict['past_key_values'].key_cache[i].detach().to("cuda:0")
            Q = output_dict['past_key_values'].query_cache[i].detach().to("cuda:0")
            K = repeat_kv(K, Q.shape[1] // K.shape[1])
            sspan = usa(K, Q, hard=True)
            kq_mask = get_kq_mask(sspan)
            raw_attention = torch.matmul(Q, K.transpose(2,3)) / math.sqrt(K.shape[-1])
            raw_attention = raw_attention + (kq_mask - 1.0)*1e35
            target = torch.nn.functional.softmax(raw_attention, dim=-1, dtype=torch.float32).to(Q.dtype)
            target_sspan = compute_target_mask(target, target_mode) * kq_mask

            target = target * kq_mask
            sspan = sspan * kq_mask
            target_sspan = target_sspan[:,:,-num_queries:,:]
            sspan = sspan[:,:,-num_queries:,:]
            kq_mask = kq_mask[:,:,-num_queries:,:]
            T = target_sspan.reshape(-1)[kq_mask.reshape(-1) >  1e-6]
            A = sspan.reshape(-1)[kq_mask.reshape(-1) > 1e-6]

            recall = torch.sum(T[A > 1e-6]) / torch.sum(T)
            precision = torch.sum(A[T > 1e-6]) / torch.sum(A)
            recalls[i] += recall
            
            precisions[i] += precision
            # for j in range(32):
            #     print("layer,att: ",i,j, torch.mean((sspan[0][j][0] == target_sspan[0][j][0]).float()))
            # print("precision", precision, "recall", recall)
            # import pdb
            # pdb.set_trace()

        num+=1

    recalls = recalls / num
    precisions = precisions / num
    print("--- recall ---")
    print(recalls)
    print("--- precisions ---")
    print(precisions)
    return {'recall' : torch.mean(recalls).item(),
            'precision' : torch.mean(precisions).item()}




def train_step(usa, K, Q, target_mode, span_criterion, num_queries, recall_weight):
    # GQA
    K = repeat_kv(K, Q.shape[1] // K.shape[1])
    output = usa(K, Q)
    train_mask = get_kq_mask(output)

    raw_attention_scores = torch.matmul(Q, K.transpose(2,3)) / math.sqrt(K.shape[-1])
    raw_attention_scores = raw_attention_scores + ( train_mask - 1.0 ) * 1e35
    attention_scores = torch.nn.functional.softmax(raw_attention_scores, dim=-1, dtype=torch.float32).to(Q.dtype)

    attention_scores = attention_scores * train_mask
    output = output * train_mask
    target = compute_target_mask(attention_scores, target_mode) * train_mask
    # keep only half queries

    span_loss = span_criterion(torch.nn.functional.relu(output)[:,:,-num_queries:,:], target[:,:,-num_queries:,:], beta=recall_weight)
    return span_loss

def train_usa(model, usas, train_data_loader, test_data_loader, target_mode, optimizer, span_criterion, max_train_itr, num_queries, accumulate_grad, recall_weight,args):
    itr = 0
    for epoch in range(1000):
        for (batch_idx, batch) in tqdm(enumerate(train_data_loader), total=train_data_loader.__len__()):

                # if batch_idx == DEBUG_SAMPLES:
                #     break
                if epoch == 0 and batch_idx < args.skip_examples:
                    continue
                if max_train_itr > 0 and itr > max_train_itr:
                    break
                if itr % 64 == 0:
                    result = evaluate_coverage(model, usas, train_data_loader, target_mode, num_queries=num_queries, num_samples=10)
                    print('TRAIN {}/{} recall:{:.6f} precision:{:.6f}'.format(batch_idx, train_data_loader.__len__(), result['recall'], result['precision']))
                    result = evaluate_coverage(model, usas, test_data_loader, target_mode, num_queries=num_queries, num_samples=100)
                    print('TEST {}/{} recall:{:.6f} precision:{:.6f}'.format(batch_idx, train_data_loader.__len__(), result['recall'], result['precision']))

                    torch.save(usas.cpu().state_dict(), "./artifacts/usa.pt")

                with torch.no_grad():        
                    output_dict = model(input_ids = batch['input_ids'].to("cuda:1"),
                            attention_mask = batch['attention_mask'].to("cuda:1"),
                            past_key_values=DynamicCache(),
                            use_cache=True,
                            output_attentions=True,
                            return_dict=True)

                total_loss = 0
                losses = []
                for i in range(len(usas)):
                    usa = usas[i].to("cuda:0")
                    K = output_dict['past_key_values'].key_cache[i].detach().to("cuda:0")
                    Q = output_dict['past_key_values'].query_cache[i].detach().to("cuda:0")
                    loss = train_step(usa, K, Q, target_mode, span_criterion, num_queries, recall_weight)
                    total_loss += loss
                    losses.append(loss.item())

                total_loss.backward()
                if itr % accumulate_grad == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                print(np.mean(losses), losses, flush=True)
                itr += 1


def data_pipeline(train_data, tokenizer, builder, max_len):
    chat_train_data = train_data.map(lambda instance: {'prompt' : tokenizer.apply_chat_template(builder(instance), tokenize=False, add_generation_prompt=True) })
    encoded_train_data = chat_train_data.map(lambda instance: tokenizer(instance['prompt']))    
    def middle_truncate(instance, length=max_len):
        assert(len(instance['attention_mask']) >= length) #TODO(pad the smaller sequences)
        mid_point = len(instance['attention_mask']) // 2
        excess = len(instance['attention_mask']) - length
        left = mid_point - excess // 2
        return {'input_ids' : instance['input_ids'][:left] + instance['input_ids'][left + excess:],
                'attention_mask' : instance['attention_mask'][:left] + instance['attention_mask'][left + excess:],
                }
    truncated_encoded_train_data = encoded_train_data.map(middle_truncate)
    truncated_encoded_train_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return truncated_encoded_train_data
    
def main():
    parser = argparse.ArgumentParser(description=" Train Script for USA")

    # Add arguments
    parser.add_argument("--dataset", type=str, help="dataset")
    parser.add_argument("--model", type=str, help="model", choices=["llama3-inst"])
    parser.add_argument("--train_epochs", type=int, help="train args: epochs", default=1)
    parser.add_argument("--batch", type=int, help="train args: batchsize", default=1)
    parser.add_argument("--max_length", type=int, help="max length", default=512)
    parser.add_argument("--max_train_itr", type=int, help="max train itr", default=-1)
    parser.add_argument("--device", type=str, help="train args: device", default="cuda")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")

    # usa 
    parser.add_argument("--usa-load", type=str, help="load usa", default=None)
    parser.add_argument("--usa-L", type=int, help="usa L", default=32)
    parser.add_argument("--usa-R", type=int, help="usa R", default=4)
    parser.add_argument("--usa-int-dim", type=int, help="usa int dim", default=256)
    parser.add_argument("--usa-aug-k", type=int, help="usa aug k", default=0)
    parser.add_argument("--usa-target-mode", type=str, help="usa aug k", default="top-10")
    parser.add_argument("--usa-t", type=float, help="usa t", default=0.5)
    parser.add_argument("--accumulate-grad", type=int, help="usa acc grad", default=8)
    parser.add_argument("--num-queries", type=int, help="num queries for training and testing", default=8)
    parser.add_argument("--recall-weight", type=float, help="num queries for training and testing", default=10.0)
    parser.add_argument("--skip-examples", type=int, help="skip examples", default=0)


    args = parser.parse_args()

    assert(torch.cuda.device_count() >= 2)
    # we will put the model inference on device 1 and perform training usas on device 0

    ## Model and prompt stuff
    model, tokenizer = load_model(args.model)
    model = model.to("cuda:1")
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    builder = prompt_builder(args.model, args.dataset)
    
    ## dataset loading
    dataset = load_data(args.dataset)
    train_data = dataset['train']
    test_data = dataset['test']

    truncated_encoded_train_data = data_pipeline(train_data, tokenizer, builder, args.max_length)
    truncated_encoded_test_data = data_pipeline(test_data, tokenizer, builder, args.max_length)

    # chat_train_data = train_data.map(lambda instance: {'prompt' : tokenizer.apply_chat_template(builder(instance), tokenize=False, add_generation_prompt=True) })
    # encoded_train_data = chat_train_data.map(lambda instance: tokenizer(instance['prompt']))    
    # def middle_truncate(instance, length=args.max_length):
    #     assert(len(instance['attention_mask']) >= length) #TODO(pad the smaller sequences)
    #     mid_point = len(instance['attention_mask']) // 2
    #     excess = len(instance['attention_mask']) - length
    #     left = mid_point - excess // 2
    #     return {'input_ids' : instance['input_ids'][:left] + instance['input_ids'][left + excess:],
    #             'attention_mask' : instance['attention_mask'][:left] + instance['attention_mask'][left + excess:],
    #             }
    # truncated_encoded_train_data = encoded_train_data.map(middle_truncate)
    # truncated_encoded_train_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    train_data_loader = torch.utils.data.DataLoader(truncated_encoded_train_data, batch_size=args.batch)
    test_data_loader = torch.utils.data.DataLoader(truncated_encoded_test_data, batch_size=args.batch)


    ## USA loading

    usas = []
    for i in range(len(model.model.layers)):
        usas.append( USA( model.model.layers[0].self_attn.num_heads, 
                        model.model.layers[i].self_attn.head_dim,
                        usa_params = {'L': args.usa_L, 'R': args.usa_R, 'int_dim': args.usa_int_dim, 'aug_k' : 0},
                        annealing_paramters = {'T': 10, 't': args.usa_t}
                        )
                    )
    usas = torch.nn.ModuleList(usas).to("cuda:0")
    if args.usa_load is not None:
        usas.load_state_dict(torch.load(args.usa_load))
    print("t value", usas[0].t, flush=True)
    
    span_criterion = torch.nn.BCELoss() # Define your loss function (e.g., torch.nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(usas.parameters()) # Define your optimizer (e.g., torch.optim.Adam(model.parameters()))
    print(model)
    train_usa(model, usas, train_data_loader, test_data_loader, args.usa_target_mode, optimizer, loss_function, args.max_train_itr, args.num_queries, args.accumulate_grad, args.recall_weight, args)
    torch.save(usas.cpu().state_dict(), "./artifacts/usa-" + args.model + "-config-" + 'L_{}-R_{}-I_{}-mode-{}'.format(args.usa_L, args.usa_R, args.usa_int_dim, args.usa_target_mode))

if __name__ == '__main__':
    main()
